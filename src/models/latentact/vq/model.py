import os
import functools

import torch
import torch.nn as nn
from src.models.latentact.vq.encdec import Encoder, Decoder, TFEncoder, TFDecoder, JointConditioner, FeedForward
from src.models.latentact.vq.residual_vq import ResidualVQ

from src.models.mdm.model.mdm import MDM
from src.models.mdm.diffusion import gaussian_diffusion as gd
from src.models.mdm.diffusion.respace import SpacedDiffusion, space_timesteps
from src.models.mdm.utils.parser_util import get_cond_mode
from src.models.mdm.diffusion.resample import create_named_schedule_sampler


def create_model_and_diffusion(args, data=None):
    model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, data=None):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)

    if data is None:
        num_actions = 1
    else:
        if hasattr(data.dataset, 'num_actions'):
            num_actions = data.dataset.num_actions
        else:
            num_actions = 1

    # SMPL defaults, TODO: check how to modify this for MANO hands
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    # MDM defaults
    pose_rep = 'rot6d'

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1
    elif args.dataset == 'holo' or args.dataset == 'arctic':
        data_rep = 'hml_vec'
        njoints = args.dim_pose
        pose_rep = 'xyz' # this is taken care of separately in the trainer
        data_rep = 'xyz'
        if args.contact_map:
            njoints += 778 # MANO hand vertices
        if args.latent and args.diffusion: # latter has to be true anyway
            njoints = args.latent_dim
        nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': pose_rep, 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset}


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
    )    


class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 **kwargs):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        
        self.opt = args
        
        if self.opt is None or not hasattr(self.opt, 'model_type'):
            model_type = 'conv'
        else:
            model_type = self.opt.model_type
        if 'conv' in model_type:
            self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                                dilation_growth_rate, activation=activation, norm=norm)
            self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                                dilation_growth_rate, activation=activation, norm=norm)
        elif 'tf' in model_type:
            self.encoder = TFEncoder(self.opt)
            self.decoder = TFDecoder(self.opt)
        
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)

        if 'mano' in self.opt.motion_type:
            from common.body_models import MANO, MODEL_DIR as MANO_PATH
            self.mano = MANO(MANO_PATH, use_pca=False, flat_hand_mean=True)
            # self.mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20] # this is from LatentAct which makes predictions in OpenPose convention
            self.mano_to_openpose = [x for x in range(21)] # set to identity since everything is predicted in MANO convention

            if self.opt.coord_sys == 'contact':
                self.unknown_cam_t_contact = nn.Parameter(torch.randn(3))
                self.unknown_cam_t_contact_ref = nn.Parameter(torch.randn(3))

        if self.opt.pred_cam:
            self.unknown_cam_transf = nn.Parameter(torch.randn(6+3)) # this is added

        if self.opt.diffusion:
            self.model, self.diffusion = create_model_and_diffusion(self.opt, data=None)
            self.ddp_model = self.model
            if self.opt.video_feats is not None or self.opt.text_feats or self.opt.contact_grid is not None:
                self.joint_conditioner = JointConditioner(self.opt)
            
            self.schedule_sampler_type = 'uniform'
            self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)

            if self.opt.latent:
                if self.opt.load_indices is None:
                    print (f'No codebook provided, initializing with random codebook')
                else:
                    ckpt_path = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.load_indices)
                    # self.load_codebook(ckpt_path)
                    self.load_vqvae_model(ckpt_path, encoder=True, quantizer=True)

            if not self.opt.joint_vqvae:
                self.freeze_modules(encoder=True, quantizer=True, codebook=True)

        if self.opt.feedforward:
            assert not self.opt.decoder_only
            self.feedforward = FeedForward(self.opt)
            self.freeze_modules(encoder=True, decoder=True, quantizer=True, codebook=True)

        if self.opt.eval_model is not None:
            eval_dataset, eval_model = self.opt.eval_model.split('/')
            ckpt_path = os.path.join(self.opt.checkpoints_dir, eval_dataset, eval_model)
            self.load_vqvae_model(ckpt_path, encoder=False, decoder=True, quantizer=True)
            self.freeze_modules(encoder=True, decoder=True, quantizer=True, codebook=True)
        
        if self.opt.decoder_only:
            ckpt_path = f"{os.environ['DOWNLOADS_DIR']}/model/latentact/checkpoints/last.ckpt"
            self.load_codebook(ckpt_path, mode='latest')
            self.freeze_modules(encoder=True, quantizer=True, codebook=True)

    def load_codebook(self, ckpt_path, mode='finest'):
        if os.path.isdir(ckpt_path):
            ckpt = torch.load(f'{ckpt_path}/model/{mode}.tar', map_location='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')
        codebook_keys = [k for k in ckpt['state_dict'].keys() if 'codebook' in k]
        codebook_ckpt = {k.replace('model.vq_model.quantizer.', ''): ckpt['state_dict'][k] for k in codebook_keys}
        self.quantizer.load_state_dict(codebook_ckpt)

    def load_vqvae_model(self, ckpt_path, encoder=False, decoder=False, quantizer=False, mode='finest', feedforward=False, diffusion=False):
        if os.path.isdir(ckpt_path):
            ckpt = torch.load(f'{ckpt_path}/model/{mode}.tar', map_location='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')
        # load encoder, decoder, quantizer
        if encoder:
            encoder_keys = [k for k in ckpt['state_dict'].keys() if 'encoder' in k]
            encoder_ckpt = {k.replace('model.vq_model.encoder.', ''): ckpt['state_dict'][k] for k in encoder_keys}
            self.encoder.load_state_dict(encoder_ckpt)
        if decoder:
            decoder_keys = [k for k in ckpt['state_dict'].keys() if 'decoder' in k]
            decoder_ckpt = {k.replace('model.vq_model.decoder.', ''): ckpt['state_dict'][k] for k in decoder_keys}
            self.decoder.load_state_dict(decoder_ckpt)
        if feedforward:
            ff_keys = [k for k in ckpt['state_dict'].keys() if 'feedforward' in k]
            ff_ckpt = {k.replace('model.vq_model.feedforward.', ''): ckpt['state_dict'][k] for k in ff_keys}
            self.feedforward.load_state_dict(ff_ckpt)
        if diffusion:
            model_keys = [k for k in ckpt['state_dict'].keys() if 'ddp_model' in k]
            model_ckpt = {k.replace('model.vq_model.ddp_model.', ''): ckpt['state_dict'][k] for k in model_keys}
            self.ddp_model.load_state_dict(model_ckpt)
        if quantizer:
            self.load_codebook(ckpt_path)
    
    def freeze_modules(self, encoder=False, decoder=False, quantizer=False, codebook=False):
        if encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
        if quantizer: # not required since quantizer does not have any parameters
            for param in self.quantizer.parameters():
                param.requires_grad = False
        if codebook:
            for i in range(self.opt.num_quantizers):
                self.quantizer.layers[i].codebook.requires_grad = False
    
    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x, **kwargs):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True, **kwargs)
        return code_idx, all_codes

    def forward(self, x, **kwargs):
        bz, ts, ch = x.shape
        more_outs = {}
        if not self.opt.decoder_only and not self.opt.feedforward:
            if ch == self.opt.dim_pose:
                x = x.reshape(bz, -1, self.opt.dim_pose)
            else:
                x = x.reshape(bz, ts, -1)
            x_in = self.preprocess(x)
            # Encode
            x_encoder = self.encoder(x_in)

            # quantization
            x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=self.opt.sample_codebook_temp)
        
        else:
            commit_loss = torch.tensor(0, dtype=torch.float32).to(x.device)
            perplexity = torch.tensor(0, dtype=torch.float32).to(x.device)
            
            if self.opt.feedforward: # dummy values
                x_quantized = None
                code_idx = None
            else:
                # load code_idx from kwargs
                code_idx = kwargs.get('code_idx', None)
                assert code_idx is not None
                # get embeddings from quantizer
                x_quantized = self.quantizer.get_codebook_entry(code_idx.reshape(bz, ts, -1))

        if self.opt.diffusion:
            if self.opt.latent or self.opt.joint_vqvae:
                micro = x_quantized.unsqueeze(2) # (bs, ts, ch) -> (bs, ch, 1, ts) # follow MDM convention
            else:
                micro = x.transpose(1, 2).unsqueeze(2) # (bs, ts, ch) -> (bs, ch, 1, ts) # follow MDM convention
            cond = {}
            cond['y'] = {'mask': torch.ones(bz, 1, 1, ts).to(x.device)} # follow MDM convention
            joint_condition = self.joint_conditioner(x, **kwargs)
            cond['y']['joint'] = joint_condition # incorporate this in the training loss
            micro_cond = cond
            
            if self.training:
                t, weights = self.schedule_sampler.sample(micro.shape[0], x.device)
                
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,  # [bs, ch, image_size, image_size]
                    t,  # [bs](int) sampled timesteps
                    model_kwargs=micro_cond,
                    dataset=None, # self.data.dataset, this shouldn't be required
                )

                losses = compute_losses()
            
            else:
                # loop over all 1000 timesteps, check how slow this is
                total_iters = len(self.schedule_sampler.weights())
                micro = torch.rand_like(micro) # start from random noise
                for idx in range(0, total_iters, 10):
                    t, weights = self.schedule_sampler.sample(micro.shape[0], x.device, index=idx)
                    compute_losses = functools.partial(
                        self.diffusion.training_losses,
                        self.ddp_model,
                        micro,  # [bs, ch, image_size, image_size]
                        t,  # [bs](int) sampled timesteps
                        model_kwargs=micro_cond,
                        dataset=None, # self.data.dataset, this shouldn't be required
                    )

                    losses = compute_losses()

                    micro = losses['model_output']
            
            loss = losses['l2_loss'].squeeze().transpose(1,2) * weights.view(-1, 1, 1)
            x_out = losses['model_output'].squeeze().transpose(1, 2)
            more_outs['diff_mse'] = loss

            if self.opt.latent:
                x_out = self.decoder(x_out.transpose(1,2), **kwargs)
                x_out = x_out.reshape(bz, ts, -1) # (bs, T, Jx3)
        
        elif self.opt.feedforward:
            x_out = self.feedforward(x_quantized, **kwargs)
            x_out = x_out.reshape(bz, ts, -1)
        
        else:
            ## decoder
            x_out = self.decoder(x_quantized, **kwargs)
            x_out = x_out.reshape(bz, ts, -1) # (bs, T, Jx3)
        
        if self.opt.return_indices:
            more_outs['code_idx'] = code_idx

        return x_out, commit_loss, perplexity, more_outs

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        x = x_d.sum(dim=0).permute(0, 2, 1)

        # decoder
        x_out = self.decoder(x)
        return x_out

class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )

        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)