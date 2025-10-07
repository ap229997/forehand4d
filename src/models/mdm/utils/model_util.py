import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args, data):
    model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    if data is not None and hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    dataset = args.get('dataset', 'arctic_ego')
    if dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1
    elif 'arctic' in dataset or 'h2o' in dataset or 'h2o3d' in dataset or 'dexycb' in dataset \
          or 'hot3d' in dataset or 'assembly' in dataset or 'holo' in dataset or 'epic' in dataset or 'egoexo' in dataset:
        data_rep = args.get('rot_rep', 'rot6d')
        if data_rep == 'rot6d':
            njoints = 15 * 6 + 6 + 3 # mano local pose + global orient + root translation
        elif data_rep == 'axis_angle':
            njoints = 15 * 3 + 3 + 3
        elif data_rep == 'rotmat':
            njoints = 15 * 9 + 9 + 3
        else:
            raise NotImplementedError
        njoints *= 2 # left and right hands
        nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': args.tf_dropout, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': dataset,
            'num_tokens': args.max_motion_length, 'use_diff_2d': args.get('use_diff_2d', False)}


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
