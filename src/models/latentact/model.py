import torch
import torch.nn as nn
import pytorch3d.transforms.rotation_conversions as rot_conv

import common.ld_utils as ld_utils
import common.data_utils as data_utils
import common.torch_utils as torch_utils
import common.transforms as tf
from common.xdict import xdict
from src.nets.hand_heads.mano_head import MANOMetricHead
from src.models.latentact.vq.model import RVQVAE


class LatentAct(nn.Module):
    def __init__(self, focal_length, img_res, args):
        super().__init__()
        self.args = args
        opt = args
        self.opt = opt

        # TODO: move this to a common location
        self.joints_num = 16 # this is for # joints in MANO params TODO: disambiguate this with opt.joints_num used in VQVAE
        self.focal_length = focal_length
        if self.args.rot_rep == 'rot6d':
            self.rot_dim = 6
        elif self.args.rot_rep == 'axis_angle':
            self.rot_dim = 3
        elif self.args.rot_rep == 'rotmat':
            self.rot_dim = 9
        else:
            raise NotImplementedError

        if 'joints' in opt.motion_type:
            opt.joints_num = 21
            if 'tf' in opt.model_type:
                opt.dim_pose = 3
            else:
                opt.dim_pose = 63
        elif 'mano' in opt.motion_type:
            opt.joints_num = 21
            # opt.dim_pose = 48+10+3 # 48 for thetas, 10 for betas, 3 for cam_t, this was used in LatentAct
            opt.dim_pose = 2 * (16 * self.rot_dim + 3) # 16 joints, 3 for translation, both hands
            if opt.coord_sys == 'contact':
                opt.dim_pose += 3 # 3 for translation to contact centroid at current timestep
                opt.dim_pose += 3 # 3 for translation component of transformation to contact centroid at reference timestep
            
            if opt.pred_cam: # are additional params needed or can these can be subsumed into mano params (doesn't work for well)
                opt.dim_pose += 6 # for cam rotation
                opt.dim_pose += 3 # for cam_t
        else:
            raise ValueError('Unknown motion type')

        self.vq_model = RVQVAE(opt,
                opt.dim_pose,
                opt.nb_code,
                opt.code_dim,
                opt.code_dim,
                opt.down_t,
                opt.stride_t,
                opt.width,
                opt.depth,
                opt.dilation_growth_rate,
                opt.vq_act,
                opt.vq_norm,
                opt=opt)
        
        self.mano_r = MANOMetricHead(is_rhand=True, args=args)
        self.mano_l = MANOMetricHead(is_rhand=False, args=args)

        self.frame_of_ref = self.args.get('frame_of_ref', 'view')

        self.img_res = img_res

        self.normalize_transl = self.args.get('normalize_transl', False)
        # mean, std of transl
        self.mean_transl = torch.tensor(data_utils.MEAN_TRANSL, dtype=torch.float32)
        self.std_transl = torch.tensor(data_utils.STD_TRANSL, dtype=torch.float32)

    def forward(self, inputs, meta_info, targets=None, **kwargs):
        bz = meta_info['intrinsics'].shape[0]
        self.device = device = meta_info['intrinsics'].device
        
        if self.frame_of_ref == 'residual':
            residual_pose_r, residual_pose_l = targets['future.residual.pose.r'], targets['future.residual.pose.l']
            residual_transl_r, residual_transl_l = targets['future.residual.transl.r'], targets['future.residual.transl.l']
        elif self.frame_of_ref == 'view':
            residual_pose_r, residual_pose_l = targets['future.view.pose.r'], targets['future.view.pose.l']
            residual_transl_r, residual_transl_l = targets['future.view.transl.r'], targets['future.view.transl.l']
        elif self.frame_of_ref == 'mano':
            residual_pose_r, residual_pose_l = targets['future.mano.pose.r'], targets['future.mano.pose.l']
            residual_transl_r, residual_transl_l = targets['future.mano.transl.r'], targets['future.mano.transl.l']

        if self.frame_of_ref == 'view' and self.normalize_transl:
            # normalize translation vectors
            residual_transl_r = (residual_transl_r - self.mean_transl.view(1,1,-1).to(device)) / self.std_transl.view(1,1,-1).to(device)
            residual_transl_l = (residual_transl_l - self.mean_transl.view(1,1,-1).to(device)) / self.std_transl.view(1,1,-1).to(device)
        
        bz, ts = residual_pose_r.shape[:2]
        if self.args.rot_rep == 'rot6d':
            # convert to 6d representation
            residual_pose_r = rot_conv.axis_angle_to_matrix(residual_pose_r.reshape(bz*ts, -1, 3))
            residual_pose_r = rot_conv.matrix_to_rotation_6d(residual_pose_r).reshape(bz, ts, -1)
            residual_pose_l = rot_conv.axis_angle_to_matrix(residual_pose_l.reshape(bz*ts, -1, 3))
            residual_pose_l = rot_conv.matrix_to_rotation_6d(residual_pose_l).reshape(bz, ts, -1)
        elif self.args.rot_rep == 'rotmat':
            residual_pose_r = rot_conv.axis_angle_to_matrix(residual_pose_r.reshape(bz*ts, -1, 3)).reshape(bz, ts, -1)
            residual_pose_l = rot_conv.axis_angle_to_matrix(residual_pose_l.reshape(bz*ts, -1, 3)).reshape(bz, ts, -1)

        motion = torch.cat([residual_pose_r, residual_pose_l], dim=-1)
        # append residual translation as well
        motion = torch.cat([motion, residual_transl_r, residual_transl_l], dim=-1)
        
        more_dict = self.prepare_inputs(motion, inputs)

        motions = motion
        pred_motion, loss_commit, perplexity, more_outs = self.vq_model(motions, **more_dict)

        motion_out = pred_motion
        pred_residual_pose_r, pred_residual_transl_r, pred_residual_pose_l, pred_residual_transl_l = self.process_motion_output(motion_out)
        # TODO: move relevant keys from targets to meta_info
        K = meta_info['intrinsics'][:, -1:].repeat(1, ts, 1, 1)
        mano_output_r, mano_output_l = self.run_mano_on_pose_predictions(
            pred_residual_pose_r, pred_residual_transl_r,
            pred_residual_pose_l, pred_residual_transl_l,
            targets, K,
        )

        mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
        mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
        
        mano_output_r['future.j3d.cam.r'] = mano_output_r.pop('mano.j3d.cam.r')
        mano_output_r['future.v3d.cam.r'] = mano_output_r.pop('mano.v3d.cam.r')
        mano_output_l['future.j3d.cam.l'] = mano_output_l.pop('mano.j3d.cam.l')
        mano_output_l['future.v3d.cam.l'] = mano_output_l.pop('mano.v3d.cam.l')

        # also pop j2d.norm keys
        mano_output_r['future.j2d.norm.r'] = mano_output_r.pop('mano.j2d.norm.r')
        mano_output_l['future.j2d.norm.l'] = mano_output_l.pop('mano.j2d.norm.l')

        output = xdict()
        output.merge(mano_output_r)
        output.merge(mano_output_l)

        output = torch_utils.expand_dict_dims(output, curr_dim=0, dims=(bz, -1))

        # store residual poses as well
        more_outs['future.residual.pose.r'] = pred_residual_pose_r
        more_outs['future.residual.transl.r'] = pred_residual_transl_r
        more_outs['future.residual.pose.l'] = pred_residual_pose_l
        more_outs['future.residual.transl.l'] = pred_residual_transl_l
        more_outs['loss_commit'] = loss_commit
        more_outs['loss_perplexity'] = perplexity # not used in loss_computation
        more_outs = xdict(more_outs)

        output.merge(more_outs)

        return output

    def process_motion_output(self, motion_out):
        bz, ts = motion_out.shape[:2]
        # convert rotations back to axis angle
        pred_residual_pose_r = motion_out[:,:,:(self.joints_num*self.rot_dim)]
        pred_residual_pose_l = motion_out[:,:,(self.joints_num*self.rot_dim):-6] # last 6 are for translation terms
        # convert pose to axis angle
        if self.args.rot_rep == 'rot6d':
            pred_residual_pose_r = rot_conv.rotation_6d_to_matrix(pred_residual_pose_r.reshape(bz*ts, -1, 6))
            pred_residual_pose_r = rot_conv.matrix_to_axis_angle(pred_residual_pose_r).reshape(bz, ts, -1)
            pred_residual_pose_l = rot_conv.rotation_6d_to_matrix(pred_residual_pose_l.reshape(bz*ts, -1, 6))
            pred_residual_pose_l = rot_conv.matrix_to_axis_angle(pred_residual_pose_l).reshape(bz, ts, -1)
        elif self.args.rot_rep == 'rotmat':
            pred_residual_pose_r = rot_conv.matrix_to_axis_angle(pred_residual_pose_r.reshape(bz*ts, -1, 3, 3)).reshape(bz, ts, -1)
            pred_residual_pose_l = rot_conv.matrix_to_axis_angle(pred_residual_pose_l.reshape(bz*ts, -1, 3, 3)).reshape(bz, ts, -1)
        
        pred_residual_transl_r = motion_out[:,:,-6:-3] # [bs, ts, ch]
        pred_residual_transl_l = motion_out[:,:,-3:] # [bs, ts, ch]

        # unnormalize translation vectors if required
        if self.frame_of_ref == 'view' and self.normalize_transl:
            device = pred_residual_transl_r.device
            pred_residual_transl_r = (pred_residual_transl_r * self.std_transl.view(1,1,-1).to(device)) + self.mean_transl.view(1,1,-1).to(device)
            pred_residual_transl_l = (pred_residual_transl_l * self.std_transl.view(1,1,-1).to(device)) + self.mean_transl.view(1,1,-1).to(device)

        return pred_residual_pose_r, pred_residual_transl_r, pred_residual_pose_l, pred_residual_transl_l
    
    def run_mano_on_pose_predictions(self, pred_residual_pose_r, pred_residual_transl_r, 
                                 pred_residual_pose_l, pred_residual_transl_l, targets, K):
        bz, ts = pred_residual_pose_r.shape[:2]

        if self.frame_of_ref == 'residual':
            mano_output_r, mano_output_l = self.convert_to_reference_frame(
                pred_residual_pose_r, pred_residual_transl_r,
                pred_residual_pose_l, pred_residual_transl_l,
                targets
            )
        elif self.frame_of_ref == 'view':
            mano_output_r = self.mano_r(
                rotmat = pred_residual_pose_r.reshape(bz * ts, -1),
                shape=targets['future_betas_r'].reshape(bz * ts, -1),
                cam_t=pred_residual_transl_r.reshape(bz * ts, -1),
                K=K.reshape(bz * ts, 3, 3),
            )

            mano_output_l = self.mano_l(
                rotmat = pred_residual_pose_l.reshape(bz * ts, -1),
                shape=targets['future_betas_l'].reshape(bz * ts, -1),
                cam_t=pred_residual_transl_l.reshape(bz * ts, -1),
                K=K.reshape(bz * ts, 3, 3),
            )

        elif self.frame_of_ref == 'mano':
            mano2view = targets['mano2view'] # [bs, 4, 4]
            mano2view = mano2view.unsqueeze(1).repeat(1, ts, 1, 1).view(-1, 4, 4)
            pred_residual_pose_r, pred_residual_transl_r = tf.get_view_transform(
                pred_residual_pose_r, pred_residual_transl_r, mano2view
            )
            pred_residual_pose_l, pred_residual_transl_l = tf.get_view_transform(
                pred_residual_pose_l, pred_residual_transl_l, mano2view
            )

            mano_output_r = self.mano_r(
                rotmat = pred_residual_pose_r.reshape(bz * ts, -1),
                shape=targets['future_betas_r'].reshape(bz * ts, -1),
                cam_t=pred_residual_transl_r.reshape(bz * ts, -1),
                K=K.reshape(bz * ts, 3, 3),
            )

            mano_output_l = self.mano_l(
                rotmat = pred_residual_pose_l.reshape(bz * ts, -1),
                shape=targets['future_betas_l'].reshape(bz * ts, -1),
                cam_t=pred_residual_transl_l.reshape(bz * ts, -1),
                K=K.reshape(bz * ts, 3, 3),
            )

        return mano_output_r, mano_output_l
    
    def sample(self, inputs, meta_info, targets=None, **kwargs): # to be consistent with mdm wrapper
        bz = meta_info['intrinsics'].shape[0]
        self.device = device = meta_info['intrinsics'].device
        
        if self.frame_of_ref == 'residual':
            residual_pose_r, residual_pose_l = targets['future.residual.pose.r'], targets['future.residual.pose.l']
            residual_transl_r, residual_transl_l = targets['future.residual.transl.r'], targets['future.residual.transl.l']
        elif self.frame_of_ref == 'view':
            residual_pose_r, residual_pose_l = targets['future.view.pose.r'], targets['future.view.pose.l']
            residual_transl_r, residual_transl_l = targets['future.view.transl.r'], targets['future.view.transl.l']
        elif self.frame_of_ref == 'mano':
            residual_pose_r, residual_pose_l = targets['future.mano.pose.r'], targets['future.mano.pose.l']
            residual_transl_r, residual_transl_l = targets['future.mano.transl.r'], targets['future.mano.transl.l']

        if self.frame_of_ref == 'view' and self.normalize_transl:
            # normalize translation vectors
            residual_transl_r = (residual_transl_r - self.mean_transl.view(1,1,-1).to(device)) / self.std_transl.view(1,1,-1).to(device)
            residual_transl_l = (residual_transl_l - self.mean_transl.view(1,1,-1).to(device)) / self.std_transl.view(1,1,-1).to(device)
        
        bz, ts = residual_pose_r.shape[:2]
        if self.args.rot_rep == 'rot6d':
            # convert to 6d representation
            residual_pose_r = rot_conv.axis_angle_to_matrix(residual_pose_r.reshape(bz*ts, -1, 3))
            residual_pose_r = rot_conv.matrix_to_rotation_6d(residual_pose_r).reshape(bz, ts, -1)
            residual_pose_l = rot_conv.axis_angle_to_matrix(residual_pose_l.reshape(bz*ts, -1, 3))
            residual_pose_l = rot_conv.matrix_to_rotation_6d(residual_pose_l).reshape(bz, ts, -1)
        elif self.args.rot_rep == 'rotmat':
            residual_pose_r = rot_conv.axis_angle_to_matrix(residual_pose_r.reshape(bz*ts, -1, 3)).reshape(bz, ts, -1)
            residual_pose_l = rot_conv.axis_angle_to_matrix(residual_pose_l.reshape(bz*ts, -1, 3)).reshape(bz, ts, -1)

        motion = torch.cat([residual_pose_r, residual_pose_l], dim=-1)
        # append residual translation as well
        motion = torch.cat([motion, residual_transl_r, residual_transl_l], dim=-1)
        
        more_dict = self.prepare_inputs(motion, inputs, mode='val')

        motions = motion
        pred_motion, loss_commit, perplexity, more_outs = self.vq_model(motions, **more_dict)

        return pred_motion
    
    def prepare_inputs(self, motions, batch_data, mode='train'):
        if self.opt.pred_cam:
            if self.opt.coord_sys == 'contact':
                if self.opt.joints_loss:
                    cam_rot_nondiff = motions[..., -6:].detach().to(self.device).float()
                    cam_rot = motions[..., -6:].to(self.device).float()
                    if mode == 'train':
                        cam_rot.requires_grad = True
                    mano_rot = motions[..., :3].to(self.device).float()
                    if mode == 'train':
                        mano_rot.requires_grad = True
                else:
                    cam_rot = motions[..., -9:-3].detach().to(self.device).float()
                    cam_rot_nondiff = cam_rot
                    cam_transl = motions[..., -3:].detach().to(self.device).float()
                    motions = motions[..., :-9]
            else:
                cam_rot = motions[..., -9:-3].detach().to(self.device).float()
                cam_rot_nondiff = cam_rot
                cam_transl = motions[..., -3:].detach().to(self.device).float()
                motions = motions[..., :-9]
        
        more_dict = {}
        if self.opt.video_feats:
            video_feats = batch_data['video_feats'].detach().to(self.device).float()
            more_dict['video_feats'] = video_feats
        if self.opt.text_feats:
            text_feats = batch_data['text_feats'].detach().to(self.device).float()
            more_dict['text_feats'] = text_feats
        if self.opt.contact_grid is not None:
            contact = batch_data['contact'].detach().to(self.device).float()
            more_dict['contact'] = contact
            grid = batch_data['grid'].detach().to(self.device).float()
            more_dict['grid'] = grid
            contact_mask = batch_data['contact_mask'].detach().to(self.device).float()
            more_dict['contact_mask'] = contact_mask
            known_mask = contact_mask.reshape(-1,)
            self.known_mask = known_mask
            grid_mask = batch_data['grid_mask'].detach().to(self.device).float()
            more_dict['grid_mask'] = grid_mask

            if self.opt.coord_sys == 'contact':
                cam_t_contact = batch_data['cam_t_contact'].detach().to(self.device).float()
                bz, ts = cam_t_contact.shape[:2]
                known_mask = contact_mask.reshape(-1,)
                cam_t_contact = cam_t_contact.reshape(-1, 3)
                # replace unknown cam_t_contact with learnable parameters
                cam_t_contact[known_mask == 0] = self.vq_model.unknown_cam_t_contact
                cam_t_contact = cam_t_contact.reshape(bz, ts, 3)
                more_dict['cam_t_contact'] = cam_t_contact
                # replace last 3 values of motion with learnable parameters
                motions = torch.cat([motions[..., :61], cam_t_contact], dim=-1)
                self.known_mask = known_mask

                cam_t_contact_ref = batch_data['cam_t_contact_ref'].detach().to(self.device).float()
                bz, ts = cam_t_contact_ref.shape[:2]
                known_mask_ref = batch_data['contact_ref_mask'].detach().to(self.device).float().reshape(-1,)
                cam_t_contact_ref = cam_t_contact_ref.reshape(-1, 3)
                # replace unknown cam_t_contact with learnable parameters
                cam_t_contact_ref[known_mask_ref == 0] = self.vq_model.unknown_cam_t_contact_ref
                cam_t_contact_ref = cam_t_contact_ref.reshape(bz, ts, 3)
                more_dict['cam_t_contact_ref'] = cam_t_contact_ref
                # replace last 3 values of motion with learnable parameters
                motions = torch.cat([motions[..., :64], cam_t_contact_ref], dim=-1)
                self.known_mask_ref = known_mask_ref

        bz, ts = motions.shape[:2]
        trainable_mask = torch.ones((bz, ts, self.opt.dim_pose)).to(self.device)

        if 'cam_mask' in batch_data:
            known_cam_mask = batch_data['cam_mask'].detach().to(self.device).float().reshape(-1,)
        else:
            known_cam_mask = torch.ones((bz, ts)).to(self.device).float().reshape(-1, )
        self.known_cam_mask = known_cam_mask

        if self.opt.pred_cam:
            if not self.opt.coord_sys == 'contact':
                cam_transf = torch.cat([cam_rot_nondiff, cam_transl], dim=-1)
            else:
                cam_transf = torch.cat([cam_rot_nondiff, cam_transl], dim=-1) 
            
            bz, ts = cam_transf.shape[:2]
            cam_transf = cam_transf.reshape(bz*ts, -1)
            
            # # replace unknown cam_transf with learnable parameters
            cam_transf[known_cam_mask == 0] = self.vq_model.unknown_cam_transf

            # append cam_transf to motions as last 9 values
            cam_transf = cam_transf.reshape(bz, ts, -1)
            
            motions = torch.cat([motions, cam_transf], dim=-1)

            if self.opt.joints_loss:
                # add trainable rotation parameters separately
                trainable_mask[..., :3] = 0
                if not self.opt.coord_sys == 'contact':
                    trainable_mask[..., -9:-3] = 0
                    motions = torch.cat([mano_rot, motions[..., 3:-9], cam_rot, cam_transf[...,-3:]], dim=-1)
                else:
                    trainable_mask[..., -6:] = 0
                    trainable_mask[..., 58:61] = 0
                    motions = torch.cat([mano_rot, motions[..., 3:-6], cam_rot], dim=-1)

        trainable_mask = trainable_mask * known_cam_mask.reshape(bz, ts, -1)
        if self.opt.coord_sys == 'contact':
            trainable_mask = trainable_mask * known_mask.reshape(bz, ts, -1)* known_mask_ref.reshape(bz, ts, -1)
        
        if self.opt.contact_map:
            gt_contact_map = batch_data['contact_map'].detach().to(self.device).float()
            motions = torch.cat([motions, gt_contact_map], dim=-1)

        if self.opt.decoder_only:
            kwargs = {}
            if mode != 'train':
                kwargs = {'temperature': 1, 'diversity': True}
            code_idx, _ = self.vq_model.encode(motions, **kwargs)
            more_dict['code_idx'] = code_idx.detach().to(self.device).long()

        return more_dict
    
    def get_target_motion(self, targets, meta_info):
        bz = meta_info['intrinsics'].shape[0]
        device = meta_info['intrinsics'].device

        if self.frame_of_ref == 'residual':
            residual_pose_r, residual_pose_l = targets['future.residual.pose.r'], targets['future.residual.pose.l']
            residual_transl_r, residual_transl_l = targets['future.residual.transl.r'], targets['future.residual.transl.l']
        elif self.frame_of_ref == 'view':
            residual_pose_r, residual_pose_l = targets['future.view.pose.r'], targets['future.view.pose.l']
            residual_transl_r, residual_transl_l = targets['future.view.transl.r'], targets['future.view.transl.l']
        elif self.frame_of_ref == 'mano':
            residual_pose_r, residual_pose_l = targets['future.mano.pose.r'], targets['future.mano.pose.l']
            residual_transl_r, residual_transl_l = targets['future.mano.transl.r'], targets['future.mano.transl.l']

        if self.frame_of_ref == 'view' and self.normalize_transl:
            # normalize translation vectors
            residual_transl_r = (residual_transl_r - self.mean_transl.view(1,1,-1).to(device)) / self.std_transl.view(1,1,-1).to(device)
            residual_transl_l = (residual_transl_l - self.mean_transl.view(1,1,-1).to(device)) / self.std_transl.view(1,1,-1).to(device)
        
        bz, ts = residual_pose_r.shape[:2]
        if self.args.rot_rep == 'rot6d':
            # convert to 6d representation
            residual_pose_r = rot_conv.axis_angle_to_matrix(residual_pose_r.reshape(bz*ts, -1, 3))
            residual_pose_r = rot_conv.matrix_to_rotation_6d(residual_pose_r).reshape(bz, ts, -1)
            residual_pose_l = rot_conv.axis_angle_to_matrix(residual_pose_l.reshape(bz*ts, -1, 3))
            residual_pose_l = rot_conv.matrix_to_rotation_6d(residual_pose_l).reshape(bz, ts, -1)
        elif self.args.rot_rep == 'rotmat':
            residual_pose_r = rot_conv.axis_angle_to_matrix(residual_pose_r.reshape(bz*ts, -1, 3)).reshape(bz, ts, -1)
            residual_pose_l = rot_conv.axis_angle_to_matrix(residual_pose_l.reshape(bz*ts, -1, 3)).reshape(bz, ts, -1)

        motion = torch.cat([residual_pose_r, residual_pose_l], dim=-1)
        # append residual translation as well
        motion = torch.cat([motion, residual_transl_r, residual_transl_l], dim=-1)

        return motion