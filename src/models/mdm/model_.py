import functools

import torch
import torch.nn as nn
import pytorch3d.transforms.rotation_conversions as rot_conv

import common.ld_utils as ld_utils
import common.data_utils as data_utils
import common.torch_utils as torch_utils
import common.transforms as tf
from common.xdict import xdict
from src.nets.hand_heads.mano_head import MANOMetricHead
from src.utils.loss_modules import (
    hand_kp3d_loss,
    joints_loss,
    mse_loss
)
from src.callbacks.loss.loss_function import mean_loss_mask

from src.models.mdm.diffusion.resample import create_named_schedule_sampler
from src.models.mdm.utils.model_util import create_model_and_diffusion
from src.models.mdm.diffusion.resample import LossAwareSampler
from src.models.mdm.model.encoder import *


class MotionDiffusion(nn.Module):
    def __init__(self, focal_length, img_res, args):
        super().__init__()
        self.args = args
        
        self.model, self.diffusion = create_model_and_diffusion(args, data=None)
        self.ddp_model = self.model
        
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)

        self.frame_of_ref = self.args.get('frame_of_ref', 'view')

        # define conditioning modules here
        self.cond_mode = self.args.get('cond_mode', 'no_cond')
        if self.cond_mode != 'no_cond':
            more_args = {}
            if 'pose' in self.cond_mode:
                more_args['pose_dim'] = self.model.njoints
            if 'img' in self.cond_mode:
                more_args['img_feat'] = args.get('img_feat', 'vit')
            self.conditioner = JointConditioner(args, self.model.latent_dim, **more_args)
            if 'spatial' in self.cond_mode:
                self.spatial_conditioner = SpatialConditioner(
                    args,
                    latent_dim=self.model.latent_dim,
                    img_feat=args.get('img_feat', 'vit'),
                )
            
            # these are for the lifting model
            if 'future_j2d' in self.cond_mode:
                self.kps_conditioner = KpsConditioner(args, self.model.latent_dim, **more_args)
            if 'future_cam' in self.cond_mode: # or 'future_plucker' in self.cond_mode or 'future_kpe' in self.cond_mode:
                self.cam_conditioner = CameraConditioner(args, self.model.latent_dim, **more_args)
            if 'future_plucker' in self.cond_mode:
                self.plucker_conditioner = CameraConditioner(args, self.model.latent_dim, use_plucker=True, **more_args)
            if 'future_kpe' in self.cond_mode:
                self.kpe_conditioner = CameraConditioner(args, self.model.latent_dim, use_kpe=True, **more_args)

        self.mano_r = MANOMetricHead(is_rhand=True, args=args)
        self.mano_l = MANOMetricHead(is_rhand=False, args=args)

        self.img_res = img_res
        
        # TODO: move this to a common location
        self.joints_num = 16
        self.focal_length = focal_length
        if self.model.data_rep == 'rot6d':
            self.rot_dim = 6
        elif self.model.data_rep == 'axis_angle':
            self.rot_dim = 3
        elif self.model.data_rep == 'rotmat':
            self.rot_dim = 9
        else:
            raise NotImplementedError
        
        self.normalize_transl = self.args.get('normalize_transl', False)
        # mean, std of transl
        self.mean_transl = torch.tensor(data_utils.MEAN_TRANSL, dtype=torch.float32)
        self.std_transl = torch.tensor(data_utils.STD_TRANSL, dtype=torch.float32)

    def forward(self, inputs, meta_info, targets=None, **kwargs):
        bz = meta_info['intrinsics'].shape[0]
        device = meta_info['intrinsics'].device

        # sampling is done by self.sample() separately during inference
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
        if self.model.data_rep == 'rot6d':
            # convert to 6d representation
            residual_pose_r = rot_conv.axis_angle_to_matrix(residual_pose_r.reshape(bz*ts, -1, 3))
            residual_pose_r = rot_conv.matrix_to_rotation_6d(residual_pose_r).reshape(bz, ts, -1)
            residual_pose_l = rot_conv.axis_angle_to_matrix(residual_pose_l.reshape(bz*ts, -1, 3))
            residual_pose_l = rot_conv.matrix_to_rotation_6d(residual_pose_l).reshape(bz, ts, -1)
        elif self.model.data_rep == 'rotmat':
            residual_pose_r = rot_conv.axis_angle_to_matrix(residual_pose_r.reshape(bz*ts, -1, 3)).reshape(bz, ts, -1)
            residual_pose_l = rot_conv.axis_angle_to_matrix(residual_pose_l.reshape(bz*ts, -1, 3)).reshape(bz, ts, -1)

        motion = torch.cat([residual_pose_r, residual_pose_l], dim=-1)
        # append residual translation as well
        motion = torch.cat([motion, residual_transl_r, residual_transl_l], dim=-1)
        motion = motion.permute(0, 2, 1).unsqueeze(2) # (bz, ch, 1, ts) # follow MDM convention
        
        cond = {'y': {}}
        cond['y']['mask'] = meta_info['mask_timesteps'].reshape(bz, 1, 1, -1) # follow MDM convention
        cond['y']['lengths'] = meta_info['lengths']

        if self.cond_mode != 'no_cond':
            cond = self.process_all_conditions(inputs, cond, meta_info=meta_info, targets=targets) # targets is only used when conditioning on future_j2d

        micro = motion
        micro_cond = cond
        t, weights = self.schedule_sampler.sample(micro.shape[0], micro.device)

        compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
            )
        
        losses = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

        more_outs = xdict()
        loss = (losses["loss"] * weights)
        
        model_out = losses['model_output'] # [bs, ch, 1, ts]
        motion_out = model_out[:,:,0].transpose(1, 2) # [bs, ch, 1, ts] -> [bs, ts, ch]

        # mask the loss properly
        is_valid = targets['is_valid'][:, -1]
        loss_mask_r = is_valid * targets['right_valid'][:, -1]
        loss_mask_l = is_valid * targets['left_valid'][:, -1]
        loss_mask_j3d_r = loss_mask_r * meta_info['is_j3d_loss'][:, -1]
        loss_mask_j3d_l = loss_mask_l * meta_info['is_j3d_loss'][:, -1]
        loss_mask_camt_r = loss_mask_r * meta_info['is_cam_loss'][:, -1]
        loss_mask_camt_l = loss_mask_l * meta_info['is_cam_loss'][:, -1]
        
        # more_outs['diff_mse'] = loss
        more_outs['diff_mse_global_r'] = loss_mask_j3d_r * self.diffusion.masked_l2(model_out[:,:self.rot_dim], motion[:,:self.rot_dim], cond['y']['mask'])
        more_outs['diff_mse_pose_r'] = loss_mask_j3d_r * self.diffusion.masked_l2(model_out[:,self.rot_dim:self.joints_num*self.rot_dim], motion[:,self.rot_dim:self.joints_num*self.rot_dim], cond['y']['mask'])
        more_outs['diff_mse_global_l'] = loss_mask_j3d_l * self.diffusion.masked_l2(model_out[:,self.joints_num*self.rot_dim:(self.joints_num+1)*self.rot_dim], motion[:,self.joints_num*self.rot_dim:(self.joints_num+1)*self.rot_dim], cond['y']['mask'])
        more_outs['diff_mse_pose_l'] = loss_mask_j3d_l * self.diffusion.masked_l2(model_out[:,(self.joints_num+1)*self.rot_dim:-6], motion[:,(self.joints_num+1)*self.rot_dim:-6], cond['y']['mask'])
        more_outs['diff_mse_transl_r'] = loss_mask_camt_r * self.diffusion.masked_l2(model_out[:,-6:-3], motion[:,-6:-3], cond['y']['mask'])
        more_outs['diff_mse_transl_l'] = loss_mask_camt_l * self.diffusion.masked_l2(model_out[:,-3:], motion[:,-3:], cond['y']['mask'])

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

        output.merge(more_outs)

        return output
    
    def process_motion_output(self, motion_out):
        bz, ts = motion_out.shape[:2]
        # convert rotations back to axis angle
        pred_residual_pose_r = motion_out[:,:,:(self.joints_num*self.rot_dim)]
        pred_residual_pose_l = motion_out[:,:,(self.joints_num*self.rot_dim):-6] # last 6 are for translation terms
        # convert pose to axis angle
        if self.model.data_rep == 'rot6d':
            pred_residual_pose_r = rot_conv.rotation_6d_to_matrix(pred_residual_pose_r.reshape(bz*ts, -1, 6))
            pred_residual_pose_r = rot_conv.matrix_to_axis_angle(pred_residual_pose_r).reshape(bz, ts, -1)
            pred_residual_pose_l = rot_conv.rotation_6d_to_matrix(pred_residual_pose_l.reshape(bz*ts, -1, 6))
            pred_residual_pose_l = rot_conv.matrix_to_axis_angle(pred_residual_pose_l).reshape(bz, ts, -1)
        elif self.model.data_rep == 'rotmat':
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
    
    def sample(self, inputs, meta_info, targets=None, **kwargs):
        bz = meta_info['intrinsics'].shape[0]

        cond = {'y': {}}
        cond['y']['mask'] = meta_info['mask_timesteps'].reshape(bz, 1, 1, -1) # follow MDM convention
        cond['y']['lengths'] = meta_info['lengths']

        if self.cond_mode != 'no_cond':
            cond = self.process_all_conditions(inputs, cond, meta_info=meta_info, targets=targets)

        skip_timesteps = 0
        if self.args.debug:
            skip_timesteps = 990
        model_kwargs = cond
        sample = self.diffusion.p_sample_loop(
            self.model,
            (bz, self.model.njoints, self.model.nfeats, self.args.max_motion_length),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=skip_timesteps,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        motion_out = sample # [bs, ch, 1, ts]
        motion_out = motion_out[:,:,0].transpose(1, 2) # [bs, ch, 1, ts] -> [bs, ts, ch]

        return motion_out
        
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
        if self.model.data_rep == 'rot6d':
            # convert to 6d representation
            residual_pose_r = rot_conv.axis_angle_to_matrix(residual_pose_r.reshape(bz*ts, -1, 3))
            residual_pose_r = rot_conv.matrix_to_rotation_6d(residual_pose_r).reshape(bz, ts, -1)
            residual_pose_l = rot_conv.axis_angle_to_matrix(residual_pose_l.reshape(bz*ts, -1, 3))
            residual_pose_l = rot_conv.matrix_to_rotation_6d(residual_pose_l).reshape(bz, ts, -1)
        elif self.model.data_rep == 'rotmat':
            residual_pose_r = rot_conv.axis_angle_to_matrix(residual_pose_r.reshape(bz*ts, -1, 3)).reshape(bz, ts, -1)
            residual_pose_l = rot_conv.axis_angle_to_matrix(residual_pose_l.reshape(bz*ts, -1, 3)).reshape(bz, ts, -1)

        motion = torch.cat([residual_pose_r, residual_pose_l], dim=-1)
        # append residual translation as well
        motion = torch.cat([motion, residual_transl_r, residual_transl_l], dim=-1)

        return motion
    
    def process_all_conditions(self, inputs, cond, meta_info=None, targets=None):
        if 'pose' in self.cond_mode or 'img' in self.cond_mode:
            cond_feats = self.get_condition_feats(inputs)
            cond['y']['joint'] = cond_feats
        if 'spatial' in self.cond_mode:
            if self.args.get('window_size', 1) == 1:
                cond['y']['spatial'] = self.spatial_conditioner(inputs['img'][:, -1]) # TODO: support for multi-timestep history for all datasets
            else:
                bz, ts, c, h, w = inputs['img'].shape
                all_imgs = inputs['img'].reshape(bz * ts, c, h, w)
                img_feats = self.spatial_conditioner(all_imgs)
                _, nt, fz = img_feats.shape
                cond['y']['spatial'] = torch.sum(img_feats.reshape(bz, ts, nt, fz), dim=1) # sum over timesteps
            if self.args.get('interpolate', False):
                    goal_feats = self.spatial_conditioner(inputs['goal_img'][:, -1])
                    cond['y']['spatial'] += goal_feats
        
        if 'future' in self.cond_mode:
            assert meta_info is not None
            assert targets is not None
            valid_mask = meta_info['mask_timesteps'] # (B, T)
            future_valid_r = targets["future_valid_r"] # (B, T, J)
            future_valid_l = targets["future_valid_l"] # (B, T, J)
            valid_r = future_valid_r * valid_mask.unsqueeze(-1) # only consider valid timesteps
            valid_l = future_valid_l * valid_mask.unsqueeze(-1)
        
        if 'future_j2d' in self.cond_mode:
            # future_j2d is only available during training
            assert targets is not None
            j2d_r = targets['future.j2d.norm.r'] # (B, T, J, 2)
            j2d_l = targets['future.j2d.norm.l']  # (B, T, J, 2)
            
            bz, ts = j2d_r.shape[:2]
            j2d_r = j2d_r * valid_r.unsqueeze(-1)
            j2d_l = j2d_l * valid_l.unsqueeze(-1)
            j2d_r = j2d_r.reshape(bz, ts, -1)  # (B, T, J*2)
            j2d_l = j2d_l.reshape(bz, ts, -1) 
            j2d = torch.cat([j2d_r, j2d_l], dim=-1)
            cond['y']['future_kps2d'] = self.kps_conditioner(j2d)
        if 'future_cam' in self.cond_mode:
            assert targets is not None
            cam_poses = torch.linalg.inv(targets['future2view'])
            bz, ts = cam_poses.shape[:2]
            cam_poses = cam_poses.reshape(bz, ts, -1)
            # mask out camera poses at invalid timesteps, this was not done previously
            cam_poses = cam_poses * valid_mask.unsqueeze(-1)  # [bz, ts, 4*4]
            cond['y']['future_cam'] = self.cam_conditioner(cam_poses)
        if 'future_plucker' in self.cond_mode:
            assert targets is not None
            cam_poses = torch.linalg.inv(targets['future2view'])  # [bz, ts, 4, 4]
            cam_t = cam_poses[:, :, :3, 3]  # [bz, ts, 3]

            if 'assembly' in self.args.val_dataset: # this only works with GT joints
                j3d_r = targets['future.j3d.cam.r']  # [bz, ts, J, 3]
                j3d_l = targets['future.j3d.cam.l']
            else:
                # unproject pixels using intrinsics and mean depth
                # need to normalize the rays at some point otherwise OOD w.r.t train data
                j2d_r = targets['future.j2d.norm.r'] # (B, 1, J, 2)
                j2d_l = targets['future.j2d.norm.l']  # (B, 1, J, 2)
                unnorm_j2d_r = (j2d_r + 1) * self.args.img_res * 0.5 # (B, 1, J, 2)
                unnorm_j2d_l = (j2d_l + 1) * self.args.img_res * 0.5
                intrx = meta_info['intrinsics'][:, -1:]  # (B, 1, 3, 3)
                j3d_r = data_utils.unproject_pixels_batch(unnorm_j2d_r, intrx, depth=self.mean_transl[-1].view(1, 1, 1)) # [bz, ts, J, 3]
                j3d_l = data_utils.unproject_pixels_batch(unnorm_j2d_l, intrx, depth=self.mean_transl[-1].view(1, 1, 1)) # [bz, ts, J, 3]
            
            plucker_r = self.compute_plucker_rays(j3d_r, cam_t) # [bz, ts, J, 6]
            plucker_l = self.compute_plucker_rays(j3d_l, cam_t) # [bz, ts, J, 6]
            bz, ts = plucker_r.shape[:2]
            plucker_r = plucker_r * valid_r.unsqueeze(-1)  # [bz, ts, J, 6]
            plucker_l = plucker_l * valid_l.unsqueeze(-1)  # [bz, ts, J, 6]
            plucker_r = plucker_r.reshape(bz, ts, -1)
            plucker_l = plucker_l.reshape(bz, ts, -1)
            plucker = torch.cat([plucker_r, plucker_l], dim=-1)
            cond['y']['future_plucker'] = self.plucker_conditioner(plucker)
        if 'future_kpe' in self.cond_mode:
            assert targets is not None
            j2d_r = targets['future.j2d.norm.r'] # (B, T, J, 2)
            j2d_l = targets['future.j2d.norm.l']  # (B, T, J, 2)
            unnorm_j2d_r = (j2d_r + 1) * self.args.img_res * 0.5 # (B, T, J, 2)
            unnorm_j2d_l = (j2d_l + 1) * self.args.img_res * 0.5
            intrx = meta_info['intrinsics'][:, -1:]  # (B, 1, 3, 3)
            kpe_freq = self.args.get('kpe_freq', 4)
            pos_enc_r = self.compute_kpe_enc(unnorm_j2d_r, intrx, freq=kpe_freq)
            pos_enc_l = self.compute_kpe_enc(unnorm_j2d_l, intrx, freq=kpe_freq)
            bz, ts = unnorm_j2d_r.shape[:2]
            pos_enc_r = pos_enc_r * valid_r.unsqueeze(-1) # [bz, ts, J, 2]
            pos_enc_l = pos_enc_l * valid_l.unsqueeze(-1)  # [bz, ts, J, 2]
            pos_enc = torch.cat([pos_enc_r, pos_enc_l], dim=-1)  # (B, T, J, 2*freq*2) # x, y angles for each joint
            cond['y']['future_kpe']= self.kpe_conditioner(pos_enc.reshape(bz, ts, -1))
        return cond

    def compute_kpe_enc(self, j2d, intrx, freq=4):
        x_angle = torch.arctan2(j2d[..., 0] - intrx[:, :, 0:1, 2], intrx[:, :, 0:1, 0])  # (B, T, J)
        y_angle = torch.arctan2(j2d[..., 1] - intrx[:, :, 1:2, 2], intrx[:, :, 1:2, 1])  # (B, T, J)
        angle = torch.stack([x_angle, y_angle], dim=-1)  # (B, T, J, 2)
        bz, ts, js = angle.shape[:3]
        freq_expand = 2**torch.arange(freq).unsqueeze(0).repeat(bz,1).reshape(bz,1,1,-1).to(angle.device)
        angle_expand = angle.unsqueeze(-2) * freq_expand.unsqueeze(-1)  # (B, T, J, 2, freq)
        pos_enc = torch.stack([torch.sin(angle_expand), torch.cos(angle_expand)], dim=-1)
        pos_enc = pos_enc.reshape(bz, ts, js, -1)
        return pos_enc
    
    def compute_plucker_rays(self, j3d, cam_t):
        # https://github.com/jasonyzhang/RayDiffusion/blob/main/ray_diffusion/utils/rays.py#L151
        ray_origin = cam_t.unsqueeze(2)
        ray_directions = j3d - ray_origin  # [bz, ts, J, 3] - [bz, ts, 1, 3] -> [bz, ts, J, 3]
        # normalize the rays
        norms = torch.norm(ray_directions, dim=-1, keepdim=True)
        ray_directions = ray_directions / (norms + 1e-8)  # avoid division by zero
        plucker_normal = torch.cross(ray_origin, ray_directions, dim=-1)  # [bz, ts, J, 3]
        plucker_rays = torch.cat([ray_directions, plucker_normal], dim=-1)  # [bz, ts, J, 6]
        return plucker_rays
    
    def get_condition_feats(self, inputs):
        more_cond = {}
        if 'pose' in self.cond_mode:
            view_pose_r, view_pose_l = inputs['view.pose.r'], inputs['view.pose.l']
            bz, inp_ts = view_pose_r.shape[:2] # only one timestep is considered for now, TODO: support multiple timesteps
            if self.rot_dim == 6:
                view_pose_r = rot_conv.axis_angle_to_matrix(view_pose_r.reshape(bz, -1, 3))
                view_pose_r = rot_conv.matrix_to_rotation_6d(view_pose_r).reshape(bz, inp_ts, -1)
                view_pose_l = rot_conv.axis_angle_to_matrix(view_pose_l.reshape(bz, -1, 3))
                view_pose_l = rot_conv.matrix_to_rotation_6d(view_pose_l).reshape(bz, inp_ts, -1)
            pose_cond = torch.cat([view_pose_r, view_pose_l], dim=-1)
            pose_cond = torch.cat([pose_cond, inputs['view.transl.r'], inputs['view.transl.l']], dim=-1) # (bz, inp_ts, ch)
            pose_cond = pose_cond.reshape(bz, -1) # (bz, ch*inp_ts) # this needs to be changed for supporting multi-timestep history
            more_cond['pose'] = pose_cond
        
        if 'img' in self.cond_mode:
            more_cond['img'] = inputs['img'][:, -1] # (bz, 3, 224, 224) TODO: support multi-timestep history
        
        if self.args.get('interpolate', False):
            more_cond['goal_img'] = inputs['goal_img'][:, -1] # (bz, 3, 224, 224) TODO: support mutliple goal images

        cond_feats = self.conditioner(more_cond)
        return cond_feats
    
    def convert_to_reference_frame(self, res_pose_r, res_transl_r, res_pose_l, res_transl_l, targets):
        bz, ts = res_pose_r.shape[:2]
        device = res_pose_r.device

        # compute future2view transforms from residual transforms
        future_pose_r, future_transl_r = tf.convert_residual_to_full_transforms(res_pose_r, res_transl_r)

        mano2view = targets['mano2view'] # [bs, 4, 4]
        mano2view = mano2view.unsqueeze(1).repeat(1, ts, 1, 1).view(-1, 4, 4)

        # transform future_pose_r, future_transl_r to mano view space
        futuremano2mano = torch.eye(4).to(device).repeat(bz*ts, 1, 1).to(device)
        futuremano2mano[:, :3, :3] = rot_conv.axis_angle_to_matrix(future_pose_r[...,:3].reshape(-1, 3))
        futuremano2mano[:, :3, 3] = future_transl_r.reshape(-1, 3)
        futuremano2view = torch.bmm(mano2view.view(-1,4,4), futuremano2mano.view(-1,4,4))
        futuremano2view_rot = rot_conv.matrix_to_axis_angle(futuremano2view[:, :3, :3]).reshape(bz, ts, 3)
        future_pose_r_ = torch.cat([futuremano2view_rot, future_pose_r[...,3:]], dim=-1)
        future_transl_r_ = futuremano2view[:, :3, 3].reshape(bz, ts, 3)

        mano_r = self.mano_r(
            rotmat = future_pose_r_.reshape(bz * ts, -1),
            shape=targets['future_betas_r'].reshape(bz * ts, -1),
            cam_t=future_transl_r_.reshape(bz * ts, -1),
        )

        future_pose_l, future_transl_l = tf.convert_residual_to_full_transforms(res_pose_l, res_transl_l)
        
        futuremano2left = torch.eye(4).to(device).repeat(bz*ts, 1, 1).to(device)
        futuremano2left[:, :3, :3] = rot_conv.axis_angle_to_matrix(future_pose_l[...,:3].reshape(-1, 3))
        futuremano2left[:, :3, 3] = future_transl_l.reshape(-1, 3)
        left2mano_mat = targets['left2mano'].unsqueeze(1).repeat(1, ts, 1, 1)
        futuremano2mano = torch.bmm(left2mano_mat.reshape(-1,4,4), futuremano2left.reshape(-1,4,4))
        futuremano2view = torch.bmm(mano2view.view(-1,4,4), futuremano2mano.view(-1,4,4))
        futuremano2view_rot = rot_conv.matrix_to_axis_angle(futuremano2view[:, :3, :3]).reshape(bz, ts, 3)
        future_pose_l_ = torch.cat([futuremano2view_rot, future_pose_l[...,3:]], dim=-1)
        future_transl_l_ = futuremano2view[:, :3, 3].reshape(bz, ts, 3)

        mano_l = self.mano_l(
            rotmat = future_pose_l_.reshape(bz * ts, -1),
            shape=targets['future_betas_l'].reshape(bz * ts, -1),
            cam_t=future_transl_l_.reshape(bz * ts, -1),
        )

        return mano_r, mano_l