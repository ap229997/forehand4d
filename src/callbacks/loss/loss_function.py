import torch
import torch.nn as nn
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix

import common.transforms as tf
import common.data_utils as data_utils
from src.utils.loss_modules import (
    hand_kp3d_loss,
    joints_loss,
    mano_loss,
    vector_loss,
    )

l1_loss = nn.L1Loss(reduction="none")
mse_loss = nn.MSELoss(reduction="none")
smooth_l1_loss = nn.SmoothL1Loss(reduction="none")


def compute_loss_motion(pred, gt, meta_info, args, targets=None): # targets not used in here, just for compatibility
    # unpacking pred and gt
    pred_betas_r = gt["future_betas_r"] # beta is not predicted
    pred_rotmat_r = pred["mano.pose.r"]
    pred_joints_r = pred["future.j3d.cam.r"]
    pred_vertices_r = pred["future.v3d.cam.r"]
    pred_betas_l = gt["future_betas_l"] # beta is not predicted
    pred_rotmat_l = pred["mano.pose.l"]
    pred_joints_l = pred["future.j3d.cam.l"]
    pred_vertices_l = pred["future.v3d.cam.l"]
    
    gt_pose_r = gt['future.view.pose.r']
    gt_betas_r = gt["future_betas_r"]
    gt_joints_r = gt["future.j3d.cam.r"]
    gt_vertices_r = gt["future.v3d.cam.r"]
    gt_pose_l = gt['future.view.pose.l']
    gt_betas_l = gt["future_betas_l"]
    gt_joints_l = gt["future.j3d.cam.l"]
    gt_vertices_l = gt["future.v3d.cam.l"]

    bz, ts = meta_info['mask_timesteps'].shape[:2]
    joints_valid_r = gt["joints_valid_r"].reshape(bz, -1, 21)
    joints_valid_l = gt["joints_valid_l"].reshape(bz, -1, 21)

    valid_mask = meta_info['mask_timesteps'].reshape(-1, 1)
    future_valid_r = gt["future_valid_r"]
    future_valid_l = gt["future_valid_l"]
    # valid if atleast 3 joints are valid
    right_valid = (torch.sum(future_valid_r, dim=-1, keepdim=True) >= 3) * valid_mask
    left_valid = (torch.sum(future_valid_l, dim=-1, keepdim=True) >= 3) * valid_mask
    right_valid = right_valid.reshape(-1)
    left_valid = left_valid.reshape(-1)
    right_valid_f = future_valid_r * valid_mask # only consider valid timesteps
    left_valid_f = future_valid_l * valid_mask
    joints_valid_r = right_valid_f.reshape(bz, ts, -1) * joints_valid_r[:, -1:] # only consider when last frame is valid
    joints_valid_l = left_valid_f.reshape(bz, ts, -1) * joints_valid_l[:, -1:]
    joints_valid_r = joints_valid_r.reshape(-1, 21)
    joints_valid_l = joints_valid_l.reshape(-1, 21)

    if pred_rotmat_r.shape[-1] == 48:
        pred_rotmat_r = axis_angle_to_matrix(pred_rotmat_r.reshape(-1, 3)).reshape(-1, 16, 3, 3)
        pred_rotmat_l = axis_angle_to_matrix(pred_rotmat_l.reshape(-1, 3)).reshape(-1, 16, 3, 3)

    # reshape
    gt_pose_r = axis_angle_to_matrix(gt_pose_r.reshape(-1, 3)).reshape(-1, 16, 3, 3)
    gt_pose_l = axis_angle_to_matrix(gt_pose_l.reshape(-1, 3)).reshape(-1, 16, 3, 3)

    # Compute loss on MANO parameters
    loss_regr_pose_r, loss_regr_betas_r = mano_loss(
        pred_rotmat_r,
        pred_betas_r,
        gt_pose_r,
        gt_betas_r,
        criterion=mse_loss,
        is_valid=right_valid,
        return_mean=False,
    )
    loss_regr_pose_l, loss_regr_betas_l = mano_loss(
        pred_rotmat_l,
        pred_betas_l,
        gt_pose_l,
        gt_betas_l,
        criterion=mse_loss,
        is_valid=left_valid,
        return_mean=False,
    )

    # Compute 3D keypoint loss for joints, this loss has both relative and absolute components
    # TODO: use relatively since there is a loss on cam_t down as well
    loss_keypoints_3d_r = hand_kp3d_loss(
        pred_joints_r, gt_joints_r, mse_loss, joints_valid_r, return_mean=False, subtract_root=args.get('relative_kp3d', False),
    )
    loss_keypoints_3d_l = hand_kp3d_loss(
        pred_joints_l, gt_joints_l, mse_loss, joints_valid_l, return_mean=False, subtract_root=args.get('relative_kp3d', False),
    )

    # Compute 3D vertex loss, currently valid is defined at joint level, not vertex level
    # TODO: define valid at vertex level in dataset function and load from there
    vertices_valid_r = right_valid.reshape(-1, 1) * joints_valid_r[:, -1:]
    vertices_valid_l = left_valid.reshape(-1, 1) * joints_valid_l[:, -1:]
    vertices_valid_r = vertices_valid_r.repeat(1, gt_vertices_r.shape[1])
    vertices_valid_l = vertices_valid_l.repeat(1, gt_vertices_l.shape[1])
    loss_vertices_3d_r = hand_kp3d_loss(
        pred_vertices_r, gt_vertices_r, mse_loss, vertices_valid_r, return_mean=False, subtract_root=args.get('relative_kp3d', False),
    )
    loss_vertices_3d_l = hand_kp3d_loss(
        pred_vertices_l, gt_vertices_l, mse_loss, vertices_valid_l, return_mean=False, subtract_root=args.get('relative_kp3d', False),
    )

    loss_cam_t_r = vector_loss(
        pred["mano.cam_t.r"],
        gt['future.view.transl.r'],
        mse_loss,
        right_valid,
        return_mean=False,
    )
    loss_cam_t_l = vector_loss(
        pred["mano.cam_t.l"],
        gt['future.view.transl.l'],
        mse_loss,
        left_valid,
        return_mean=False,
    ) 

    bz = meta_info['is_j2d_loss'].shape[0] # last dim represents history window

    # only consider losses where mask is valid
    loss_cam_t_r = mean_loss_mask(loss_cam_t_r, right_valid, meta_info['is_cam_loss'][..., -1])
    loss_cam_t_l = mean_loss_mask(loss_cam_t_l, left_valid, meta_info['is_cam_loss'][..., -1])
    loss_keypoints_3d_r = mean_loss_mask(loss_keypoints_3d_r, joints_valid_r, meta_info['is_j3d_loss'][..., -1])
    loss_keypoints_3d_l = mean_loss_mask(loss_keypoints_3d_l, joints_valid_l, meta_info['is_j3d_loss'][..., -1])
    loss_vertices_3d_r = mean_loss_mask(loss_vertices_3d_r, vertices_valid_r, meta_info['is_j3d_loss'][..., -1])
    loss_vertices_3d_l = mean_loss_mask(loss_vertices_3d_l, vertices_valid_l, meta_info['is_j3d_loss'][..., -1])
    loss_regr_pose_r = mean_loss_mask(loss_regr_pose_r, right_valid, meta_info['is_pose_loss'][..., -1])
    loss_regr_pose_l = mean_loss_mask(loss_regr_pose_l, left_valid, meta_info['is_pose_loss'][..., -1])
    loss_regr_betas_r = mean_loss_mask(loss_regr_betas_r, right_valid, meta_info['is_beta_loss'][..., -1])
    loss_regr_betas_l = mean_loss_mask(loss_regr_betas_l, left_valid, meta_info['is_beta_loss'][..., -1])

    loss_dict = {
        "loss/mano/cam_t/r": (loss_cam_t_r.mean().view(-1), 0.01),
        "loss/mano/cam_t/l": (loss_cam_t_l.mean().view(-1), 0.01),
        "loss/mano/kp3d/r": (loss_keypoints_3d_r.mean().view(-1), 1.0),
        "loss/mano/v3d/r": (loss_vertices_3d_r.mean().view(-1), 1.0),
        "loss/mano/pose/r": (loss_regr_pose_r.mean().view(-1), 1.0),
        "loss/mano/beta/r": (loss_regr_betas_r.mean().view(-1), 1.0),
        "loss/mano/kp3d/l": (loss_keypoints_3d_l.mean().view(-1), 1.0),
        "loss/mano/v3d/l": (loss_vertices_3d_l.mean().view(-1), 1.0),
        "loss/mano/pose/l": (loss_regr_pose_l.mean().view(-1), 1.0),
        "loss/mano/beta/l": (loss_regr_betas_l.mean().view(-1), 1.0),
    }
    
    return loss_dict


# this only works when predictions are done in view frame, doesn't work for mano or residual frame
def compute_loss_2d(pred, gt, meta_info, args, targets=None):
    # these are in view frame
    pred_joints_r = pred["future.j3d.cam.r"]
    pred_joints_l = pred["future.j3d.cam.l"]
    
    bz, ts = meta_info['mask_timesteps'].shape[:2]
    K = meta_info['intrinsics'][:, -1:].repeat(1, ts, 1, 1).reshape(-1, 3, 3)
    # joints are in view frame
    future2view = targets['future2view'].reshape(-1, 4, 4)
    view2future = torch.linalg.inv(future2view) # [bs*ts, 4, 4]
    # convert to future frame
    pred_j3d_cam_r = tf.transform_points_batch(view2future, pred_joints_r)
    pred_j3d_cam_l = tf.transform_points_batch(view2future, pred_joints_l)
    # project to 2D
    j2d_r = torch.bmm(K.reshape(-1, 3, 3), pred_j3d_cam_r.permute(0, 2, 1)).permute(0, 2, 1)
    j2d_l = torch.bmm(K.reshape(-1, 3, 3), pred_j3d_cam_l.permute(0, 2, 1)).permute(0, 2, 1)
    j2d_r = j2d_r[..., :2] / (j2d_r[..., 2:3] + 1e-3) # add 1e-3 to avoid numerical issues
    j2d_l = j2d_l[..., :2] / (j2d_l[..., 2:3] + 1e-3)
    # normalize to [-1, 1]
    pred_j2d_r = data_utils.normalize_kp2d(j2d_r, args.img_res)
    pred_j2d_l = data_utils.normalize_kp2d(j2d_l, args.img_res)
    
    gt_j2d_r = gt['future.j2d.norm.r']
    gt_j2d_l = gt['future.j2d.norm.l']

    bz, ts = meta_info['mask_timesteps'].shape[:2]
    joints_valid_r = gt["joints_valid_r"].reshape(bz, -1, 21)
    joints_valid_l = gt["joints_valid_l"].reshape(bz, -1, 21)
    
    valid_mask = meta_info['mask_timesteps'].reshape(-1, 1)
    future_valid_r = gt["future_valid_r"]
    future_valid_l = gt["future_valid_l"]
    right_valid_f = future_valid_r * valid_mask # only consider valid timesteps
    left_valid_f = future_valid_l * valid_mask
    joints_valid_r = right_valid_f.reshape(bz, ts, -1) * joints_valid_r[:, -1:] # only consider when last frame is valid
    joints_valid_l = left_valid_f.reshape(bz, ts, -1) * joints_valid_l[:, -1:]
    joints_valid_r = joints_valid_r.reshape(-1, 21)
    joints_valid_l = joints_valid_l.reshape(-1, 21)

    # Compute 2D reprojection loss for the keypoints
    loss_joints_r = joints_loss(
        pred_j2d_r,
        gt_j2d_r,
        criterion=smooth_l1_loss, # og: mse_loss,
        jts_valid=joints_valid_r,
        return_mean=False,
    )
    loss_joints_l = joints_loss(
        pred_j2d_l,
        gt_j2d_l,
        criterion=smooth_l1_loss, # og: mse_loss,
        jts_valid=joints_valid_l,
        return_mean=False,
    )

    bz = meta_info['is_j2d_loss'].shape[0] # last dim represents history window

    # only consider losses where mask is valid
    loss_joints_r = mean_loss_mask(loss_joints_r, joints_valid_r, meta_info['is_j2d_loss'][..., -1])
    loss_joints_l = mean_loss_mask(loss_joints_l, joints_valid_l, meta_info['is_j2d_loss'][..., -1])

    loss_dict = {
        "loss/mano/j2d/r": (loss_joints_r.mean().view(-1), 1.0),
        "loss/mano/j2d/l": (loss_joints_l.mean().view(-1), 1.0),
    }
    
    return loss_dict


def compute_loss_motion_hybrid(pred, gt, meta_info, args, targets=None): # targets not used in here, just for compatibility
    # unpacking pred and gt
    pred_betas_r = gt["future_betas_r"] # beta is not predicted
    pred_rotmat_r = pred["tf.mano.pose.r"]
    pred_joints_r = pred["tf.future.j3d.cam.r"]
    pred_vertices_r = pred["tf.future.v3d.cam.r"]
    pred_betas_l = gt["future_betas_l"] # beta is not predicted
    pred_rotmat_l = pred["tf.mano.pose.l"]
    pred_joints_l = pred["tf.future.j3d.cam.l"]
    pred_vertices_l = pred["tf.future.v3d.cam.l"]
    
    gt_pose_r = gt['future.view.pose.r'] # gt['future.residual.pose.r']
    gt_betas_r = gt["future_betas_r"]
    gt_joints_r = gt["future.j3d.cam.r"]
    gt_vertices_r = gt["future.v3d.cam.r"]
    gt_pose_l = gt['future.view.pose.l'] # gt['future.residual.pose.l']
    gt_betas_l = gt["future_betas_l"]
    gt_joints_l = gt["future.j3d.cam.l"]
    gt_vertices_l = gt["future.v3d.cam.l"]

    bz, ts = meta_info['mask_timesteps'].shape[:2]
    joints_valid_r = gt["joints_valid_r"].reshape(bz, -1, 21)
    joints_valid_l = gt["joints_valid_l"].reshape(bz, -1, 21)

    valid_mask = meta_info['mask_timesteps'].reshape(-1, 1)
    future_valid_r = gt["future_valid_r"]
    future_valid_l = gt["future_valid_l"]
    # valid if atleast 3 joints are valid
    right_valid = (torch.sum(future_valid_r, dim=-1, keepdim=True) >= 3) * valid_mask
    left_valid = (torch.sum(future_valid_l, dim=-1, keepdim=True) >= 3) * valid_mask
    right_valid = right_valid.reshape(-1)
    left_valid = left_valid.reshape(-1)
    right_valid_f = future_valid_r * valid_mask # only consider valid timesteps
    left_valid_f = future_valid_l * valid_mask
    joints_valid_r = right_valid_f.reshape(bz, ts, -1) * joints_valid_r[:, -1:] # only consider when last frame is valid
    joints_valid_l = left_valid_f.reshape(bz, ts, -1) * joints_valid_l[:, -1:]
    joints_valid_r = joints_valid_r.reshape(-1, 21)
    joints_valid_l = joints_valid_l.reshape(-1, 21)

    if pred_rotmat_r.shape[-1] == 48:
        pred_rotmat_r = axis_angle_to_matrix(pred_rotmat_r.reshape(-1, 3)).reshape(-1, 16, 3, 3)
        pred_rotmat_l = axis_angle_to_matrix(pred_rotmat_l.reshape(-1, 3)).reshape(-1, 16, 3, 3)

    # reshape
    gt_pose_r = axis_angle_to_matrix(gt_pose_r.reshape(-1, 3)).reshape(-1, 16, 3, 3)
    gt_pose_l = axis_angle_to_matrix(gt_pose_l.reshape(-1, 3)).reshape(-1, 16, 3, 3)

    # Compute loss on MANO parameters
    loss_regr_pose_r, loss_regr_betas_r = mano_loss(
        pred_rotmat_r,
        pred_betas_r,
        gt_pose_r,
        gt_betas_r,
        criterion=mse_loss,
        is_valid=right_valid,
        return_mean=False,
    )
    loss_regr_pose_l, loss_regr_betas_l = mano_loss(
        pred_rotmat_l,
        pred_betas_l,
        gt_pose_l,
        gt_betas_l,
        criterion=mse_loss,
        is_valid=left_valid,
        return_mean=False,
    )

    # Compute 3D keypoint loss for joints, this loss has both relative and absolute components
    # TODO: use relatively since there is a loss on cam_t down as well
    loss_keypoints_3d_r = hand_kp3d_loss(
        pred_joints_r, gt_joints_r, mse_loss, joints_valid_r, return_mean=False, subtract_root=args.get('relative_kp3d', False),
    )
    loss_keypoints_3d_l = hand_kp3d_loss(
        pred_joints_l, gt_joints_l, mse_loss, joints_valid_l, return_mean=False, subtract_root=args.get('relative_kp3d', False),
    )

    # Compute 3D vertex loss, currently valid is defined at joint level, not vertex level
    # TODO: define valid at vertex level in dataset function and load from there
    vertices_valid_r = right_valid.reshape(-1, 1) * joints_valid_r[:, -1:]
    vertices_valid_l = left_valid.reshape(-1, 1) * joints_valid_l[:, -1:]
    vertices_valid_r = vertices_valid_r.repeat(1, gt_vertices_r.shape[1])
    vertices_valid_l = vertices_valid_l.repeat(1, gt_vertices_l.shape[1])
    loss_vertices_3d_r = hand_kp3d_loss(
        pred_vertices_r, gt_vertices_r, mse_loss, vertices_valid_r, return_mean=False, subtract_root=args.get('relative_kp3d', False),
    )
    loss_vertices_3d_l = hand_kp3d_loss(
        pred_vertices_l, gt_vertices_l, mse_loss, vertices_valid_l, return_mean=False, subtract_root=args.get('relative_kp3d', False),
    )

    loss_cam_t_r = vector_loss(
        pred["tf.mano.cam_t.r"],
        gt['future.view.transl.r'],
        mse_loss,
        right_valid,
        return_mean=False,
    )
    loss_cam_t_l = vector_loss(
        pred["tf.mano.cam_t.l"],
        gt['future.view.transl.l'],
        mse_loss,
        left_valid,
        return_mean=False,
    ) 

    bz = meta_info['is_j2d_loss'].shape[0] # last dim represents history window

    # only consider losses where mask is valid
    loss_cam_t_r = mean_loss_mask(loss_cam_t_r, right_valid, meta_info['is_cam_loss'][..., -1])
    loss_cam_t_l = mean_loss_mask(loss_cam_t_l, left_valid, meta_info['is_cam_loss'][..., -1])
    loss_keypoints_3d_r = mean_loss_mask(loss_keypoints_3d_r, joints_valid_r, meta_info['is_j3d_loss'][..., -1])
    loss_keypoints_3d_l = mean_loss_mask(loss_keypoints_3d_l, joints_valid_l, meta_info['is_j3d_loss'][..., -1])
    loss_vertices_3d_r = mean_loss_mask(loss_vertices_3d_r, vertices_valid_r, meta_info['is_j3d_loss'][..., -1])
    loss_vertices_3d_l = mean_loss_mask(loss_vertices_3d_l, vertices_valid_l, meta_info['is_j3d_loss'][..., -1])
    loss_regr_pose_r = mean_loss_mask(loss_regr_pose_r, right_valid, meta_info['is_pose_loss'][..., -1])
    loss_regr_pose_l = mean_loss_mask(loss_regr_pose_l, left_valid, meta_info['is_pose_loss'][..., -1])
    loss_regr_betas_r = mean_loss_mask(loss_regr_betas_r, right_valid, meta_info['is_beta_loss'][..., -1])
    loss_regr_betas_l = mean_loss_mask(loss_regr_betas_l, left_valid, meta_info['is_beta_loss'][..., -1])

    loss_dict = {
        "loss/hybrid/cam_t/r": (loss_cam_t_r.mean().view(-1), 0.001),
        "loss/hybrid/cam_t/l": (loss_cam_t_l.mean().view(-1), 0.001),
        "loss/hybrid/kp3d/r": (loss_keypoints_3d_r.mean().view(-1), 1.0),
        "loss/hybrid/v3d/r": (loss_vertices_3d_r.mean().view(-1), 1.0),
        "loss/hybrid/pose/r": (loss_regr_pose_r.mean().view(-1), 1.0),
        "loss/hybrid/beta/r": (loss_regr_betas_r.mean().view(-1), 1.0),
        "loss/hybrid/kp3d/l": (loss_keypoints_3d_l.mean().view(-1), 1.0),
        "loss/hybrid/v3d/l": (loss_vertices_3d_l.mean().view(-1), 1.0),
        "loss/hybrid/pose/l": (loss_regr_pose_l.mean().view(-1), 1.0),
        "loss/hybrid/beta/l": (loss_regr_betas_l.mean().view(-1), 1.0),
    }
    
    return loss_dict


# this only works when predictions are done in view frame, doesn't work for mano or residual frame
def compute_loss_2d_hybrid(pred, gt, meta_info, args, targets=None):
    # these are in view frame
    pred_joints_r = pred["tf.future.j3d.cam.r"]
    pred_joints_l = pred["tf.future.j3d.cam.l"]
    
    bz, ts = meta_info['mask_timesteps'].shape[:2]
    K = meta_info['intrinsics'][:, -1:].repeat(1, ts, 1, 1).reshape(-1, 3, 3)
    # joints are in view frame, verify this
    future2view = targets['future2view'].reshape(-1, 4, 4)
    view2future = torch.linalg.inv(future2view) # [bs*ts, 4, 4]
    # convert to future frame
    pred_j3d_cam_r = tf.transform_points_batch(view2future, pred_joints_r)
    pred_j3d_cam_l = tf.transform_points_batch(view2future, pred_joints_l)
    # project to 2D
    j2d_r = torch.bmm(K.reshape(-1, 3, 3), pred_j3d_cam_r.permute(0, 2, 1)).permute(0, 2, 1)
    j2d_l = torch.bmm(K.reshape(-1, 3, 3), pred_j3d_cam_l.permute(0, 2, 1)).permute(0, 2, 1)
    j2d_r = j2d_r[..., :2] / (j2d_r[..., 2:3] + 1e-3) # add 1e-3 to avoid numerical issues
    j2d_l = j2d_l[..., :2] / (j2d_l[..., 2:3] + 1e-3)
    # normalize to [-1, 1]
    pred_j2d_r = data_utils.normalize_kp2d(j2d_r, args.img_res)
    pred_j2d_l = data_utils.normalize_kp2d(j2d_l, args.img_res)
    
    gt_j2d_r = gt['future.j2d.norm.r']
    gt_j2d_l = gt['future.j2d.norm.l']

    bz, ts = meta_info['mask_timesteps'].shape[:2]
    joints_valid_r = gt["joints_valid_r"].reshape(bz, -1, 21)
    joints_valid_l = gt["joints_valid_l"].reshape(bz, -1, 21)
    
    valid_mask = meta_info['mask_timesteps'].reshape(-1, 1)
    future_valid_r = gt["future_valid_r"]
    future_valid_l = gt["future_valid_l"]
    right_valid_f = future_valid_r * valid_mask # only consider valid timesteps
    left_valid_f = future_valid_l * valid_mask
    joints_valid_r = right_valid_f.reshape(bz, ts, -1) * joints_valid_r[:, -1:] # only consider when last frame is valid
    joints_valid_l = left_valid_f.reshape(bz, ts, -1) * joints_valid_l[:, -1:]
    joints_valid_r = joints_valid_r.reshape(-1, 21)
    joints_valid_l = joints_valid_l.reshape(-1, 21)

    # Compute 2D reprojection loss for the keypoints
    loss_joints_r = joints_loss(
        pred_j2d_r,
        gt_j2d_r,
        criterion=mse_loss,
        jts_valid=joints_valid_r,
        return_mean=False,
    )
    loss_joints_l = joints_loss(
        pred_j2d_l,
        gt_j2d_l,
        criterion=mse_loss,
        jts_valid=joints_valid_l,
        return_mean=False,
    )

    bz = meta_info['is_j2d_loss'].shape[0] # last dim represents history window

    # only consider losses where mask is valid
    loss_joints_r = mean_loss_mask(loss_joints_r, joints_valid_r, meta_info['is_j2d_loss'][..., -1])
    loss_joints_l = mean_loss_mask(loss_joints_l, joints_valid_l, meta_info['is_j2d_loss'][..., -1])
    
    loss_dict = {
        "loss/hybrid/j2d/r": (loss_joints_r.mean().view(-1), 1.0),
        "loss/hybrid/j2d/l": (loss_joints_l.mean().view(-1), 1.0),
    }
    
    return loss_dict


def mean_loss_mask(diff, mask, meta):
    """
    Compute the mean of `diff` for the entire batch, considering only the elements
    where `mask` is True and only for the valid samples indicated by `meta`. If `mask`
    has fewer dimensions than `diff`, it is repeated (broadcasted) along the missing dimensions.
    
    Args:
        diff (Tensor): A tensor of shape (bz*ts, ...) containing differences or errors.
        mask (Tensor): A boolean (or float) tensor of shape (bz*ts, ...) or with fewer trailing 
                       dimensions than `diff`. Only the elements where mask is True (or nonzero)
                       are considered in the mean.
        meta (Tensor): A tensor of shape (bz) indicating which batch items are valid.
                       It will be broadcast to all timesteps of each batch item.
    
    Returns:
        loss (Tensor): A tensor of shape (1,) containing the mean loss computed over all valid elements.
    """
    # Convert mask to float so that True becomes 1.0 and False becomes 0.0.
    mask = mask.float()
    # If mask has fewer dimensions than diff, unsqueeze at the end until they match.
    while mask.dim() < diff.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(diff)
    
    # Determine the number of batch items (bz) and time steps (ts).
    bz = meta.shape[0]
    total = diff.shape[0]
    ts = total // bz  # assume diff is flattened along (bz, ts)
    
    # Expand meta (shape (bz,)) so that each batch item's meta value applies to all its timesteps.
    # We want to end up with a tensor of shape (bz, ts, 1, 1, ..., 1) matching diff's dimensions (except for the first).
    # Compute the number of extra dimensions needed:
    extra_dims = (1,) * (diff.dim() - 1)
    # First, reshape meta to (bz, 1) then expand to (bz, ts) and then add extra dimensions.
    meta_expanded = meta.view(bz, 1).expand(bz, ts).view(bz, ts, *([1] * (diff.dim() - 1)))
    # Now flatten meta_expanded to shape (bz*ts, 1, 1, ..., 1)
    meta_expanded = meta_expanded.reshape(total, *([1] * (diff.dim() - 1)))
    meta_expanded = meta_expanded.float()
    
    # The effective mask is the element-wise product of mask and meta.
    effective_mask = mask * meta_expanded

    # Multiply diff by the effective mask; only valid elements contribute.
    masked_diff = diff * effective_mask
    
    # Sum all valid diff values and count the number of valid elements.
    sum_loss = masked_diff.sum()
    count = effective_mask.sum()
    
    # Compute the mean loss; add a small epsilon to avoid division by zero.
    epsilon = 1e-8
    loss = sum_loss / (count + epsilon)
    
    # Return the loss as a tensor of shape (1,)
    return loss.unsqueeze(0)