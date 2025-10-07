from tqdm import tqdm

import torch
import torch.nn as nn

import common.transforms as tf
import common.data_utils as data_utils
from src.utils.loss_modules import (
    hand_kp3d_loss,
    joints_loss,
    mse_loss
)
from src.callbacks.loss.loss_function import mean_loss_mask


class MANOOptimizer(nn.Module):
    def __init__(self, args, pose_r, transl_r, pose_l, transl_l):
        super().__init__()
        self.args = args

        self.pose_r = nn.Parameter(pose_r)
        self.pose_l = nn.Parameter(pose_l)
        self.transl_r = nn.Parameter(transl_r)
        self.transl_l = nn.Parameter(transl_l)

    def forward(self, model, targets, K):
        mano_output_r, mano_output_l = model.run_mano_on_pose_predictions(self.pose_r, self.transl_r, self.pose_l, self.transl_l, targets, K)
        return mano_output_r, mano_output_l
    
    def compute_loss(self, mano_output_r, mano_output_l, meta_info, targets, K):
        # these are in view frame
        pred_joints_r = mano_output_r["j3d.cam.r"]
        pred_joints_l = mano_output_l["j3d.cam.l"]
        
        # joints are in view frame, verify this
        future2view = targets['future2view'].reshape(-1, 4, 4)
        view2future = torch.linalg.inv(future2view) # [bs*ts, 4, 4]
        # convert to future frame
        pred_j3d_cam_r = tf.transform_points_batch(view2future, pred_joints_r)
        pred_j3d_cam_l = tf.transform_points_batch(view2future, pred_joints_l)
        # project to 2D
        j2d_r = tf.project2d_batch(K.reshape(-1, 3, 3), pred_j3d_cam_r)
        j2d_l = tf.project2d_batch(K.reshape(-1, 3, 3), pred_j3d_cam_l)
        # normalize to [-1, 1]
        pred_j2d_r = data_utils.normalize_kp2d(j2d_r, self.args.img_res)
        pred_j2d_l = data_utils.normalize_kp2d(j2d_l, self.args.img_res)
        
        gt_joints_r = targets["future_joints3d_r"].clone()
        gt_joints_l = targets["future_joints3d_l"].clone()
        gt_j2d_r = targets["future.j2d.norm.r"].clone()
        gt_j2d_l = targets["future.j2d.norm.l"].clone()

        if len(gt_joints_r.shape) == 4:
            bz, ts = gt_joints_r.shape[:2]
            gt_joints_r = gt_joints_r.reshape(bz * ts, -1, 3)
            gt_joints_l = gt_joints_l.reshape(bz * ts, -1, 3)
            gt_j2d_r = gt_j2d_r.reshape(bz * ts, -1, 2)
            gt_j2d_l = gt_j2d_l.reshape(bz * ts, -1, 2)
            future_valid_r = targets["future_valid_r"].clone().reshape(bz * ts, -1)
            future_valid_l = targets["future_valid_l"].clone().reshape(bz * ts, -1)

        # clip to valid range [-1, 1] to avoid numerical issues
        pred_j2d_r = torch.clip(pred_j2d_r, -1.0, 1.0)
        pred_j2d_l = torch.clip(pred_j2d_l, -1.0, 1.0)
        gt_j2d_r = torch.clip(gt_j2d_r, -1.0, 1.0)
        gt_j2d_l = torch.clip(gt_j2d_l, -1.0, 1.0)
        
        bz, ts = meta_info['mask_timesteps'].shape[:2]
        joints_valid_r = targets["joints_valid_r"].reshape(bz, -1, 21)
        joints_valid_l = targets["joints_valid_l"].reshape(bz, -1, 21)

        valid_mask = meta_info['mask_timesteps'].reshape(-1, 1)
        right_valid = future_valid_r * valid_mask # only consider valid timesteps
        left_valid = future_valid_l * valid_mask
        joints_valid_r = right_valid.reshape(bz, ts, -1) * joints_valid_r[:, -1:] # only consider when last frame is valid
        joints_valid_l = left_valid.reshape(bz, ts, -1) * joints_valid_l[:, -1:]
        joints_valid_r = joints_valid_r.reshape(-1, 21)
        joints_valid_l = joints_valid_l.reshape(-1, 21)

        # Compute 3D keypoint loss for joints
        # this loss has both relative and absolute components
        loss_keypoints_3d_r = hand_kp3d_loss(pred_j3d_cam_r, gt_joints_r, mse_loss, joints_valid_r, return_mean=False, subtract_root=False,)
        loss_keypoints_3d_l = hand_kp3d_loss(pred_j3d_cam_l, gt_joints_l, mse_loss, joints_valid_l, return_mean=False, subtract_root=False,)

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
        loss_keypoints_3d_r = mean_loss_mask(loss_keypoints_3d_r, joints_valid_r, meta_info['is_j3d_loss'][..., -1])
        loss_keypoints_3d_l = mean_loss_mask(loss_keypoints_3d_l, joints_valid_l, meta_info['is_j3d_loss'][..., -1])
        loss_joints_r = mean_loss_mask(loss_joints_r, joints_valid_r, meta_info['is_j2d_loss'][..., -1])
        loss_joints_l = mean_loss_mask(loss_joints_l, joints_valid_l, meta_info['is_j2d_loss'][..., -1])

        kps3d_wt, kps2d_wt = 1.0, 0.1
        if self.args.get('finetune_3d', 0) == 0:
            kps3d_wt = 0.0
            kps2d_wt = 1.0
        
        loss_dict = {
            "loss/mano/kp3d/r": (loss_keypoints_3d_r.mean().view(-1), kps3d_wt),
            "loss/mano/kp3d/l": (loss_keypoints_3d_l.mean().view(-1), kps3d_wt),
            "loss/mano/j2d/r": (loss_joints_r.mean().view(-1), kps2d_wt),
            "loss/mano/j2d/l": (loss_joints_l.mean().view(-1), kps2d_wt),
        }

        loss = sum(loss_dict[k][0] * loss_dict[k][1] for k in loss_dict)
        return loss, loss_dict


def optimize_mano_params(inputs, meta_info, targets, model, args):
    model.eval()
    with torch.no_grad():
        motion_out = model.sample(inputs, meta_info, targets)

    pose_r, transl_r, pose_l, transl_l = model.process_motion_output(motion_out)

    # bz, ts = motion_out.shape[:2]
    bz, ts = meta_info['mask_timesteps'].shape[:2]
    intrx = meta_info['intrinsics'][:, -1:].repeat(1, ts, 1, 1)

    num_iters = args.get('refine_iters', 100)
    if args.get('debug', False):
        num_iters = 10
    loss_thres = args.get('loss_thres', 1e-2)
    print (f"Optimizing MANO parameters using joint losses")
    # device = motion_out.device
    device = meta_info['intrinsics'].device
    mano_class = MANOOptimizer(args, pose_r, transl_r, pose_l, transl_l)
    mano_class = mano_class.to(device)
    optimizer = torch.optim.Adam(mano_class.parameters(), lr=args.get('lr_mano', 1e-2))
    # run optimization loop
    is_nan = False
    for i in tqdm(range(num_iters)):
        optimizer.zero_grad()
        mano_output_r, mano_output_l = mano_class(model, targets, intrx)
        loss, loss_dict = mano_class.compute_loss(mano_output_r, mano_output_l, meta_info, targets, intrx)
        if i == 0:
            print (f"Initial loss: {loss.item()}")
            for k, v in loss_dict.items():
                print (f"{k}: {v[0].item()}")
        # if i % 20 == 0:
        #     for k, v in loss_dict.items():
        #         print (f"{k}: {v[0].item()}")
        if loss.item() < loss_thres:
            break
        try:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mano_class.parameters(), max_norm=1.0)
            optimizer.step()
        except KeyboardInterrupt:
            exit(0)
        except:
            print ("NaN loss, stopping optimization")
            is_nan = True
            break
    
    if is_nan:
        return None, None, None, None
    
    print (f"Final loss: {loss.item()} at iteration {i}")
    for k, v in loss_dict.items():
        print (f"{k}: {v[0].item()}")

    pose_r_out = mano_class.pose_r.detach()
    transl_r_out = mano_class.transl_r.detach()
    pose_l_out = mano_class.pose_l.detach()
    transl_l_out = mano_class.transl_l.detach()
    return pose_r_out, transl_r_out, pose_l_out, transl_l_out