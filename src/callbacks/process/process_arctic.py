import torch
import common.camera as camera
import common.data_utils as data_utils
import common.transforms as tf
import common.torch_utils as torch_utils
from common.xdict import xdict
import pytorch3d.transforms.rotation_conversions as rot_conv


def process_data_light(
    models, inputs, targets, meta_info, mode, args
):
    img_res = args.img_res
    K = meta_info["intrinsics"]

    inp_dims = len(K.shape)
    if inp_dims == 4: # B, T, 3, 3
        bz, ts = K.shape[:2]
        K = K.view(bz * ts, 3, 3)
        targets = torch_utils.reduce_dict_dims(targets, dims=[0, 1])
    
    gt_pose_r = targets["mano.pose.r"]  # MANO pose parameters
    gt_betas_r = targets["mano.beta.r"]  # MANO beta parameters

    gt_pose_l = targets["mano.pose.l"]  # MANO pose parameters
    gt_betas_l = targets["mano.beta.l"]  # MANO beta parameters

    # pose MANO in MANO canonical space
    gt_out_r = models["mano_r"](
        betas=gt_betas_r,
        hand_pose=gt_pose_r[:, 3:],
        global_orient=gt_pose_r[:, :3],
        transl=None,
    )
    gt_model_joints_r = gt_out_r.joints # MANO canonical space
    gt_vertices_r = gt_out_r.vertices # MANO canonical space
    gt_root_cano_r = gt_out_r.joints[:, 0] # MANO canonical space

    targets['mano.joints3d.r'] = gt_out_r.joints # MANO canonical space
    targets['mano.vertices.r'] = gt_out_r.vertices # MANO canonical space

    gt_out_l = models["mano_l"](
        betas=gt_betas_l,
        hand_pose=gt_pose_l[:, 3:],
        global_orient=gt_pose_l[:, :3],
        transl=None,
    )
    gt_model_joints_l = gt_out_l.joints # MANO canonical space
    gt_vertices_l = gt_out_l.vertices # MANO canonical space
    gt_root_cano_l = gt_out_l.joints[:, 0] # MANO canonical space

    targets['mano.joints3d.l'] = gt_out_l.joints # MANO canonical space
    targets['mano.vertices.l'] = gt_out_l.vertices # MANO canonical space

    Tr0 = targets['mano.j3d.full.r'][:, 0] - gt_model_joints_r[:, 0]
    Tl0 = targets['mano.j3d.full.l'][:, 0] - gt_model_joints_l[:, 0]
    ego_gt_vertices_r = gt_vertices_r + Tr0[:, None, :]
    ego_gt_vertices_l = gt_vertices_l + Tl0[:, None, :]
    
    ####### run this part and replace above variables with the new ones for arctic_exo portion of the data
    # define right mano posed frames as cano frame and recompute 3D quantities in this frame
    # this is required since arctic_exo modifies the intrinsics to align with camera center
    # leading to projection issues since camera has changed
    cano_joints3d = gt_model_joints_r.clone()
    exo_gt_model_joints_r, exo_gt_model_joints_l, exo_gt_vertices_r, exo_gt_vertices_l = process_arctic_exo_data_light(
        cano_joints3d, gt_model_joints_r.clone(), gt_model_joints_l.clone(), gt_vertices_r.clone(), gt_vertices_l.clone(), targets, meta_info, img_res
    )
    # replace arctic_exo targets with the new ones
    bz = len(meta_info['dataset'])
    for i in range(bz):
        if meta_info['dataset'][i][-1] == 'arctic_exo': # all timesteps of each sequence belong to the same dataset
            gt_vertices_r[i] = exo_gt_vertices_r[i]
            gt_vertices_l[i] = exo_gt_vertices_l[i]
            gt_model_joints_r[i] = exo_gt_model_joints_r[i]
            gt_model_joints_l[i] = exo_gt_model_joints_l[i]
            targets['mano.j3d.full.r'][i] = exo_gt_model_joints_r[i]
            targets['mano.j3d.full.l'][i] = exo_gt_model_joints_l[i]
        else:
            gt_vertices_r[i] = ego_gt_vertices_r[i]
            gt_vertices_l[i] = ego_gt_vertices_l[i]
    #############################

    # roots
    gt_root_cam_patch_r = targets['mano.j3d.full.r'][:, 0]
    gt_root_cam_patch_l = targets['mano.j3d.full.l'][:, 0]
    gt_cam_t_r = gt_root_cam_patch_r - gt_root_cano_r
    gt_cam_t_l = gt_root_cam_patch_l - gt_root_cano_l

    targets["mano.cam_t.r"] = gt_cam_t_r
    targets["mano.cam_t.l"] = gt_cam_t_l

    avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
    gt_cam_t_wp_r = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_r, avg_focal_length, img_res
    )

    gt_cam_t_wp_l = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_l, avg_focal_length, img_res
    )

    targets["mano.cam_t.wp.r"] = gt_cam_t_wp_r
    targets["mano.cam_t.wp.l"] = gt_cam_t_wp_l
    targets["mano.v3d.cam.r"] = gt_vertices_r
    targets["mano.v3d.cam.l"] = gt_vertices_l
    targets["mano.j3d.cam.r"] = targets['mano.j3d.full.r']
    targets["mano.j3d.cam.l"] = targets['mano.j3d.full.l']

    if inp_dims == 4:
        targets = torch_utils.expand_dict_dims(targets, curr_dim=0, dims=[bz, -1])

    return inputs, targets, meta_info


def process_arctic_exo_data_light(cano_joints3d, 
                                  gt_model_joints_r, gt_model_joints_l, 
                                  gt_vertices_r, gt_vertices_l, 
                                  targets, meta_info, img_res):
    try:
        R0, T0 = tf.batch_solve_rigid_tf(targets["mano.j3d.full.r"], cano_joints3d)
        joints3d_r0 = tf.rigid_tf_torch_batch(targets["mano.j3d.full.r"], R0, T0)
        joints3d_l0 = tf.rigid_tf_torch_batch(targets["mano.j3d.full.l"], R0, T0)
    except:
        joints3d_r0 = targets["mano.j3d.full.r"].clone()
        joints3d_l0 = targets["mano.j3d.full.l"].clone()

    Tr0 = (joints3d_r0 - gt_model_joints_r).mean(dim=1)
    Tl0 = (joints3d_l0 - gt_model_joints_l).mean(dim=1)
    gt_vertices_r += Tr0[:, None, :]
    gt_vertices_l += Tl0[:, None, :]
    
    # now solve for camera translation w.r.t this new cano frame
    # unnorm 2d keypoints
    gt_j2d_r = targets['mano.j2d.norm.r'].clone()
    gt_j2d_r = data_utils.unormalize_kp2d(gt_j2d_r, img_res)
    # estimate camera translation by solving 2d to 3d correspondence
    try:
        gt_transl = camera.estimate_translation_k(
            joints3d_r0,
            gt_j2d_r,
            meta_info["intrinsics"].cpu().numpy().reshape(-1, 3, 3),
            use_all_joints=True,
            pad_2d=True,
        )
    except:
        gt_transl = torch.zeros_like(joints3d_r0[:, 0, :])

    gt_model_joints_r = joints3d_r0 + gt_transl[:, None, :]
    gt_model_joints_l = joints3d_l0 + gt_transl[:, None, :]
    gt_vertices_r = gt_vertices_r + gt_transl[:, None, :]
    gt_vertices_l = gt_vertices_l + gt_transl[:, None, :]
    
    return gt_model_joints_r, gt_model_joints_l, gt_vertices_r, gt_vertices_l


def process_2d_data(
    models, inputs, targets, meta_info, mode, args
):
    return inputs, targets, meta_info


def process_future_data(
    models, inputs, targets, meta_info, mode, args
):
    img_res = args.img_res
    K = meta_info["intrinsics"]

    # extract keys with future data
    future_targets = xdict()
    for k, v in targets.items():
        if "future" in k:
            future_targets[k] = v

    inp_dims = len(K.shape)
    is_future = (inp_dims == 4)
    if is_future:
        bz, ts = future_targets['future_joints3d_r'].shape[:2]
        K = K[:, -1:].repeat(1, ts, 1, 1).view(bz * ts, 3, 3)
        future_targets = torch_utils.reduce_dict_dims(future_targets, dims=[0, 1])
    
    gt_pose_r = future_targets["future_pose_r"]  # MANO pose parameters
    gt_betas_r = future_targets["future_betas_r"]  # MANO beta parameters

    gt_pose_l = future_targets["future_pose_l"]  # MANO pose parameters
    gt_betas_l = future_targets["future_betas_l"]  # MANO beta parameters

    # pose MANO in MANO canonical space
    gt_out_r = models["mano_r"](
        betas=gt_betas_r,
        hand_pose=gt_pose_r[:, 3:],
        global_orient=gt_pose_r[:, :3],
        transl=None,
    )
    gt_model_joints_r = gt_out_r.joints # MANO canonical space
    gt_vertices_r = gt_out_r.vertices # MANO canonical space
    gt_root_cano_r = gt_out_r.joints[:, 0] # MANO canonical space

    future_targets['future.joints3d.r'] = gt_out_r.joints # MANO canonical space
    future_targets['future.vertices.r'] = gt_out_r.vertices # MANO canonical space

    gt_out_l = models["mano_l"](
        betas=gt_betas_l,
        hand_pose=gt_pose_l[:, 3:],
        global_orient=gt_pose_l[:, :3],
        transl=None,
    )
    gt_model_joints_l = gt_out_l.joints # MANO canonical space
    gt_vertices_l = gt_out_l.vertices # MANO canonical space
    gt_root_cano_l = gt_out_l.joints[:, 0] # MANO canonical space

    future_targets['future.joints3d.l'] = gt_out_l.joints # MANO canonical space
    future_targets['future.vertices.l'] = gt_out_l.vertices # MANO canonical space

    # translation from MANO cano space to camera coord space
    # TODO: how to handle this when batch has both ego and exo data
    Tr0 = future_targets['future_joints3d_r'][:, 0] - gt_model_joints_r[:, 0]
    Tl0 = future_targets['future_joints3d_l'][:, 0] - gt_model_joints_l[:, 0]
    ego_gt_vertices_r = gt_vertices_r + Tr0[:, None, :]
    ego_gt_vertices_l = gt_vertices_l + Tl0[:, None, :]
    
    ##### run this part and replace above variables with the new ones for arctic_exo portion of the data
    # define right mano posed frames as cano frame and recompute 3D quantities in this frame
    # this is required since arctic_exo modifies the intrinsics to align with camera center
    # leading to projection issues since camera has changed
    cano_joints3d = gt_model_joints_r.clone()
    exo_gt_model_joints_r, exo_gt_model_joints_l, exo_gt_vertices_r, exo_gt_vertices_l = process_arctic_exo_future_data(
        cano_joints3d, gt_model_joints_r.clone(), gt_model_joints_l.clone(), gt_vertices_r.clone(), gt_vertices_l.clone(), future_targets, meta_info, img_res
    )
    # replace arctic_exo targets with the new ones
    bz = len(meta_info['dataset'])
    for i in range(bz):
        st, end = i * ts, (i + 1) * ts
        if meta_info['dataset'][i][-1] == 'arctic_exo': # all timesteps of each sequence belong to the same dataset
            gt_vertices_r[st:end] = exo_gt_vertices_r[st:end]
            gt_vertices_l[st:end] = exo_gt_vertices_l[st:end]
            gt_model_joints_r[st:end] = exo_gt_model_joints_r[st:end]
            gt_model_joints_l[st:end] = exo_gt_model_joints_l[st:end]
            future_targets['future_joints3d_r'][st:end] = exo_gt_model_joints_r[st:end]
            future_targets['future_joints3d_l'][st:end] = exo_gt_model_joints_l[st:end]
        else:
            gt_vertices_r[st:end] = ego_gt_vertices_r[st:end]
            gt_vertices_l[st:end] = ego_gt_vertices_l[st:end]
    #############################

    # roots
    gt_root_cam_patch_r = future_targets['future_joints3d_r'][:, 0]
    gt_root_cam_patch_l = future_targets['future_joints3d_l'][:, 0]
    gt_cam_t_r = gt_root_cam_patch_r - gt_root_cano_r
    gt_cam_t_l = gt_root_cam_patch_l - gt_root_cano_l

    future_targets["future.cam_t.r"] = gt_cam_t_r
    future_targets["future.cam_t.l"] = gt_cam_t_l

    avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
    gt_cam_t_wp_r = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_r, avg_focal_length, img_res
    )

    gt_cam_t_wp_l = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_l, avg_focal_length, img_res
    )

    future_targets["future.cam_t.wp.r"] = gt_cam_t_wp_r
    future_targets["future.cam_t.wp.l"] = gt_cam_t_wp_l

    # transform from future camera to current camera coordiante frame
    future2view = future_targets["future2view"]
    gt_vertices_r = tf.transform_points_batch(future2view, gt_vertices_r)
    gt_vertices_l = tf.transform_points_batch(future2view, gt_vertices_l)
    gt_joints_r = tf.transform_points_batch(future2view, future_targets["future_joints3d_r"])
    gt_joints_l = tf.transform_points_batch(future2view, future_targets["future_joints3d_l"])

    future_targets["future.v3d.cam.r"] = gt_vertices_r
    future_targets["future.v3d.cam.l"] = gt_vertices_l
    future_targets["future.j3d.cam.r"] = gt_joints_r
    future_targets["future.j3d.cam.l"] = gt_joints_l
    
    if is_future:
        future_targets = torch_utils.expand_dict_dims(future_targets, curr_dim=0, dims=[bz, -1])

    # TODO: clean this and move to a function or merge with above code
    # future2view is conversion to camera @ t=0, not mano future to camera
    # convert "future2view" to mano coordinate system at t=0
    device = future_targets["future2view"].device
    curr_pose_r = targets['mano.pose.r'][:, -1, :3]
    curr_cam_t_r = targets['mano.cam_t.r'][:, -1]
    # create mano2view 4x4 matrix
    mano2view = torch.eye(4).to(device).repeat(bz, 1, 1)
    mano2view[:, :3, :3] = rot_conv.axis_angle_to_matrix(curr_pose_r)
    mano2view[:, :3, 3] = curr_cam_t_r
    view2mano = torch.linalg.inv(mano2view) # B x 4 x 4
    # apply to view2mano to future2view to get future2mano, future2view is B x T x 4 x 4
    view2mano = view2mano.unsqueeze(1).repeat(1, ts, 1, 1)
    B, T = view2mano.shape[:2]
    
    # this goes from future camera in future view to right hand mano in reference view
    req_transf = torch.bmm(view2mano.view(-1, 4, 4), future2view.view(-1, 4, 4)).reshape(B, T, 4, 4)
    
    residual_pose_r, residual_transl_r = tf.convert_full_to_residual_transform(
        future_targets['future_pose_r'].clone(),
        future_targets['future.cam_t.r'].clone(),
        req_transf.clone(),
    )

    ref_view_pose_r, ref_view_transl_r = tf.get_view_transform(
        future_targets['future_pose_r'].clone(),
        future_targets['future.cam_t.r'].clone(),
        future2view.clone(),
    )

    ref_mano_pose_r, ref_mano_transl_r = tf.get_view_transform(
        future_targets['future_pose_r'].clone(),
        future_targets['future.cam_t.r'].clone(),
        req_transf.clone(),
    )
    
    # transform left hand to right hand mano coordinate frame
    curr_pose_l = targets['mano.pose.l'][:, -1, :3] # these are in the camera frame of reference view
    curr_cam_t_l = targets['mano.cam_t.l'][:, -1]
    left_transf = torch.eye(4).to(device).repeat(bz, 1, 1)
    left_transf[:, :3, :3] = rot_conv.axis_angle_to_matrix(curr_pose_l)
    left_transf[:, :3, 3] = curr_cam_t_l
    # transform to the mano frame of right hand using view2mano
    left2mano = torch.bmm(view2mano[:,0], left_transf.view(-1, 4, 4))
    left2mano_rot = left2mano[:, :3, :3]
    left2mano_aa = rot_conv.matrix_to_axis_angle(left2mano_rot)
    left2mano_transl = left2mano[:, :3, 3]
    # replace curr_pose_l and curr_cam_t_l
    targets['mano.pose.l'][:, -1, :3] = left2mano_aa
    targets['mano.cam_t.l'][:, -1] = left2mano_transl

    residual_pose_l, residual_transl_l = tf.convert_full_to_residual_transform(
        future_targets['future_pose_l'].clone(),
        future_targets['future.cam_t.l'].clone(),
        req_transf.clone(),
    ) # this is in right hand mano frame

    ref_view_pose_l, ref_view_transl_l = tf.get_view_transform(
        future_targets['future_pose_l'].clone(),
        future_targets['future.cam_t.l'].clone(),
        future2view.clone(),
    ) # this is in view frame

    ref_mano_pose_l, ref_mano_transl_l = tf.get_view_transform(
        future_targets['future_pose_l'].clone(),
        future_targets['future.cam_t.l'].clone(),
        req_transf.clone(),
    ) # this is in right hand mano frame in view 

    # update residual for t=0 of left hand, these are w.r.t right hand mano in ref frame
    # convert these to the left hand mano in ref frame
    residual_pose_l_first = residual_pose_l[:, 0, :3] # T x 3
    residual_transl_l_first = residual_transl_l[:, 0]
    # create 4x4 matrix
    first_residual_mat = torch.eye(4).to(device).repeat(bz, 1, 1)
    first_residual_mat[:, :3, :3] = rot_conv.axis_angle_to_matrix(residual_pose_l_first)
    first_residual_mat[:, :3, 3] = residual_transl_l_first
    
    # left2mano is in the mano frame frame, same as the residual quantities above
    # create 4x4 matrix for using left2mano_aa and left2mano_transl
    ref_pose_mat = torch.eye(4).to(device).repeat(bz, 1, 1)
    ref_pose_mat[:, :3, :3] = rot_conv.axis_angle_to_matrix(left2mano_aa)
    ref_pose_mat[:, :3, 3] = left2mano_transl
    # take inverse of ref_pose_mat
    first_residual_ref_l = torch.bmm(torch.linalg.inv(ref_pose_mat), first_residual_mat)
    first_residual_ref_l_aa = rot_conv.matrix_to_axis_angle(first_residual_ref_l[:, :3, :3])
    first_residual_ref_l_transl = first_residual_ref_l[:, :3, 3]
    # update residual_pose_l
    residual_pose_l[:, 0, :3] = first_residual_ref_l_aa
    residual_transl_l[:, 0] = first_residual_ref_l_transl

    future_targets['future.residual.pose.r'] = residual_pose_r
    future_targets['future.residual.transl.r'] = residual_transl_r
    future_targets['future.residual.pose.l'] = residual_pose_l
    future_targets['future.residual.transl.l'] = residual_transl_l
    future_targets['mano2view'] = mano2view
    future_targets['left2mano'] = ref_pose_mat

    future_targets['future.view.pose.r'] = ref_view_pose_r
    future_targets['future.view.transl.r'] = ref_view_transl_r
    future_targets['future.view.pose.l'] = ref_view_pose_l
    future_targets['future.view.transl.l'] = ref_view_transl_l

    future_targets['future.mano.pose.r'] = ref_mano_pose_r
    future_targets['future.mano.transl.r'] = ref_mano_transl_r
    future_targets['future.mano.pose.l'] = ref_mano_pose_l
    future_targets['future.mano.transl.l'] = ref_mano_transl_l

    # view pose is used as condition to the diffusion model, add it to the input
    view_pose_r = targets['mano.pose.r'][:, -1:].clone()
    view_cam_t_r = targets['mano.cam_t.r'][:, -1:].clone()
    view_pose_l = targets['mano.pose.l'][:, -1:].clone()
    view_cam_t_l = targets['mano.cam_t.l'][:, -1:].clone()
    # set view_pose_r to be origin
    view_pose_r[..., :3] = 0
    view_cam_t_r = torch.zeros_like(view_cam_t_r)
    # add to the input
    inputs["view.pose.r"] = view_pose_r
    inputs["view.transl.r"] = view_cam_t_r
    inputs["view.pose.l"] = view_pose_l
    inputs["view.transl.l"] = view_cam_t_l

    # update targets with future_targets:
    for k, v in future_targets.items():
        if k in targets:
            targets.overwrite(k, future_targets[k])
        else:
            targets[k] = future_targets[k]

    return inputs, targets, meta_info


def process_arctic_exo_future_data(cano_joints3d, 
                                   gt_model_joints_r, gt_model_joints_l, 
                                   gt_vertices_r, gt_vertices_l, 
                                   future_targets, meta_info, img_res):
    try:
        R0, T0 = tf.batch_solve_rigid_tf(future_targets["future_joints3d_r"], cano_joints3d)
        joints3d_r0 = tf.rigid_tf_torch_batch(future_targets["future_joints3d_r"], R0, T0)
        joints3d_l0 = tf.rigid_tf_torch_batch(future_targets["future_joints3d_l"], R0, T0)
    except:
        joints3d_r0 = future_targets["future_joints3d_r"].clone()
        joints3d_l0 = future_targets["future_joints3d_l"].clone()

    Tr0 = (joints3d_r0 - gt_model_joints_r).mean(dim=1)
    Tl0 = (joints3d_l0 - gt_model_joints_l).mean(dim=1)
    gt_vertices_r += Tr0[:, None, :]
    gt_vertices_l += Tl0[:, None, :]
    
    bz = meta_info['intrinsics'].shape[0]
    bz_mul_ts = future_targets['future_joints3d_r'].shape[0]
    ts = bz_mul_ts // bz
    # now solve for camera translation w.r.t this new cano frame
    # unnorm 2d keypoints
    gt_j2d_r = future_targets['future.j2d.norm.r'].clone()
    gt_j2d_r = data_utils.unormalize_kp2d(gt_j2d_r, img_res)
    # estimate camera translation by solving 2d to 3d correspondence
    intrx = meta_info["intrinsics"][:, 0:1] # all are same
    intrx = intrx.repeat(1, ts, 1, 1).view(bz * ts, 3, 3)
    intrx = intrx.cpu().numpy()
    try:
        gt_transl = camera.estimate_translation_k(
            joints3d_r0,
            gt_j2d_r,
            intrx,
            use_all_joints=True,
            pad_2d=True,
        )
    except:
        gt_transl = torch.zeros_like(joints3d_r0[:, 0, :])

    gt_model_joints_r = joints3d_r0 + gt_transl[:, None, :]
    gt_model_joints_l = joints3d_l0 + gt_transl[:, None, :]
    gt_vertices_r = gt_vertices_r + gt_transl[:, None, :]
    gt_vertices_l = gt_vertices_l + gt_transl[:, None, :]

    return gt_model_joints_r, gt_model_joints_l, gt_vertices_r, gt_vertices_l