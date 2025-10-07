import numpy as np
import torch
import pytorch3d.transforms.rotation_conversions as rot_conv

import common.data_utils as data_utils
from common.np_utils import permute_np

"""
Useful geometric operations, e.g. Perspective projection and a differentiable Rodrigues formula
Parts of the code are taken from https://github.com/MandyMo/pytorch_HMR
"""


def to_xy(x_homo):
    assert isinstance(x_homo, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x_homo.shape[1] == 3
    assert len(x_homo.shape) == 2
    batch_size = x_homo.shape[0]
    x = torch.ones(batch_size, 2, device=x_homo.device)
    x = x_homo[:, :2] / x_homo[:, 2:3]
    return x


def to_xyz(x_homo):
    assert isinstance(x_homo, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x_homo.shape[1] == 4
    assert len(x_homo.shape) == 2
    batch_size = x_homo.shape[0]
    x = torch.ones(batch_size, 3, device=x_homo.device)
    x = x_homo[:, :3] / x_homo[:, 3:4]
    return x


def to_homo(x):
    assert isinstance(x, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x.shape[1] == 3
    assert len(x.shape) == 2
    batch_size = x.shape[0]
    x_homo = torch.ones(batch_size, 4, device=x.device)
    x_homo[:, :3] = x.clone()
    return x_homo


def to_homo_batch(x):
    assert isinstance(x, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x.shape[2] == 3
    assert len(x.shape) == 3
    batch_size = x.shape[0]
    num_pts = x.shape[1]
    x_homo = torch.ones(batch_size, num_pts, 4, device=x.device)
    x_homo[:, :, :3] = x.clone()
    return x_homo


def to_xyz_batch(x_homo):
    """
    Input: (B, N, 4)
    Ouput: (B, N, 3)
    """
    assert isinstance(x_homo, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x_homo.shape[2] == 4
    assert len(x_homo.shape) == 3
    batch_size = x_homo.shape[0]
    num_pts = x_homo.shape[1]
    x = torch.ones(batch_size, num_pts, 3, device=x_homo.device)
    x = x_homo[:, :, :3] / x_homo[:, :, 3:4]
    return x


def to_xy_batch(x_homo):
    assert isinstance(x_homo, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x_homo.shape[2] == 3
    assert len(x_homo.shape) == 3
    batch_size = x_homo.shape[0]
    num_pts = x_homo.shape[1]
    x = torch.ones(batch_size, num_pts, 2, device=x_homo.device)
    x = x_homo[:, :, :2] / x_homo[:, :, 2:3]
    return x


# VR Distortion Correction Using Vertex Displacement
# https://stackoverflow.com/questions/44489686/camera-lens-distortion-in-opengl
def distort_pts3d_all(_pts_cam, dist_coeffs):
    # egocentric cameras commonly has heavy distortion
    # this function transform points in the undistorted camera coord
    # to distorted camera coord such that the 2d projection can match the pixels.
    pts_cam = _pts_cam.clone().double()
    z = pts_cam[:, :, 2]

    z_inv = 1 / z

    x1 = pts_cam[:, :, 0] * z_inv
    y1 = pts_cam[:, :, 1] * z_inv

    # precalculations
    x1_2 = x1 * x1
    y1_2 = y1 * y1
    x1_y1 = x1 * y1
    r2 = x1_2 + y1_2
    r4 = r2 * r2
    r6 = r4 * r2

    r_dist = (1 + dist_coeffs[0] * r2 + dist_coeffs[1] * r4 + dist_coeffs[4] * r6) / (
        1 + dist_coeffs[5] * r2 + dist_coeffs[6] * r4 + dist_coeffs[7] * r6
    )

    # full (rational + tangential) distortion
    x2 = x1 * r_dist + 2 * dist_coeffs[2] * x1_y1 + dist_coeffs[3] * (r2 + 2 * x1_2)
    y2 = y1 * r_dist + 2 * dist_coeffs[3] * x1_y1 + dist_coeffs[2] * (r2 + 2 * y1_2)
    # denormalize for projection (which is a linear operation)
    cam_pts_dist = torch.stack([x2 * z, y2 * z, z], dim=2).float()
    return cam_pts_dist


def rigid_tf_torch_batch(points, R, T):
    """
    Performs rigid transformation to incoming points but batched
    Q = (points*R.T) + T
    points: (batch, num, 3)
    R: (batch, 3, 3)
    T: (batch, 3, 1)
    out: (batch, num, 3)
    """
    points_out = torch.bmm(R, points.permute(0, 2, 1)) + T
    points_out = points_out.permute(0, 2, 1)
    return points_out


def solve_rigid_tf_np(A: np.ndarray, B: np.ndarray):
    """
    “Least-Squares Fitting of Two 3-D Point Sets”, Arun, K. S. , May 1987
    Input: expects Nx3 matrix of points
    Returns R,t
    R = 3x3 rotation matrix
    t = 3x1 column vector

    This function should be a fix for compute_rigid_tf when the det == -1
    """

    assert A.shape == B.shape
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def batch_solve_rigid_tf(A, B):
    """
    “Least-Squares Fitting of Two 3-D Point Sets”, Arun, K. S. , May 1987
    Input: expects BxNx3 matrix of points
    Returns R,t
    R = Bx3x3 rotation matrix
    t = Bx3x1 column vector
    """

    assert A.shape == B.shape
    dev = A.device
    A = A.cpu().numpy()
    B = B.cpu().numpy()
    A = permute_np(A, (0, 2, 1))
    B = permute_np(B, (0, 2, 1))

    batch, num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    _, num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=2)
    centroid_B = np.mean(B, axis=2)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(batch, -1, 1)
    centroid_B = centroid_B.reshape(batch, -1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = np.matmul(Am, permute_np(Bm, (0, 2, 1)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(permute_np(Vt, (0, 2, 1)), permute_np(U, (0, 2, 1)))

    # special reflection case
    neg_idx = np.linalg.det(R) < 0
    if neg_idx.sum() > 0:
        raise Exception(
            f"some rotation matrices are not orthogonal; make sure implementation is correct for such case: {neg_idx}"
        )
    Vt[neg_idx, 2, :] *= -1
    R[neg_idx, :, :] = np.matmul(
        permute_np(Vt[neg_idx], (0, 2, 1)), permute_np(U[neg_idx], (0, 2, 1))
    )

    t = np.matmul(-R, centroid_A) + centroid_B

    R = torch.FloatTensor(R).to(dev)
    t = torch.FloatTensor(t).to(dev)
    return R, t


def rigid_tf_np(points, R, T):
    """
    Performs rigid transformation to incoming points
    Q = (points*R.T) + T
    points: (num, 3)
    R: (3, 3)
    T: (1, 3)

    out: (num, 3)
    """

    assert isinstance(points, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert isinstance(T, np.ndarray)
    assert len(points.shape) == 2
    assert points.shape[1] == 3
    assert R.shape == (3, 3)
    assert T.shape == (1, 3)
    points_new = np.matmul(R, points.T).T + T
    return points_new


def transform_points(world2cam_mat, pts):
    """
    Map points from one coord to another based on the 4x4 matrix.
    e.g., map points from world to camera coord.
    pts: (N, 3), in METERS!!
    world2cam_mat: (4, 4)
    Output: points in cam coord (N, 3)
    We follow this convention:
    | R T |   |pt|
    | 0 1 | * | 1|
    i.e. we rotate first then translate as T is the camera translation not position.
    """
    assert isinstance(pts, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(world2cam_mat, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert world2cam_mat.shape == (4, 4)
    assert len(pts.shape) == 2
    assert pts.shape[1] == 3
    pts_homo = to_homo(pts)

    # mocap to cam
    pts_cam_homo = torch.matmul(world2cam_mat, pts_homo.T).T
    pts_cam = to_xyz(pts_cam_homo)

    assert pts_cam.shape[1] == 3
    return pts_cam


def transform_points_batch(world2cam_mat, pts):
    """
    Map points from one coord to another based on the 4x4 matrix.
    e.g., map points from world to camera coord.
    pts: (B, N, 3), in METERS!!
    world2cam_mat: (B, 4, 4)
    Output: points in cam coord (B, N, 3)
    We follow this convention:
    | R T |   |pt|
    | 0 1 | * | 1|
    i.e. we rotate first then translate as T is the camera translation not position.
    """
    assert isinstance(pts, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(world2cam_mat, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert world2cam_mat.shape[1:] == (4, 4)
    assert len(pts.shape) == 3
    assert pts.shape[2] == 3
    batch_size = pts.shape[0]
    pts_homo = to_homo_batch(pts)

    # mocap to cam
    pts_cam_homo = torch.bmm(world2cam_mat, pts_homo.permute(0, 2, 1)).permute(0, 2, 1)
    pts_cam = to_xyz_batch(pts_cam_homo)

    assert pts_cam.shape[2] == 3
    return pts_cam


def project2d_batch(K, pts_cam):
    """
    K: (B, 3, 3)
    pts_cam: (B, N, 3)
    """

    assert isinstance(K, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(pts_cam, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert K.shape[1:] == (3, 3)
    assert pts_cam.shape[2] == 3
    assert len(pts_cam.shape) == 3
    pts2d_homo = torch.bmm(K, pts_cam.permute(0, 2, 1)).permute(0, 2, 1)
    pts2d = to_xy_batch(pts2d_homo)
    return pts2d


def project2d_norm_batch(K, pts_cam, patch_width):
    """
    K: (B, 3, 3)
    pts_cam: (B, N, 3)
    """

    assert isinstance(K, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(pts_cam, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert K.shape[1:] == (3, 3)
    assert pts_cam.shape[2] == 3
    assert len(pts_cam.shape) == 3
    v2d = project2d_batch(K, pts_cam)
    v2d_norm = data_utils.normalize_kp2d(v2d, patch_width)
    return v2d_norm


def project2d(K, pts_cam):
    assert isinstance(K, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(pts_cam, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert K.shape == (3, 3)
    assert pts_cam.shape[1] == 3
    assert len(pts_cam.shape) == 2
    pts2d_homo = torch.matmul(K, pts_cam.T).T
    pts2d = to_xy(pts2d_homo)
    return pts2d


def transform_rotation_batch(pose_aa, transform):
    """
    Arguments:
        pose_aa: axis angle representation of rotation
        transform: 4x4 matrix
    Returns:
        transformed rotation in axis angle representation
    """
    pose_mat = rot_conv.axis_angle_to_matrix(pose_aa) # B x 3 x 3
    pose_mat_4x4 = torch.eye(4).to(pose_mat.device).repeat(pose_mat.shape[0], 1, 1)
    pose_mat_4x4[:, :3, :3] = pose_mat
    transformed_mat_4x4 = torch.bmm(transform, pose_mat_4x4)
    transformed_mat = transformed_mat_4x4[:, :3, :3]
    transformed_aa = rot_conv.matrix_to_axis_angle(transformed_mat)
    return transformed_aa


def compute_rotation_no_translation_batched(A, B):
    """
    Compute the best-fit rotation matrix R_i (3x3) for each batch i,
    aligning two already-centered 3D point sets A[i], B[i] of shape (N, 3).
    
    This version does not consider translation: A[i] and B[i] are assumed
    to already be centered at the origin (i.e., mean zero).

    Args:
        A (Tensor): (B, N, 3) batch of 3D points (already centered)
        B (Tensor): (B, N, 3) batch of 3D points (already centered),
                    corresponding point-for-point with A

    Returns:
        R (Tensor): (B, 3, 3) batch of rotation matrices that best align
                    A[i] to B[i] in a least-squares sense.
    """
    # A, B: (B, N, 3).  We'll build the covariance-like matrix H for each batch:
    # H[i] = A[i]^T * B[i], shape => (3,3) for each batch i.
    # => shape of H: (B, 3, 3)
    H = torch.bmm(A.transpose(1, 2), B)  # (B,3,3)

    # Perform batched SVD on H
    # => U, S, Vt are each (B, 3, 3)
    U, S, Vt = torch.linalg.svd(H)

    # Compute rotation: R = V * U^T
    # but we need to ensure det(R) = +1 => handle possible reflections
    # det(R[i]) = det(V[i]*U[i]^T). We'll check sign for each batch item.
    # Vt^T = V
    V = Vt.transpose(-2, -1)    # (B,3,3)
    U_t = U.transpose(-2, -1)   # (B,3,3)

    # Check the determinant of (V * U^T)
    # = det(V) * det(U^T) = det(V) * det(U), but easier to do a direct matrix multiply and det
    R_candidate = torch.bmm(V, U_t)  # (B,3,3)
    dets = torch.linalg.det(R_candidate)  # (B,)

    # Where determinant is negative, we flip the sign of the last row of V
    # (this corrects the reflection)
    mask = (dets < 0)
    if mask.any():
        # Flip the sign of the last row in V for each item with negative determinant
        V[mask, :, -1] *= -1.0
        R_candidate = torch.bmm(V, U_t)  # recalc R with corrected V

    return R_candidate  # shape (B,3,3)


def convert_full_to_residual_transform(pose, transl, transform):
    """
    Compute the residual rotation and translation from 't' to 't+1'.
    Args:
        pose_aa: B x T x 48 rotation from mano to camera frame at each timestep
        transl: B x T x 3 translation from mano to camera frame at each timestep
        transform: B x T x 4 x 4 matrix transform from each timestep to reference frame

    Returns:
        residual_pose: residual rotation in axis angle
        residual_transl: residual translation
    """
    pose_aa = pose[..., :3].clone()
    B, T = pose_aa.shape[:2]
    device = pose_aa.device
    # compute 4x4 matrix for mano to camera transform at each timestep
    mano2cam = torch.eye(4).to(device).repeat(B*T, 1, 1)
    mano2cam[:, :3, :3] = rot_conv.axis_angle_to_matrix(pose_aa.reshape(-1, 3))
    mano2cam[:, :3, 3] = transl.reshape(-1, 3)

    # apply transform to each timestep to get mano2ref
    mano2ref = torch.bmm(transform.view(-1, 4, 4), mano2cam)
    mano2ref = mano2ref.reshape(B, T, 4, 4)

    BTm1 = B * (T - 1)
    prev_mano2ref = mano2ref[:, :-1] # B x (T-1) x 4 x 4
    curr_mano2ref = mano2ref[:, 1:] # B x (T-1) x 4 x 4
    prev_mano2ref = prev_mano2ref.reshape(BTm1, 4, 4)
    curr_mano2ref = curr_mano2ref.reshape(BTm1, 4, 4)
    # residual_transform = torch.bmm(torch.linalg.inv(curr_mano2ref), prev_mano2ref) # B(T-1) x 4 x 4
    residual_transform = torch.bmm(torch.linalg.inv(prev_mano2ref), curr_mano2ref) # check which is correct
    residual_transform = residual_transform.reshape(B, T-1, 4, 4)
    first_residual = mano2ref[:, 0:1] # B x 4 x 4
    residual_transform = torch.cat([first_residual, residual_transform], dim=1) # B x T x 4 x 4

    # extract residual rotation and translation
    residual_pose_mat = residual_transform[:, :, :3, :3]
    residual_pose_aa = rot_conv.matrix_to_axis_angle(residual_pose_mat.reshape(-1, 3, 3)).reshape(B, T, 3)
    pose[..., :3] = residual_pose_aa
    residual_transl = residual_transform[:, :, :3, 3]

    return pose, residual_transl


def convert_residual_to_full_transforms(rotation, translation):
        """
        Convert a sequence of 'residual' transforms (R, t) into their
        cumulative 'full' transforms. Avoids in-place updates to preserve autograd.

        Arguments:
        rotation:    (B, T, 3)   Axis-angle rotations per time step
        translation: (B, T, 3)   Translation vectors per time step
        rot_conv:    A module with:
                        axis_angle_to_matrix(aa): (N,3)->(N,3,3)
                        matrix_to_axis_angle(R):  (N,3,3)->(N,3)

        Returns:
        full_rotation_aa: (B, T, 3 or more) The cumulative axis-angle rotation
                            at each step (plus any leftover dims from 'rotation').
        full_translation:  (B, T, 3)        The cumulative translation at each step.
        """
        B, T = rotation.shape[:2]
        device = rotation.device

        # 1) Build the "residual" 4×4 transforms for each step
        #    shape => (B, T, 4, 4)
        residual_transforms = torch.eye(4, device=device).repeat(B, T, 1, 1)
        # Fill rotation
        # (B*T, 3,3) from axis angles
        rot_mats = rot_conv.axis_angle_to_matrix(rotation[..., :3].reshape(-1, 3))  # (B*T,3,3)
        rot_mats = rot_mats.reshape(B, T, 3, 3)  # (B, T,3,3)
        residual_transforms[:, :, :3, :3] = rot_mats
        # Fill translation
        residual_transforms[:, :, :3, 3] = translation

        # 2) Compute the "full" transforms in an out-of-place manner
        #    Instead of assigning full_transforms[:, t] = ..., we accumulate in a list.
        full_list = []
        # At t=0, the full transform is just the residual transform
        acc = residual_transforms[:, 0]  # shape (B,4,4)
        full_list.append(acc)

        # For t >= 1, accumulate the product
        for t_idx in range(1, T):
            # Multiply the previous "acc" by the current residual
            acc = torch.bmm(acc, residual_transforms[:, t_idx])
            # acc = torch.bmm(residual_transforms[:, t_idx], acc)
            full_list.append(acc)

        # Stack them into (B, T, 4,4)
        full_transforms = torch.stack(full_list, dim=1)

        # 3) Extract the cumulative rotation & translation
        #    Convert rotation matrix -> axis-angle
        #      shape => (B*T,3,3) -> (B*T,3) -> (B,T,3)
        full_rot_mats = full_transforms[:, :, :3, :3].reshape(B * T, 3, 3)
        full_rotation = rot_conv.matrix_to_axis_angle(full_rot_mats).reshape(B, T, 3)

        # The translation is simply the last column of the 4×4
        full_translation = full_transforms[:, :, :3, 3]

        # 4) If your input 'rotation' had >3 dims in the last axis, you can
        #    concatenate them back. For instance:
        leftover_dims = rotation.shape[-1] - 3
        if leftover_dims > 0:
            # rotation[..., 3:] are leftover dims => shape (B,T,leftover_dims)
            extra = rotation[..., 3:]
            # concat along last dimension => shape (B,T,3 + leftover_dims)
            full_rotation_aa = torch.cat([full_rotation, extra], dim=-1)
        else:
            full_rotation_aa = full_rotation

        return full_rotation_aa, full_translation


def get_view_transform(pose, transl, transform):
    """
    Compute the rotation and translation to the reference frame defined by transform.
    Args:
        pose_aa: B x T x 48 rotation from mano to camera frame at each timestep
        transl: B x T x 3 translation from mano to camera frame at each timestep
        transform: B x T x 4 x 4 matrix transform from each timestep to reference frame

    Returns:
        residual_pose: reference rotation in axis angle
        residual_transl: reference translation
    """
    pose_aa = pose[..., :3] # .clone() not needed here, pass in arguments when needed
    B, T = pose_aa.shape[:2]
    device = pose_aa.device
    # compute 4x4 matrix for mano to camera transform at each timestep
    mano2cam = torch.eye(4).to(device).repeat(B*T, 1, 1)
    mano2cam[:, :3, :3] = rot_conv.axis_angle_to_matrix(pose_aa.reshape(-1, 3))
    mano2cam[:, :3, 3] = transl.reshape(-1, 3)

    # apply transform to each timestep to get mano2ref
    mano2ref = torch.bmm(transform.view(-1, 4, 4), mano2cam)

    # extract reference rotation and translation
    ref_pose = rot_conv.matrix_to_axis_angle(mano2ref[:, :3, :3]).reshape(B, T, 3)
    ref_pose = torch.cat([ref_pose, pose[..., 3:]], dim=-1)
    ref_transl = mano2ref[:, :3, 3].reshape(B, T, 3)

    return ref_pose, ref_transl