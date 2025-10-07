import os
import glob
import pickle
import os.path as op

from tqdm import tqdm
import numpy as np
import torch
import pytorch3d.transforms as rot_conv
from torchvision.transforms import Normalize

import common.data_utils as data_utils
import common.transforms as tf
from common.data_utils import read_img
from src.datasets.dataset_utils import pad_jts2d
from common.body_models import build_mano_aa
from src.datasets.base_dataset import BaseDataset


class H2ODataset(BaseDataset):
    def __init__(self,args,split='train'):
        self.baseDir = f"{os.environ['DOWNLOADS_DIR']}/data/h2o"
        if 'train' in split:
            self.split = 'train'
        elif 'val' in split:
            self.split = 'val'
        else:
            raise Exception('split not supported')
        self.args = args

        self._load_motion_data()

        self.samples = []
        self.max_video_frames = {}
        for i in range(len(self.motion_data['names'])):
            video_name = self.motion_data['names'][i]
            start_idx = str(self.motion_data['ranges'][i][0]).zfill(6)
            self.samples.append((video_name, start_idx))
            
            if video_name in self.max_video_frames:
                continue
            video_dir = os.path.join(self.baseDir, video_name)
            total_frames = len(glob.glob(os.path.join(video_dir, 'rgb/*.png')))
            self.max_video_frames[video_name] = total_frames
        self.imgnames = self.samples

        self.aug_data = split.endswith("train")
        self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)

        self.image_dir = os.path.join(self.baseDir, '{}', 'rgb', '{}.png')
        self.hand_dir = os.path.join(self.baseDir, '{}', 'hand_pose', '{}.txt')
        self.mano_dir = os.path.join(self.baseDir, '{}', 'hand_pose_mano', '{}.txt')
        self.intrx_dir = os.path.join(self.baseDir, '{}', 'cam_intrinsics.txt')
        self.cam_pose = os.path.join(self.baseDir, '{}', 'cam_pose', '{}.txt')

        self.mano_r = build_mano_aa(True, flat_hand=True)
        self.mano_l = build_mano_aa(False, flat_hand=True)

        self.h2o_to_mano_ordering = np.array([0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20])

        # preload hand labels for all samples, avoids too much I/O in dataloading
        self._load_hand_labels()

        self.img_h, self.img_w = 720, 1280

    def _load_motion_data(self):
        if self.args.get('use_fixed_length', False):
            data_file = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/h2o/fixed_{self.args.max_motion_length:03d}/h2o/{self.split}.pkl'
        else:
            data_file = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/h2o/{self.split}.pkl'
        with open(data_file, 'rb') as f:
            self.motion_data = pickle.load(f) # dict of lists
        
        for i in range(len(self.motion_data['names'])):
            name = self.motion_data['names'][i]
            splits = name.split('/')
            splits[0] += '_ego' # rgb images are available in ego folder
            video_name = '/'.join(splits)
            self.motion_data['names'][i] = video_name

    def _load_hand_labels(self):
        save_dir = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/h2o'
        hand_label_file = os.path.join(save_dir, f'hand_labels_{self.split}.pkl')
        cam_pose_file = os.path.join(save_dir, f'ego_cam_pose_{self.split}.pkl')
        if os.path.exists(hand_label_file) and os.path.exists(cam_pose_file):
            with open(hand_label_file, 'rb') as f:
                self.hand_labels = pickle.load(f)
            with open(cam_pose_file, 'rb') as f:
                self.ego_cam_pose = pickle.load(f)
            print ('Loaded hand labels from disk')
        
        else:
            self.hand_labels = {}
            self.ego_cam_pose = {}
            for ind in tqdm(range(len(self.motion_data['names']))):
                seqname = self.motion_data['names'][ind]
                max_video_frames = self.max_video_frames[seqname]
                for j in range(max_video_frames):
                    index = str(j).zfill(6)
                    hand_label = self.get_hand_labels(seqname, index)
                    ego_pose = self.get_egocam_pose(seqname, index)
                    if seqname not in self.hand_labels:
                        self.hand_labels[seqname] = {}
                        self.ego_cam_pose[seqname] = {}
                    if index not in self.hand_labels[seqname]:
                        self.hand_labels[seqname][index] = hand_label
                        self.ego_cam_pose[seqname][index] = ego_pose

            # save self.hand_labels and self.ego_cam_pose to disk in pickle format
            os.makedirs(save_dir, exist_ok=True)
            with open(hand_label_file, 'wb') as f:
                pickle.dump(self.hand_labels, f)
            with open(cam_pose_file, 'wb') as f:
                pickle.dump(self.ego_cam_pose, f)

            print ('Saved hand labels to {}'.format(save_dir))

    
    def __len__(self):
        return len(self.samples)
    
    def get_fixed_length_sequence(self, imgname, history_size, prediction_horizon):
        img_idx = int(imgname.split("/")[-1].split(".")[0])

        future_ind = (np.arange(prediction_horizon) + img_idx + 1).astype(np.int64)
        num_frames = self.max_video_frames[imgname.split('/')[0]]
        future_ind[future_ind >= num_frames] = num_frames - 1
        
        past_ind = (np.arange(history_size) - history_size + img_idx + 1).astype(np.int64)
        past_ind[past_ind < 0] = 0
        return past_ind, future_ind
    
    def get_variable_length_sequence(self, imgname, history_size, curr_length, max_length):
        splits = imgname.split('/')
        seqname = '/'.join(splits[-6:-2])
        img_idx = int(imgname.split("/")[-1].split(".")[0])

        future_ind = (np.arange(curr_length) + img_idx + 1).astype(np.int64)
        mask_ind = np.ones(curr_length)
        if curr_length < max_length:
            # repeat the last frame to make it max_length
            future_ind = np.concatenate([future_ind, np.repeat(future_ind[-1:], max_length - curr_length)])
            # add zero mask to indices
            mask_ind = np.concatenate([mask_ind, np.zeros(max_length - curr_length)])
        num_frames = self.max_video_frames[seqname]
        future_ind[future_ind >= num_frames] = num_frames - 1

        past_ind = (np.arange(history_size) - history_size + img_idx + 1).astype(np.int64)
        past_ind[past_ind < 0] = 0

        return past_ind, future_ind, mask_ind
    
    def get_egocam_pose(self, seqname, index):
        if not isinstance(index, str):
            index = str(index).zfill(6)
        cam_file = self.cam_pose.format(seqname, index)
        cam_pose = np.loadtxt(cam_file)
        cam_pose = cam_pose.reshape(4, 4).astype(np.float32)
        return cam_pose
    
    def get_imgname_from_index(self, seqname, index):
        if not isinstance(index, str):
            index = str(index).zfill(6)
        imgname = self.image_dir.format(seqname, index)
        assert op.exists(imgname), f"Image {imgname} does not exist"
        return imgname

    def get_img_data(self, imgname, load_rgb=True):
        splits = imgname.split('/')
        seqname = '/'.join(splits[-6:-2])
        index = splits[-1].split('.')[0]
        inputs, targets, meta_info = self.getitem(seqname, index)
        return inputs, targets, meta_info
    
    def get_future_data(self, imgname, indices):
        """
        Get future data for the given image name and future indices.
        """

        splits = imgname.split("/")
        seqname = '/'.join(splits[-6:-2])
        curr_idx = int(imgname.split("/")[-1].split(".")[0])

        cam_file = self.intrx_dir.format(seqname)
        intrx = np.loadtxt(cam_file)
        intrx = np.array([[intrx[0], 0, intrx[2]], [0, intrx[1], intrx[3]], [0, 0, 1]])
        intrx = intrx.astype(np.float32)

        args = self.args
        ######### this is added for camera rotation augmentation #########
        # check if augm_dict is cached as a class attribute
        if hasattr(self, 'augm_dict'):
            augm_dict = self.augm_dict
            cam_rot_aug = self.cam_rot_aug
        else:
            # augment parameters
            augm_dict = data_utils.augm_params(
                self.aug_data,
                args.flip_prob,
                args.noise_factor,
                args.rot_factor,
                args.scale_factor,
                debug=args.debug,
            )
            cam_rot_aug = augm_dict['rot']
            augm_dict['rot'] = 0 # this is taken care separately by cam_rot_aug
            # cache the augment parameters to use for other timesteps
            self.augm_dict = augm_dict
            self.cam_rot_aug = cam_rot_aug
        ########### end of camera rotation augmentation #########

        joints3d_r, joints3d_l = [], []
        joints2d_r, joints2d_l = [], []
        pose_r, pose_l = [], []
        betas_r, betas_l = [], []
        right_valid, left_valid = [], []
        world2future = []
        
        for ind in indices:
            if not isinstance(ind, str):
                ind = str(ind).zfill(6)
            # hand_labels = self.get_hand_labels(seqname, ind)
            hand_labels = self.hand_labels[seqname][ind]
            right_pose, right_trans, right_beta, right_joints, r_valid, \
                left_pose, left_trans, left_beta, left_joints, l_valid = hand_labels
            
            if cam_rot_aug != 0 and ('img' not in self.args.cond_mode) and ('spatial' not in self.args.cond_mode): # this only works if image is not used in conditioning
                # transform 3d quantities
                # right_pose and left_pose are 48-dimensional vectors, take the first 3 dimension for global orientation
                global_pose_r = right_pose[:3]
                global_pose_l = left_pose[:3]
                right_joints, left_joints, rightHandKps, leftHandKps, global_pose_r, global_pose_l = \
                    self.transform_3d_quantities(
                        right_joints, left_joints,
                        global_pose_r, global_pose_l,
                        intrx, theta_deg=cam_rot_aug
                    )
                # update the right and left pose with the new global pose
                right_pose[:3] = global_pose_r
                left_pose[:3] = global_pose_l
            
            right_joints2d, left_joints2d = self.get_joints2d(intrx, right_joints, left_joints)
            
            joints3d_r.append(right_joints)
            joints3d_l.append(left_joints)
            joints2d_r.append(right_joints2d)
            joints2d_l.append(left_joints2d)
            pose_r.append(right_pose)
            pose_l.append(left_pose)
            betas_r.append(right_beta)
            betas_l.append(left_beta)
            right_valid.append(r_valid)
            left_valid.append(l_valid)

            # egocam_pose = self.get_egocam_pose(seqname, ind)
            egocam_pose = self.ego_cam_pose[seqname][ind]
            world2future.append(egocam_pose)

        joints3d_r = np.stack(joints3d_r, axis=0).astype(np.float32)
        joints3d_l = np.stack(joints3d_l, axis=0).astype(np.float32)
        joints2d_r = np.stack(joints2d_r, axis=0).astype(np.float32)
        joints2d_l = np.stack(joints2d_l, axis=0).astype(np.float32)
        pose_r = np.stack(pose_r, axis=0).astype(np.float32)
        pose_l = np.stack(pose_l, axis=0).astype(np.float32)
        betas_r = np.stack(betas_r, axis=0).astype(np.float32)
        betas_l = np.stack(betas_l, axis=0).astype(np.float32)
        right_valid = np.stack(right_valid, axis=0)
        left_valid = np.stack(left_valid, axis=0)
        # right_valid and left_valid are binary labels, extend them to 21
        # this is done to be consistent with other datasets
        right_valid = np.tile(right_valid[:, None], (1, 21)).astype(np.float32)
        left_valid = np.tile(left_valid[:, None], (1, 21)).astype(np.float32)
        world2future = np.stack(world2future, axis=0).astype(np.float32)

        world2view = self.get_egocam_pose(seqname, curr_idx)[None]
        future2view = world2view @ np.linalg.inv(world2future)

        # store in a dict
        future_data = {
            "future_joints3d_r": joints3d_r,
            "future_joints3d_l": joints3d_l,
            "future_pose_r": pose_r,
            "future_betas_r": betas_r,
            "future_pose_l": pose_l,
            "future_betas_l": betas_l,
            "future.j2d.norm.r": joints2d_r,
            "future.j2d.norm.l": joints2d_l,
            "future_valid_r": right_valid,
            "future_valid_l": left_valid,
            "future2view": future2view,
        }
        return future_data

    def get_joints2d(self, intrx, right_joints, left_joints):
        rightHandKps = (intrx @ right_joints.T).T
        rightHandKps = rightHandKps[:, :2] / rightHandKps[:, 2:3]  # convert to 2D
        leftHandKps = (intrx @ left_joints.T).T
        leftHandKps = leftHandKps[:, :2] / leftHandKps[:, 2:3]  # convert to 2D
        
        image_size = {"width": self.img_w, "height": self.img_h}
        bbox = [image_size['width'] / 2, image_size['height'] / 2, max(image_size['width'], image_size['height']) / 200] # original bbox
        center = [bbox[0], bbox[1]]
        scale = bbox[2]
        
        args = self.args
        if hasattr(self, 'augm_dict'):
            augm_dict = self.augm_dict
        else:
            # augment parameters
            augm_dict = data_utils.augm_params(
                self.aug_data,
                args.flip_prob,
                args.noise_factor,
                args.rot_factor,
                args.scale_factor,
                debug=args.debug,
            )
            augm_dict['rot'] = 0 # this is taken care separately by cam_rot_aug
        
        j2d_r = pad_jts2d(rightHandKps)
        j2d_l = pad_jts2d(leftHandKps)

        j2d_r = data_utils.j2d_processing(
            j2d_r, center, scale, augm_dict, args.img_res
        )
        j2d_l = data_utils.j2d_processing(
            j2d_l, center, scale, augm_dict, args.img_res
        )

        return j2d_r[..., :2], j2d_l[..., :2]
    
    def transform_3d_quantities(self,
                            joints3d_r,   # (N,3) or (T, N, 3)
                            joints3d_l,   # (N,3) or (T, N, 3)
                            rot_r,        # (3,)   or (T, 3)
                            rot_l,        # (3,)   or (T, 3)
                            K,            # (3, 3)
                            theta_deg=0.0):
        """
        Rotate 3D joints and camera rotations by theta_deg degrees clockwise about Z,
        then reproject into 2D using shared intrinsics K.
        
        Inputs may be single-frame:
        joints3d_* : (N, 3), rot_* : (3,)
        or multi-frame:
        joints3d_* : (T, N, 3), rot_* : (T, 3)
        """
        # --- 0) Handle single- vs multi-frame ---
        single_frame = (joints3d_r.ndim == 2)
        if single_frame:
            # add time dim
            joints3d_r = joints3d_r[None, ...]
            joints3d_l = joints3d_l[None, ...]
            rot_r      = rot_r[None, ...]
            rot_l      = rot_l[None, ...]
        
        T, Nr, _ = joints3d_r.shape
        _, Nl, _ = joints3d_l.shape

        # --- 1) Build 4×4 clockwise Z-rotation ---
        θ = np.deg2rad(theta_deg)
        Rz = np.array([
            [ np.cos(θ),  np.sin(θ), 0],
            [-np.sin(θ),  np.cos(θ), 0],
            [         0,          0, 1]
        ], dtype=np.float32)
        R4 = np.eye(4, dtype=np.float32)
        R4[:3, :3] = Rz

        # --- 2) Rotate batched 3D joints ---
        def _rotate_batch(j3):
            flat = j3.reshape(-1, 3).astype(np.float32)           # (T*Ni, 3)
            homo = np.concatenate([flat, np.ones((flat.shape[0],1),dtype=np.float32)], axis=1)  # (T*Ni,4)
            rot  = (homo @ R4.T)[:, :3]                           # (T*Ni,3)
            return rot.reshape(T, -1, 3)                          # (T, Ni, 3)

        j3r_rot = _rotate_batch(joints3d_r)
        j3l_rot = _rotate_batch(joints3d_l)

        # --- 3) Reproject into 2D per frame ---
        def _project_batch(j3_rot):
            uv_seq = []
            for t in range(T):
                pts = j3_rot[t]                     # (Ni,3)
                proj = (K @ pts.T).T                # (Ni,3)
                uv   = proj[:, :2] / proj[:, 2:3]   # (Ni,2)
                uv_seq.append(uv)
            return np.stack(uv_seq, axis=0)        # (T, Ni, 2)

        j2r = _project_batch(j3r_rot)
        j2l = _project_batch(j3l_rot)

        # --- 4) Rotate axis-angle vectors per frame ---
        def _rotate_axes(axes):
            aa_seq = []
            for t in range(T):
                aa = axes[t].astype(np.float32)                        # (3,)
                M  = rot_conv.axis_angle_to_matrix(
                        torch.from_numpy(aa)[None]
                    ).numpy()[0]                                     # (3,3)
                M4 = np.eye(4, dtype=np.float32); M4[:3,:3] = M        # (4,4)
                M4t = R4 @ M4                                          # (4,4)
                aa_new = rot_conv.matrix_to_axis_angle(
                            torch.from_numpy(M4t[:3,:3])[None]
                        ).numpy()[0]                                 # (3,)
                aa_seq.append(aa_new)
            return np.stack(aa_seq, axis=0)                           # (T, 3)

        rot_r_new = _rotate_axes(rot_r)
        rot_l_new = _rotate_axes(rot_l)

        # --- 5) Squeeze time dim if single frame ---
        if single_frame:
            j3r_rot, j3l_rot = j3r_rot[0], j3l_rot[0]   # → (N,3)
            j2r,     j2l     = j2r[0],     j2l[0]       # → (N,2)
            rot_r_new, rot_l_new = rot_r_new[0], rot_l_new[0]  # → (3,)

        return j3r_rot, j3l_rot, j2r, j2l, rot_r_new, rot_l_new
    
    def __getitem__(self, idx):
        seqname, index = self.samples[idx]
        data = self.getitem(seqname, index)
        return data
    
    def get_hand_labels(self, seqname, index):
        hand_file = self.hand_dir.format(seqname, index) # txt file
        mano_file = self.mano_dir.format(seqname, index) # txt file
        
        hand_info = np.loadtxt(hand_file)
        left_hand, right_hand = hand_info[:64], hand_info[64:]
        left_valid = left_hand[0]
        left_joints = left_hand[1:64].reshape(21,3)
        left_joints = left_joints[self.h2o_to_mano_ordering]
        right_valid = right_hand[0]
        right_joints = right_hand[1:64].reshape(21,3)
        right_joints = right_joints[self.h2o_to_mano_ordering]

        mano_info = np.loadtxt(mano_file)
        left_mano, right_mano = mano_info[:62], mano_info[62:]
        left_mano_valid = left_mano[0]
        left_trans = left_mano[1:4]
        left_pose = left_mano[4:4+48]
        left_beta = left_mano[4+48:]
        right_mano_valid = right_mano[0]
        right_trans = right_mano[1:4]
        right_pose = right_mano[4:4+48]
        right_beta = right_mano[4+48:]

        right_joints = right_joints.astype(np.float32)
        left_joints = left_joints.astype(np.float32)
        right_beta = right_beta.astype(np.float32)
        right_pose = right_pose.astype(np.float32)
        right_trans = right_trans.astype(np.float32)
        left_beta = left_beta.astype(np.float32)
        left_pose = left_pose.astype(np.float32)
        left_trans = left_trans.astype(np.float32)

        return right_pose, right_trans, right_beta, right_joints, right_valid, \
                left_pose, left_trans, left_beta, left_joints, left_valid

    def getitem(self, seqname, index):
        cv_img, _ = read_img(self.image_dir.format(seqname, index), (2800, 2000, 3))

        cam_file = self.intrx_dir.format(seqname)
        intrx = np.loadtxt(cam_file)
        intrx = np.array([[intrx[0], 0, intrx[2]], [0, intrx[1], intrx[3]], [0, 0, 1]])
        intrx = intrx.astype(np.float32)

        # hand_labels = self.get_hand_labels(seqname, index)
        hand_labels = self.hand_labels[seqname][index]
        right_pose, right_trans, right_beta, right_joints, right_valid, \
        left_pose, left_trans, left_beta, left_joints, left_valid = hand_labels

        j3d = torch.from_numpy(right_joints).unsqueeze(0)
        K = torch.from_numpy(intrx).unsqueeze(0)
        j2d = tf.project2d_batch(K, j3d)
        rightHandKps = j2d.squeeze(0).numpy()

        j3d = torch.from_numpy(left_joints).unsqueeze(0)
        K = torch.from_numpy(intrx).unsqueeze(0)
        j2d = tf.project2d_batch(K, j3d)
        leftHandKps = j2d.squeeze(0).numpy()

        args = self.args
        ######### this is added for camera rotation augmentation #########
        # check if augm_dict is cached as a class attribute
        if hasattr(self, 'augm_dict'):
            augm_dict = self.augm_dict
            cam_rot_aug = self.cam_rot_aug
        else:
            # augment parameters
            augm_dict = data_utils.augm_params(
                self.aug_data,
                args.flip_prob,
                args.noise_factor,
                args.rot_factor,
                args.scale_factor,
                debug=args.debug,
            )
            cam_rot_aug = augm_dict['rot']
            augm_dict['rot'] = 0 # this is taken care separately by cam_rot_aug
            # cache the augment parameters to use for other timesteps
            self.augm_dict = augm_dict
            self.cam_rot_aug = cam_rot_aug
        
        if cam_rot_aug != 0 and ('img' not in self.args.cond_mode) and ('spatial' not in self.args.cond_mode): # this only works if image is not used in conditioning
            # transform 3d quantities
            # right_pose and left_pose are 48-dimensional vectors, take the first 3 dimension for global orientation
            global_pose_r = right_pose[:3]
            global_pose_l = left_pose[:3]
            right_joints, left_joints, rightHandKps, leftHandKps, global_pose_r, global_pose_l = \
                self.transform_3d_quantities(
                    right_joints, left_joints,
                    global_pose_r, global_pose_l,
                    intrx, theta_deg=cam_rot_aug
                )
            # update the right and left pose with the new global pose
            right_pose[:3] = global_pose_r
            left_pose[:3] = global_pose_l
        ########### end of camera rotation augmentation #########

        image_size = {"width": cv_img.shape[1], "height": cv_img.shape[0]}
        bbox = [image_size['width'] / 2, image_size['height'] / 2, max(image_size['width'], image_size['height']) / 200] # original bbox
        # is_egocam = True
        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        
        use_gt_k = True

        joints2d_r = pad_jts2d(rightHandKps)
        joints2d_l = pad_jts2d(leftHandKps)

        joints2d_r = data_utils.j2d_processing(
            joints2d_r, center, scale, augm_dict, args.img_res
        )
        joints2d_l = data_utils.j2d_processing(
            joints2d_l, center, scale, augm_dict, args.img_res
        )

        img = data_utils.rgb_processing(
                self.aug_data,
                cv_img,
                center,
                scale,
                augm_dict,
                img_res=args.img_res,
            )
        img = torch.from_numpy(img).float()
        norm_img = self.normalize_img(img)

        inputs = {}
        targets = {}
        meta_info = {}

        inputs["img"] = norm_img        
        
        meta_info["imgname"] = self.get_imgname_from_index(seqname, index)
        meta_info["query_names"] = '' # dummy value
        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        
        targets = {}
        targets['mano.pose.r'] = torch.FloatTensor(right_pose)
        targets['mano.pose.l'] = torch.FloatTensor(left_pose)
        targets['mano.beta.r'] = torch.FloatTensor(right_beta)
        targets['mano.beta.l'] = torch.FloatTensor(left_beta)
        targets['mano.j2d.norm.r'] = torch.from_numpy(joints2d_r[:, :2]).float() # torch.FloatTensor(rightJoints2d)
        targets['mano.j2d.norm.l'] = torch.from_numpy(joints2d_l[:, :2]).float() # torch.FloatTensor(leftJoints2d)
        targets['mano.j3d.full.r'] = torch.FloatTensor(right_joints)
        targets['mano.j3d.full.l'] = torch.FloatTensor(left_joints)
        
        meta_info["query_names"] = '' # dummy value
        meta_info["window_size"] = torch.LongTensor(np.array([args.window_size]))

        # scale and center in the original image space
        scale_original = max([image_size["width"], image_size["height"]]) / 200.0
        center_original = [image_size["width"] / 2.0, image_size["height"] / 2.0]
        fixed_focal_length = args.focal_length
        intrx = data_utils.get_aug_intrix(
            intrx,
            fixed_focal_length,
            args.img_res,
            use_gt_k,
            center_original[0],
            center_original[1],
            augm_dict["sc"] * scale_original,
        )

        if cam_rot_aug != 0:
            self.intrx = intrx

        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        meta_info["dist"] = torch.FloatTensor(torch.zeros(8)) # dummy value for distortion params
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["rot_angle"] = np.float32(augm_dict["rot"])
        meta_info['loss_mask'] = 1
        meta_info['dataset'] = 'h2o'
        # meta_info["sample_index"] = index

        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info['is_j2d_loss'] = args.get('finetune_2d', 0) # 'future_j2d' in self.args.get('cond_mode','no_cond') # shouldn't matter since we have 3D labels available
        meta_info['is_j3d_loss'] = 1
        meta_info['is_beta_loss'] = 1
        meta_info['is_pose_loss'] = 1
        meta_info['is_cam_loss'] = 1

        targets['is_valid'] = (left_valid + right_valid) > 0
        targets['left_valid'] = left_valid
        targets['right_valid'] = right_valid
        targets["joints_valid_r"] = np.ones(21) * targets["right_valid"]
        targets["joints_valid_l"] = np.ones(21) * targets["left_valid"]

        return inputs, targets, meta_info


if __name__ == '__main__':
    from common.xdict import xdict
    args = xdict()
    args.img_res = 224
    args.img_norm_mean = [0.485, 0.456, 0.406]
    args.img_norm_std = [0.229, 0.224, 0.225]
    dat = H2ODataset(args)
    print(dat[0])
    
