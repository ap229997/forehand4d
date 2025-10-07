import os
import yaml
import pickle
import os.path as op

import numpy as np
from tqdm import tqdm
import torch
from torchvision.transforms import Normalize
import pytorch3d.transforms.rotation_conversions as rot_conv

import common.data_utils as data_utils
from common.data_utils import read_img
from src.datasets.dataset_utils import pad_jts2d
from src.datasets.dexycb_utils import _SERIALS, DEXYCB_DIR, dexycb_to_mano_ordering, get_hand_labels
from src.datasets.base_dataset import BaseDataset


class DexYCBDataset(BaseDataset):
    def __init__(self, args, split='train'):
        super().__init__()
        self.args = args
        if 'train' in split:
            self.split = 'train'
        elif 'val' in split:
            self.split = 'val'
        else:
            self.split = 'test'

        self._data_dir = DEXYCB_DIR
        self._calib_dir = op.join(self._data_dir, "calibration")
        self._model_dir = op.join(self._data_dir, "models")

        self._color_format = "color_{:06d}.jpg"
        self._depth_format = "aligned_depth_to_color_{:06d}.png"
        self._label_format = "labels_{:06d}.npz"
        self.img_h = 480
        self.img_w = 640

        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        self._serials = [_SERIALS[i] for i in serial_ind]
        self._intrinsics = []
        for s in self._serials:
            intr_file = os.path.join(self._calib_dir, "intrinsics",
                                    "{}_{}x{}.yml".format(s, self.img_w, self.img_h))
            with open(intr_file, 'r') as f:
                intr = yaml.load(f, Loader=yaml.FullLoader)
            intr = intr['color']
            self._intrinsics.append(intr)

        self._load_motion_data()

        self.samples = []
        self.max_video_frames = {}
        for i in range(len(self.motion_data['names'])):
            video_name = self.motion_data['names'][i]
            start_idx, end_idx = self.motion_data['ranges'][i]
            if self.args.max_motion_length > 15:
                start_idx = 15 # first 15 frames typically don't have hands
            self.motion_data['ranges'][i] = (start_idx, end_idx) 
            self.samples.append((video_name, start_idx))
            
            if video_name in self.max_video_frames:
                continue
            # video_dir = op.join(self._data_dir, video_name)
            total_frames = end_idx - start_idx + 1 # len(glob(op.join(video_dir, 'color_*.jpg')))
            self.max_video_frames[video_name] = total_frames
        self.imgnames = self.samples

        self.dexycb_to_mano_ordering = dexycb_to_mano_ordering
        self.get_hand_labels = get_hand_labels
        self._load_hand_labels()

        self.aug_data = split.endswith("train")
        self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)
    
    def _load_motion_data(self):
        if self.args.get('use_fixed_length', False):
            data_file = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/dexycb/fixed_{self.args.max_motion_length:03d}/dexycb/s3_{self.split}.pkl'
        else:
            data_file = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/dexycb/s3_{self.split}.pkl'
        with open(data_file, 'rb') as f:
            self.motion_data = pickle.load(f)

    def _load_hand_labels(self):
        save_dir = f"{os.environ['DOWNLOADS_DIR']}/motion_splits/dexycb"
        hand_label_file = os.path.join(save_dir, f'hand_labels_s3_{self.split}.pkl')
        if os.path.exists(hand_label_file):
            with open(hand_label_file, 'rb') as f:
                self.hand_labels = pickle.load(f)
            print ('Loaded hand labels from disk')
        
        else:
            self.hand_labels = {}
            for ind in tqdm(range(len(self.motion_data['names']))):
                seqname = self.motion_data['names'][ind]
                start_idx, end_idx = self.motion_data['ranges'][ind]
                for frame_idx in range(start_idx, end_idx + 1):
                    label_file = os.path.join(self._data_dir, seqname, self._label_format.format(frame_idx))
                    meta_file = os.path.join(self._data_dir, '/'.join(seqname.split('/')[:2]), "meta.yml")
                    hand_labels = self.get_hand_labels(meta_file, label_file)
                    if seqname not in self.hand_labels:
                        self.hand_labels[seqname] = {}
                    self.hand_labels[seqname][frame_idx] = hand_labels

            # save self.hand_labels and self.ego_cam_pose to disk in pickle format
            os.makedirs(save_dir, exist_ok=True)
            with open(hand_label_file, 'wb') as f:
                pickle.dump(self.hand_labels, f)

            print ('Saved hand labels to {}'.format(save_dir))

    def __len__(self):
        return len(self.samples)
    
    def get_fixed_length_sequence(self, imgname, history_size, prediction_horizon):
        splits = imgname.split('/')
        img_idx = int(splits[-1].split('_')[-1].split(".")[0])

        future_ind = (np.arange(prediction_horizon) + img_idx + 1).astype(np.int64)
        num_frames = self.max_video_frames[imgname.split('/')[0]]
        future_ind[future_ind >= num_frames] = num_frames - 1
        
        past_ind = (np.arange(history_size) - history_size + img_idx + 1).astype(np.int64)
        past_ind[past_ind < 0] = 0
        return past_ind, future_ind
    
    def get_variable_length_sequence(self, imgname, history_size, curr_length, max_length):
        splits = imgname.split('/')
        seqname = '/'.join(splits[-4:-1])
        img_idx = int(splits[-1].split('_')[-1].split(".")[0])

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

    def get_imgname_from_index(self, seqname, index):
        imgname = op.join(self._data_dir, seqname, self._color_format.format(index))
        assert op.exists(imgname), f"Image {imgname} does not exist"
        return imgname
    
    def get_img_data(self, imgname, load_rgb=True):
        splits = imgname.split('/')
        seqname = '/'.join(splits[-4:-1])
        index = int(splits[-1].split('_')[-1].split('.')[0])
        inputs, targets, meta_info = self.getitem(seqname, index)
        return inputs, targets, meta_info
    
    def get_future_data(self, imgname, indices):
        """
        Get future data for the given image name and future indices.
        """

        splits = imgname.split("/")
        seqname = '/'.join(splits[-4:-1])
        curr_idx = int(splits[-1].split("_")[-1].split(".")[0])

        ######## this is added for camera rotation augmentation ########
        # modify 3D quantities as per the augmentation parameters
        # check if augm_dict is cached as a class attribute
        if hasattr(self, 'augm_dict'):
            augm_dict = self.augm_dict
            cam_rot_aug = self.cam_rot_aug
        else:
            # augment parameters
            args = self.args
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
        intrx_idx = _SERIALS.index(seqname.split('/')[-1])
        intrx = self._intrinsics[intrx_idx]
        intrx = np.array([[intrx['fx'], 0, intrx['ppx']],
                        [0, intrx['fy'], intrx['ppy']],
                        [0, 0, 1]], dtype=np.float32)
        ########### these are needed for camera rotation augmentation #########

        joints3d_r, joints3d_l = [], []
        joints2d_r, joints2d_l = [], []
        pose_r, pose_l = [], []
        betas_r, betas_l = [], []
        right_valid, left_valid = [], []
        future2view = []
        
        for ind in indices:
            hand_labels = self.hand_labels[seqname][ind]
            right_pose, right_beta, right_transl, right_joints3d, right_joints2d, r_valid, \
                left_pose, left_beta, left_transl, left_joints3d, left_joints2d, l_valid = hand_labels
            
            if cam_rot_aug != 0 and ('img' not in self.args.cond_mode) and ('spatial' not in self.args.cond_mode): # this only works if image is not used in conditioning
                # transform 3d quantities
                # right_pose and left_pose are 48-dimensional vectors, take the first 3 dimension for global orientation
                global_pose_r = left_pose[..., :3]
                global_pose_l = right_pose[..., :3]
                right_joints3d, left_joints3d, right_joints2d, left_joints2d, global_pose_r, global_pose_l = \
                    self.transform_3d_quantities(
                        right_joints3d, left_joints3d,
                        global_pose_r, global_pose_l,
                        intrx, theta_deg=cam_rot_aug
                    )
                # update the right and left pose with the new global pose
                left_pose[..., :3] = global_pose_r
                right_pose[..., :3] = global_pose_l
            
            right_joints2d, left_joints2d = self.process_joints2d(right_joints2d, left_joints2d)

            joints3d_r.append(right_joints3d)
            joints3d_l.append(left_joints3d)
            joints2d_r.append(right_joints2d)
            joints2d_l.append(left_joints2d)
            pose_r.append(right_pose)
            pose_l.append(left_pose)
            betas_r.append(right_beta)
            betas_l.append(left_beta)
            right_valid.append(r_valid)
            left_valid.append(l_valid)
            future2view.append(np.eye(4).astype(np.float32))

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
        future2view = np.stack(future2view, axis=0).astype(np.float32)

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
    
    def process_joints2d(self, right_joints2d, left_joints2d):
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
            )
            augm_dict['rot'] = 0

        j2d_r = pad_jts2d(right_joints2d)
        j2d_l = pad_jts2d(left_joints2d)

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
    
    def getitem(self, seqname, index):
        intrx_idx = _SERIALS.index(seqname.split('/')[-1])
        intrx = self._intrinsics[intrx_idx]
        intrx = np.array([[intrx['fx'], 0, intrx['ppx']],
                              [0, intrx['fy'], intrx['ppy']],
                              [0, 0, 1]], dtype=np.float32)

        hand_labels = self.hand_labels[seqname][index]
        right_pose, right_beta, right_transl, right_joints3d, right_joints2d, right_valid, \
            left_pose, left_beta, left_transl, left_joints3d, left_joints2d, left_valid = hand_labels
        
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
            right_joints3d, left_joints3d, right_joints2d, left_joints2d, global_pose_r, global_pose_l = \
                self.transform_3d_quantities(
                    right_joints3d, left_joints3d,
                    global_pose_r, global_pose_l,
                    intrx, theta_deg=cam_rot_aug
                )
            # update the right and left pose with the new global pose
            right_pose[:3] = global_pose_r
            left_pose[:3] = global_pose_l
        ########### end of camera rotation augmentation #########

        image_size = {"width": self.img_w, "height": self.img_h}
        bbox = [image_size['width'] / 2, image_size['height'] / 2, max(image_size['width'], image_size['height']) / 200] # original bbox
        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        use_gt_k = True

        joints2d_r = pad_jts2d(right_joints2d)
        joints2d_l = pad_jts2d(left_joints2d)

        joints2d_r = data_utils.j2d_processing(
            joints2d_r, center, scale, augm_dict, args.img_res
        )
        joints2d_l = data_utils.j2d_processing(
            joints2d_l, center, scale, augm_dict, args.img_res
        )

        rgb_file = op.join(self._data_dir, seqname, self._color_format.format(index))
        cv_img, _ = read_img(rgb_file, (self.img_w, self.img_h, 3))

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
        
        meta_info["imgname"] = rgb_file # self.get_imgname_from_index(seqname, index)
        meta_info["query_names"] = '' # dummy value
        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        
        targets = {}
        targets['mano.pose.r'] = torch.FloatTensor(right_pose)
        targets['mano.pose.l'] = torch.FloatTensor(left_pose)
        targets['mano.beta.r'] = torch.FloatTensor(right_beta)
        targets['mano.beta.l'] = torch.FloatTensor(left_beta)
        targets['mano.j2d.norm.r'] = torch.from_numpy(joints2d_r[:, :2]).float()
        targets['mano.j2d.norm.l'] = torch.from_numpy(joints2d_l[:, :2]).float()
        targets['mano.j3d.full.r'] = torch.FloatTensor(right_joints3d)
        targets['mano.j3d.full.l'] = torch.FloatTensor(left_joints3d)
        
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
        meta_info['dataset'] = 'dexycb'
        # meta_info["sample_index"] = index

        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info['is_j2d_loss'] = 0 # args.get('finetune_2d', 0) # shouldn't matter since 3d labels are available
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