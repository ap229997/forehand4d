import os
import pickle

from glob import glob
import numpy as np
import torch
from torchvision.transforms import Normalize

import common.data_utils as data_utils
from src.datasets.dataset_utils import pad_jts2d, downsample
from src.datasets.base_dataset import BaseDataset


class EgoExoDataset(BaseDataset):
    def __init__(self, args, mode='test') -> None:
        super().__init__()

        mode = 'val' # training not supported
        self.args = args
        self.split = mode
        self.aug_data = self.split.endswith("train")
        self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)
        self.egocam_k = None

        self.image_dir = os.path.join(os.environ['DOWNLOADS_DIR'], "data/egoexo/raw_frames/image/undistorted/{}/{}/{}.jpg")

        self._load_motion_data()

        self.samples = []
        self.video_info = {}
        # every sequence starts with id 0000, skip that
        for i in range(len(self.motion_data['names'])):
            curr_name = self.motion_data['names'][i]
            video_name = curr_name[0]
            # start_idx = self.motion_data['ranges'][i][0]
            subsampled_indices = [str(x).zfill(6) for x in self.motion_data['subsampled_indices'][i]]
            st = self.motion_data['ranges'][i][0]
            start_idx = subsampled_indices[st]
            self.samples.append((video_name, start_idx))
            total_frames = len(subsampled_indices)
            if video_name not in self.video_info:
                self.video_info[video_name] = {}
            # for every index in subsampled_indices, store the corresponding index in the original video
            for index in subsampled_indices:
                self.video_info[video_name][index] = i
        self.imgnames = self.samples

        self._load_hand_labels()

        # mean values of beta, computed from val set of arctic, used for datasets without MANO fits
        # can also use default beta values in MANO, either is fine as long as it is consistent across training
        self.mean_beta_r = [0.82747316,  0.13775729, -0.39435294, 0.17889787, -0.73901576, 0.7788163, -0.5702684, 0.4947751, -0.24890041, 1.5943261]
        self.mean_beta_l = [-0.19330633, -0.08867972, -2.5790455, -0.10344583, -0.71684015, -0.28285977, 0.55171007, -0.8403888, -0.8490544, -1.3397144]

        # define joint ordering
        self.index2joints = {0: 'wrist', 1: 'index_1', 2: 'index_2', 3: 'index_3', 4: 'middle_1', 5: 'middle_2', 6: 'middle_3', 7: 'pinky_1', 8: 'pinky_2', 9: 'pinky_3',
                    10: 'ring_1', 11: 'ring_2', 12: 'ring_3', 13: 'thumb_1', 14: 'thumb_2', 15: 'thumb_3', 16: 'thumb_4', 17: 'index_4', 18: 'middle_4', 19: 'ring_4', 20: 'pinky_4'}
        self.joints2index = {v: k for k, v in self.index2joints.items()}

        # subsample indices
        all_keys = list(range(len(self.imgnames)))
        self.subsampled_keys = downsample(all_keys, mode)

        print ("Number of samples in EgoExo %s: %d" % (mode, len(self.subsampled_keys)))

    def _load_motion_data(self):
        if self.args.get('use_fixed_length', False):
            data_file = glob(f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/egoexo/fixed_{self.args.max_motion_length:03d}/egoexo/{self.split}*.pkl')
        else:
            data_file = glob(f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/egoexo/{self.split}*.pkl')
        data_file = sorted(data_file)[-1]
        if not os.path.exists(data_file):
            print(f"Data file {data_file} does not exist")
            exit(1)
        with open(data_file, 'rb') as f:
            self.motion_data = pickle.load(f) # dict of lists

    def _load_hand_labels(self):
        save_dir = os.path.join(os.environ['DOWNLOADS_DIR'], 'motion_splits/egoexo')
        hand_label_file = os.path.join(save_dir, f'hand_labels_{self.split}.pkl')
        cam_pose_file = os.path.join(save_dir, f'cam_pose_{self.split}.pkl')
        if os.path.exists(hand_label_file) and os.path.exists(cam_pose_file):
            with open(hand_label_file, 'rb') as f:
                self.hand_labels = pickle.load(f)
            with open(cam_pose_file, 'rb') as f:
                self.cam_info = pickle.load(f)
            print ('Loaded hand labels from disk')

    def get_fixed_length_sequence(self, imgname, history_size, prediction_horizon):
        splits = imgname.split('/')
        seqname = splits[0]
        img_idx = splits[-1].zfill(6)
        
        # get corresponding index in motion_data
        motion_idx = self.video_info[seqname][img_idx]
        motion_range = self.motion_data['ranges'][motion_idx]
        range_indices = self.motion_data['subsampled_indices'][motion_idx]
        img_idx = range_indices.index(int(img_idx))

        future_ind = (np.arange(prediction_horizon) + img_idx + 1).astype(np.int64)
        num_frames = motion_range[1] + 1
        future_ind[future_ind >= num_frames] = num_frames - 1
        
        past_ind = (np.arange(history_size) - history_size + img_idx + 1).astype(np.int64)
        past_ind[past_ind < 0] = 0
        return past_ind, future_ind
    
    def get_variable_length_sequence(self, imgname, history_size, curr_length, max_length):
        splits = imgname.split('/')
        seqname = splits[0]
        img_idx = splits[-1].zfill(6)

        # get corresponding index in motion_data
        motion_idx = self.video_info[seqname][img_idx]
        motion_range = self.motion_data['ranges'][motion_idx]
        range_indices = self.motion_data['subsampled_indices'][motion_idx]
        img_idx = range_indices.index(int(img_idx))

        future_ind = (np.arange(curr_length) + img_idx + 1).astype(np.int64)
        mask_ind = np.ones(curr_length)
        if curr_length < max_length:
            # repeat the last frame to make it max_length
            future_ind = np.concatenate([future_ind, np.repeat(future_ind[-1:], max_length - curr_length)])
            # add zero mask to indices
            mask_ind = np.concatenate([mask_ind, np.zeros(max_length - curr_length)])
        num_frames = motion_range[1] + 1
        future_ind[future_ind >= num_frames] = num_frames - 1

        past_ind = (np.arange(history_size) - history_size + img_idx + 1).astype(np.int64)
        past_ind[past_ind < 0] = 0

        return past_ind, future_ind, mask_ind
    
    def get_imgname_from_index(self, seqname, index):
        if isinstance(seqname, tuple):
            video_name, img_idx = seqname
            motion_idx = self.video_info[video_name][img_idx]
            index = self.motion_data['subsampled_indices'][motion_idx][index]
            if not isinstance(index, str):
                index = str(index).zfill(6)
        else:
            video_name = seqname
        
        imgname = video_name + '/' + index.lstrip("0")
        assert imgname in self.hand_labels
        return imgname
    
    def get_img_data(self, imgname, load_rgb=True):
        splits = imgname.split('/')
        seqname = splits[0]
        index = splits[-1].zfill(6)
        inputs, targets, meta_info = self.getitem(seqname, index)
        return inputs, targets, meta_info
    
    def get_future_data(self, imgname, indices):
        """
        Get future data for the given image name and future indices.
        """

        splits = imgname.split("/")
        seqname = splits[0]
        curr_idx = splits[-1].zfill(6)

        joints2d_r, joints2d_l = [], []
        joints3d_r, joints3d_l = [], []
        pose_r, pose_l = [], []
        betas_r, betas_l = [], []
        right_valid, left_valid = [], []
        world2future = []
        
        for j, ind in enumerate(indices):
            if not isinstance(ind, str):
                motion_idx = self.video_info[seqname][curr_idx]
                range_indices = self.motion_data['subsampled_indices'][motion_idx]
                ind = str(range_indices[ind])
            
            if j == 0 or indices[j] != indices[j-1]:
                # hand_labels = self.get_hand_labels(seqname, ind)
                imgname = seqname + '/' + ind.lstrip("0")
                data = self.hand_labels[imgname]

                joint3d = data['j3d'].copy()
                right_joints, left_joints, r_valid, l_valid = self.get_joints3d_data(joint3d)
                joint2d = data['j2d'].copy()
                right_j2d, left_j2d, _, _ = self.get_joints2d_data(joint2d)
                right_j2d, left_j2d = self.process_joints2d(right_j2d, left_j2d, data.copy())
                right_j2d = right_j2d[..., :2]
                left_j2d = left_j2d[..., :2]
                right_pose, left_pose = np.zeros((48,)), np.zeros((48,))
                right_betas, left_betas = self.mean_beta_r, self.mean_beta_l
            else:
                # end of unique indices, repeat the last one for the rest
                pass
            
            joints3d_r.append(right_joints)
            joints3d_l.append(left_joints)
            joints2d_r.append(right_j2d)
            joints2d_l.append(left_j2d)
            pose_r.append(right_pose)
            pose_l.append(left_pose)
            betas_r.append(right_betas)
            betas_l.append(left_betas)
            right_valid.append(r_valid)
            left_valid.append(l_valid)

            egocam_pose = np.array(self.cam_info[seqname][ind.lstrip("0")]).astype(np.float32) # 3 x 4 matrix
            # in the current coord system, x is y and y is -x of original coord system
            # change the egocam pose to the curr coord system
            egocam_pose = self.apply_xy_rotation(egocam_pose)
            egocam_pose = np.vstack([egocam_pose, np.array([0, 0, 0, 1])])
            world2future.append(egocam_pose)

        joints3d_r = np.stack(joints3d_r, axis=0).astype(np.float32)
        joints3d_l = np.stack(joints3d_l, axis=0).astype(np.float32)
        joints2d_r = np.stack(joints2d_r, axis=0).astype(np.float32)
        joints2d_l = np.stack(joints2d_l, axis=0).astype(np.float32)
        pose_r = np.stack(pose_r, axis=0).astype(np.float32)
        pose_l = np.stack(pose_l, axis=0).astype(np.float32)
        betas_r = np.stack(betas_r, axis=0).astype(np.float32)
        betas_l = np.stack(betas_l, axis=0).astype(np.float32)
        right_valid = np.stack(right_valid, axis=0).astype(np.float32)
        left_valid = np.stack(left_valid, axis=0).astype(np.float32)
        world2future = np.stack(world2future, axis=0).astype(np.float32)

        world2view = np.array(self.cam_info[seqname][curr_idx.lstrip("0")]).astype(np.float32) # 3 x 4 matrix
        world2view = self.apply_xy_rotation(world2view)
        world2view = np.vstack([world2view, np.array([0, 0, 0, 1])]).astype(np.float32)
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
    
    def apply_xy_rotation(self, T):
        """
        Applies a 90° clockwise rotation in the XY plane to a 3x4 transformation matrix T.

        Args:
            T (np.ndarray): A 3x4 transformation matrix, where T = [R | t].
                            R is 3x3 and t is 3x1.

        Returns:
            np.ndarray: A 3x4 transformed matrix T_new = [R_xy * R | R_xy * t],
                        where R_xy rotates points 90° clockwise in the XY plane.
        """
        # Separate rotation (R) and translation (t)
        R = T[:, :3]    # Shape: (3,3)
        t = T[:, 3]     # Shape: (3,)

        # Define the 3D rotation matrix for 90° clockwise rotation in the XY plane:
        R_xy = np.array([
            [ 0, 1, 0],
            [-1, 0, 0],
            [ 0, 0, 1]
        ])

        # Apply the rotation: new rotation and translation
        R_new = R_xy @ R
        t_new = R_xy @ t

        # Reassemble the transformed transformation matrix
        T_new = np.hstack((R_new, t_new[:, None]))
        return T_new
    
    def get_joints3d_data(self, j3d_data):
        joints3d_r, joints3d_l = [], []
        joints3d_valid_r, joints3d_valid_l = [], []
        for idx in self.index2joints:
            joint = self.index2joints[idx]
            for hand_type in ['left', 'right']:
                curr = hand_type+'_'+joint
                if curr in j3d_data:
                    curr_j3d = j3d_data[curr]
                    new_j3d = [curr_j3d['x'], curr_j3d['y'], curr_j3d['z']]
                    new_valid = 1
                else:
                    new_j3d = [0, 0, 0]
                    new_valid = 0
                if hand_type == 'left':
                    joints3d_l.append(new_j3d)
                    joints3d_valid_l.append(new_valid)
                else:
                    joints3d_r.append(new_j3d)
                    joints3d_valid_r.append(new_valid)
        joints3d_r = np.array(joints3d_r).copy()
        joints3d_l = np.array(joints3d_l).copy()
        joints3d_valid_r = np.array(joints3d_valid_r).copy() # only for egoexo
        joints3d_valid_l = np.array(joints3d_valid_l).copy()
        
        return joints3d_r, joints3d_l, joints3d_valid_r, joints3d_valid_l
    
    def get_joints2d_data(self, j2d_data):
        joints2d_r, joints2d_l = [], []
        joints_valid_r, joints_valid_l = [], []
        for idx in self.index2joints:
            joint =  self.index2joints[idx]
            for hand_type in ['left', 'right']:
                curr = hand_type+'_'+joint
                if curr in j2d_data:
                    curr_j2d = j2d_data[curr]
                    new_j2d = [curr_j2d['x'], curr_j2d['y']]
                    new_valid = 1
                else:
                    new_j2d = [0, 0]
                    new_valid = 0
                if hand_type == 'left':
                    joints2d_l.append(new_j2d)
                    joints_valid_l.append(new_valid)
                else:
                    joints2d_r.append(new_j2d)
                    joints_valid_r.append(new_valid)

        joints2d_r = pad_jts2d(np.array(joints2d_r).copy())
        joints2d_l = pad_jts2d(np.array(joints2d_l).copy())
        joints_valid_r = np.array(joints_valid_r).copy()
        joints_valid_l = np.array(joints_valid_l).copy()

        return joints2d_r, joints2d_l, joints_valid_r, joints_valid_l
    
    def process_joints2d(self, joints2d_r, joints2d_l, data):
        height, width = data['crop_size']
        image_size = {"width": width, "height": height}

        bbox = [image_size['width'] / 2, image_size['height'] / 2, max(image_size['width'], image_size['height']) / 200] # original bbox

        center = [bbox[0], bbox[1]]
        scale = bbox[2]
        args = self.args
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )

        joints2d_r = data_utils.j2d_processing(
            joints2d_r, center, scale, augm_dict, args.img_res
        )
        joints2d_l = data_utils.j2d_processing(
            joints2d_l, center, scale, augm_dict, args.img_res
        )

        return joints2d_r, joints2d_l
    
    def __len__(self):
        if self.args.debug:
            return 10
        return len(self.subsampled_keys)

    def __getitem__(self, idx):
        seqName, index = self.samples[idx]
        return self.getitem(seqName, index)

    def getitem(self, seqName, index):
        args = self.args
        imgname = self.get_imgname_from_index(seqName, index)
        data = self.hand_labels[imgname]
        
        # these are used for 2d joints processing, full image size is used for intrinsics processing
        height, width = data['crop_size']
        image_size = {"width": width, "height": height}

        bbox = [image_size['width'] / 2, image_size['height'] / 2, max(image_size['width'], image_size['height']) / 200] # original bbox
        is_egocam = True

        j3d_data = data['j3d']
        joints3d_r, joints3d_l, joints3d_valid_r, joints3d_valid_l = self.get_joints3d_data(j3d_data.copy())

        j2d_data = data['j2d'] # these joints correspond to 256x256 image
        joints2d_r, joints2d_l, joints_valid_r, joints_valid_l = self.get_joints2d_data(j2d_data.copy())
        
        cv_img = data['img']
        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )

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

        targets["mano.j2d.norm.r"] = torch.from_numpy(joints2d_r[:, :2]).float()
        targets["mano.j2d.norm.l"] = torch.from_numpy(joints2d_l[:, :2]).float()
        targets["mano.j3d.full.r"] = torch.FloatTensor(joints3d_r[:, :3])
        targets["mano.j3d.full.l"] = torch.FloatTensor(joints3d_l[:, :3])

        # dummy values below
        targets["mano.pose.r"] = torch.zeros((48,)) # dummy values
        targets["mano.pose.l"] = torch.zeros((48,)) # dummy values
        targets["mano.beta.r"] = torch.tensor(self.mean_beta_r)
        targets["mano.beta.l"] = torch.tensor(self.mean_beta_l)

        meta_info["imgname"] = imgname

        # scale and center in the original image space
        scale_original = max(image_size["width"], image_size["height"]) / 200.0
        center_original = [image_size["width"] / 2.0, image_size["height"] / 2.0]
        # new intrinsics for center crop, no scaling
        intrx = data['crop_intrx'].copy()
        intrx[0,0] = data['intrx'][0,0]
        intrx[1,1] = data['intrx'][1,1]
        fixed_focal_length = args.focal_length * (args.img_res / max(image_size["width"], image_size["height"]))
        intrx = data_utils.get_aug_intrix(
            intrx,
            fixed_focal_length,
            args.img_res,
            True,
            center_original[0],
            center_original[1],
            augm_dict["sc"] * scale_original,
        )

        if is_egocam and self.egocam_k is None:
            self.egocam_k = intrx
        elif is_egocam and self.egocam_k is not None:
            intrx = self.egocam_k
        else:
            intrx = intrx.numpy()
        if not isinstance(intrx, np.ndarray):
            intrx = intrx.numpy()

        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        meta_info["query_names"] = '' # dummy value
        meta_info['dataset'] = 'egoexo'
        meta_info["window_size"] = torch.LongTensor(np.array([args.window_size]))
        meta_info["dist"] = torch.FloatTensor(torch.zeros(8)) # dummy value
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["rot_angle"] = np.float32(augm_dict["rot"])

        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info['is_j2d_loss'] = 0
        meta_info['is_j3d_loss'] = 1
        meta_info['is_beta_loss'] = 0
        meta_info['is_pose_loss'] = 0
        meta_info['is_cam_loss'] = 0
        meta_info['is_grasp_loss'] = 0

        is_valid = 1
        left_valid = (joints_valid_l.sum()>3) # atleast 3 joints are visible
        right_valid = (joints_valid_r.sum()>3) # atleast 3 joints are visible
        targets["is_valid"] = float(is_valid)
        targets["left_valid"] = float(left_valid) * targets['is_valid']
        targets["right_valid"] = float(right_valid) * targets['is_valid']
        targets["joints_valid_r"] = joints_valid_r * targets['right_valid']
        targets["joints_valid_l"] = joints_valid_l * targets['left_valid']
        targets['joints3d_valid_r'] = joints3d_valid_r * targets['right_valid']
        targets['joints3d_valid_l'] = joints3d_valid_l * targets['left_valid']

        return inputs, targets, meta_info