import os
import pickle

import numpy as np
from tqdm import tqdm
import torch
from torchvision.transforms import Normalize

import common.data_utils as data_utils
from common.data_utils import read_img
from src.datasets.dataset_utils import pad_jts2d
from src.datasets.hot3d_utils import HOT3D_DIR, get_hand_labels, get_camera_intrx, transform_pose
from src.datasets.base_dataset import BaseDataset


class HOT3DDataset(BaseDataset):
    def __init__(self, args, split='train'):
        super().__init__()
        self.args = args
        if 'train' in split:
            self.split = 'train'
        elif 'val' in split:
            self.split = 'val'
        else:
            self.split = 'test'

        self._data_dir = HOT3D_DIR
        self._img_dir = os.path.join(self._data_dir, '{}/{}/{}/{}.jpg')
        self._label_dir = os.path.join(self._data_dir, '{}/meta.pkl')

        self._load_motion_data()

        self._load_hand_labels()

        self.max_video_frames = {}
        for i in range(len(self.motion_data['names'])):
            video_name = self.motion_data['names'][i][0]
            start_idx, end_idx = self.motion_data['ranges'][i]
            if video_name in self.max_video_frames:
                continue
            if video_name not in self.hand_labels:
                continue
            total_frames = end_idx - start_idx + 1
            self.max_video_frames[video_name] = total_frames
        self.imgnames = self.samples = self.motion_data['names']

        self.rot_cam = args.get('rot_hot3d_cam', False)
        self.aug_data = split.endswith("train")
        self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)

    def _load_motion_data(self):
        if self.args.get('use_fixed_length', False):
            data_file = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/hot3d/{self.split}.pkl'
        else:
            data_file = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/hot3d/{self.split}.pkl'
        with open(data_file, 'rb') as f:
            self.motion_data = pickle.load(f)

    def _load_hand_labels(self):
        save_dir = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/hot3d'
        hand_label_file = os.path.join(save_dir, f'hand_labels_{self.split}.pkl')
        if os.path.exists(hand_label_file):
            with open(hand_label_file, 'rb') as f:
                self.hand_labels = pickle.load(f)
            print ('Loaded hand labels from disk')
        
        else:
            self.hand_labels = {}
            for ind in tqdm(range(len(self.motion_data['names']))):
                start_idx, end_idx = self.motion_data['ranges'][ind]
                name = self.motion_data['names'][ind]
                seqname = name[0]
                label_file = self._label_dir.format(seqname.split('/')[0])
                with open(label_file, 'rb') as f:
                    label_data = pickle.load(f)
                if seqname not in self.hand_labels:
                    self.hand_labels[seqname] = {}

                for frame_idx in range(start_idx, end_idx + 1):
                    frame_name = str(frame_idx).zfill(6)
                    curr_labels = label_data[frame_name]['hands']
                    hand_labels = get_hand_labels(curr_labels)
                    self.hand_labels[seqname][frame_idx] = hand_labels

            # save self.hand_labels to disk in pickle format
            os.makedirs(save_dir, exist_ok=True)
            with open(hand_label_file, 'wb') as f:
                pickle.dump(self.hand_labels, f)

            print ('Saved hand labels to {}'.format(save_dir))

    def __len__(self):
        return len(self.samples)

    def get_fixed_length_sequence(self, imgname, history_size, prediction_horizon):
        splits = imgname.split('/')
        img_idx = int(splits[-1].split('.')[0])

        future_ind = (np.arange(prediction_horizon) + img_idx + 1).astype(np.int64)
        num_frames = self.max_video_frames[imgname.split('/')[0]]
        future_ind[future_ind >= num_frames] = num_frames - 1
        
        past_ind = (np.arange(history_size) - history_size + img_idx + 1).astype(np.int64)
        past_ind[past_ind < 0] = 0
        return past_ind, future_ind
    
    def get_variable_length_sequence(self, imgname, history_size, curr_length, max_length):
        splits = imgname.split('/')
        seqname = '/'.join(splits[-4:-1])
        img_idx = int(splits[-1].split('.')[0])

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
        if isinstance(seqname, tuple):
            seqname = seqname[0]
        video_name, camera_name, img_type = tuple(seqname.split('/'))
        if not isinstance(index, str):
            index = str(index).zfill(6)
        imgname = self._img_dir.format(video_name, camera_name, img_type, index)
        return imgname

    def get_img_data(self, imgname, load_rgb=True):
        splits = imgname.split('/')
        seqname = '/'.join(splits[-4:-1])
        index = int(splits[-1].split('.')[0])
        inputs, targets, meta_info = self.getitem(seqname, index)
        return inputs, targets, meta_info
    
    def get_future_data(self, imgname, indices):
        """
        Get future data for the given image name and future indices.
        """

        splits = imgname.split("/")
        seqname = '/'.join(splits[-4:-1])
        curr_idx = int(splits[-1].split(".")[0])

        # load labels
        video_name, camera_name, img_type = seqname.split('/')
        label_file = self._label_dir.format(video_name)
        with open(label_file, 'rb') as f:
            label_data = pickle.load(f)

        joints3d_r, joints3d_l = [], []
        joints2d_r, joints2d_l = [], []
        pose_r, pose_l = [], []
        betas_r, betas_l = [], []
        right_valid, left_valid = [], []
        world2future = []

        if hasattr(self, 'augm_dict'):
            augm_dict = self.augm_dict
            cam_rot_aug = self.cam_rot_aug
        else:
            # augment parameters
            augm_dict = data_utils.augm_params(
                self.aug_data,
                self.args.flip_prob,
                self.args.noise_factor,
                self.args.rot_factor,
                self.args.scale_factor,
            )
            augm_dict['rot'] = 0  # this is taken care separately by cam_rot_aug
            cam_rot_aug = augm_dict['rot']
        
        for ind in indices:
            frame_name = str(ind).zfill(6)
            curr_labels = label_data[frame_name]['hands']
            camera_info = label_data[frame_name]['cameras']
            
            # hand_labels = get_hand_labels(curr_labels)
            hand_labels = self.hand_labels[seqname][frame_name]
            right_pose, right_beta, right_transl, right_joints3d, right_vertices, \
              left_pose, left_beta, left_transl, left_joints3d, left_vertices = hand_labels
            
            cam2world_transf = camera_info[camera_name][img_type]['T_world_from_eye']
            world2cam_transf = np.linalg.inv(cam2world_transf)
            if self.rot_cam:
                world2cam_transf = self.rotate_transf_matrix(world2cam_transf, theta_deg=-90) # rotate the camera space by 90 degrees clockwise

            if cam_rot_aug != 0 and ('img' not in self.args.cond_mode) and ('spatial' not in self.args.cond_mode): # this only works if image is not used in conditioning
                world2cam_transf = self.rotate_transf_matrix(world2cam_transf, theta_deg=cam_rot_aug)
            
            # convert joints to camera coordinates
            right_joints3d = np.einsum('ij, kj->ki', world2cam_transf[:3, :3], right_joints3d) + world2cam_transf[:3, 3]
            left_joints3d = np.einsum('ij, kj->ki', world2cam_transf[:3, :3], left_joints3d) + world2cam_transf[:3, 3]

            # transform global pose of right hand to camera coordinates
            right_pose, right_transl = transform_pose(right_pose, right_transl, world2cam_transf)
            # transform global pose of left hand to camera coordinates
            left_pose, left_transl = transform_pose(left_pose, left_transl, world2cam_transf)

            right_joints2d, left_joints2d = self.get_joints2d(camera_info, right_joints3d, left_joints3d, camera_name, img_type)

            r_valid = self.check_valid_joints2d(right_joints2d[..., :2])
            l_valid = self.check_valid_joints2d(left_joints2d[..., :2])
            
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

            world2future.append(world2cam_transf)

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
        world2future = np.stack(world2future, axis=0).astype(np.float32)

        curr_frame_name = str(curr_idx).zfill(6)
        curr_camera_info = label_data[curr_frame_name]['cameras']
        curr_cam2world_transf = curr_camera_info[camera_name][img_type]['T_world_from_eye']
        world2view = np.linalg.inv(curr_cam2world_transf)
        if self.rot_cam:
            # rotate the camera space by 90 degrees clockwise
            world2view = self.rotate_transf_matrix(world2view, theta_deg=-90)
        
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
    
    def check_valid_joints2d(self, joints2d):
        """
        Check if each joint in joints2d is within the valid range [-1, 1].
        
        Args:
            joints2d: Numpy array of shape (21, 2) representing 21 joints with (x, y) coordinates
            
        Returns:
            Numpy array of length 21, where each element is:
                1 if the corresponding joint is within the valid range [-1, 1]
                0 if the corresponding joint is outside the valid range
        """
        # Ensure joints2d is correct shape
        assert joints2d.shape == (21, 2), f"Expected joints2d shape (21, 2), got {joints2d.shape}"
        
        # Check if both x and y coordinates of each joint are within [-1, 1]
        # Create a mask of shape (21,) where each element indicates validity of a joint
        valid_mask = np.logical_and(
            np.all(joints2d >= -1, axis=1),  # Check all coordinates >= -1
            np.all(joints2d <= 1, axis=1)    # Check all coordinates <= 1
        )
        
        # Convert boolean mask to integers (1 for valid, 0 for invalid)
        return valid_mask.astype(np.int32)

    
    def get_joints2d(self, camera_info, right_joints3d, left_joints3d, camera_name, img_type):
        img_w, img_h = camera_info[camera_name][img_type]['width'], camera_info[camera_name][img_type]['height']
        focal, principal = get_camera_intrx(camera_info, camera_name, img_type)
        intrx = np.array([[focal[0], 0, principal[0]], [0, focal[1], principal[1]], [0, 0, 1]])

        if self.rot_cam:
            # change the intrx as well
            intrx = np.array([[focal[1], 0, principal[1]], [0, focal[0], principal[0]], [0, 0, 1]])
        
        # project 3d joints to 2D:
        right_joints2d = (intrx @ right_joints3d.T).T
        right_joints2d = right_joints2d[:, :2] / right_joints2d[:, 2:3]
        left_joints2d = (intrx @ left_joints3d.T).T
        left_joints2d = left_joints2d[:, :2] / left_joints2d[:, 2:3]
        
        image_size = {"width": img_w, "height": img_h}
        bbox = [image_size['width'] / 2, image_size['height'] / 2, max(image_size['width'], image_size['height']) / 200] # original bbox
        # is_egocam = True
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

        joints2d_r = pad_jts2d(right_joints2d)
        joints2d_l = pad_jts2d(left_joints2d)

        joints2d_r = data_utils.j2d_processing(
            joints2d_r, center, scale, augm_dict, args.img_res
        )
        joints2d_l = data_utils.j2d_processing(
            joints2d_l, center, scale, augm_dict, args.img_res
        )

        return joints2d_r[..., :2], joints2d_l[..., :2]
    
    def rotate_transf_matrix(self, transf_mat, theta_deg=-90):
        """
        Rotate a 4x4 transformation matrix by a given degree around the Z-axis.
        """
        # apply the rotation here
        # theta_deg = -90 # makes hot3d images upright
        theta_rad = np.radians(theta_deg)  # Convert degrees to radians
        # Rotation matrix for clockwise rotation around Z-axis
        R = np.array([
            [np.cos(theta_rad),  np.sin(theta_rad), 0],
            [-np.sin(theta_rad), np.cos(theta_rad), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        # create 4x4 transformation matrix for rotation
        R_4x4 = np.eye(4, dtype=np.float32)
        R_4x4[:3, :3] = R
        # apply the rotation to the world2cam transformation
        transf_mat = R_4x4 @ transf_mat
        return transf_mat
    
    def __getitem__(self, idx):
        seqname, index = self.samples[idx]
        data = self.getitem(seqname, index)
        return data

    def getitem(self, seqname, index):
        if not isinstance(index, str):
            index = str(index).zfill(6)
        
        # img_type = 'undistorted' or 'distorted', only undistorted supported for now
        video_name, camera_name, img_type = seqname.split('/')
        label_file = self._label_dir.format(video_name)
        with open(label_file, 'rb') as f:
            label_data = pickle.load(f)

        camera_info = label_data[index]['cameras']
        img_w, img_h = camera_info[camera_name][img_type]['width'], camera_info[camera_name][img_type]['height']
        focal, principal = get_camera_intrx(camera_info, camera_name, img_type)
        intrx = np.array([[focal[0], 0, principal[0]], [0, focal[1], principal[1]], [0, 0, 1]])
        
        args = self.args
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

        # hand_labels = get_hand_labels(label_data[index]['hands'])
        hand_labels = self.hand_labels[seqname][index]
        right_pose, right_beta, right_transl, right_joints3d, right_vertices, \
              left_pose, left_beta, left_transl, left_joints3d, left_vertices = hand_labels

        cam2world_transf = camera_info[camera_name][img_type]['T_world_from_eye']
        world2cam_transf = np.linalg.inv(cam2world_transf)
        if self.rot_cam:
            world2cam_transf = self.rotate_transf_matrix(world2cam_transf, theta_deg=-90) # rotate the camera space by 90 degrees clockwise
        
        if cam_rot_aug != 0 and ('img' not in self.args.cond_mode) and ('spatial' not in self.args.cond_mode): # this only works if image is not used in conditioning
            world2cam_transf = self.rotate_transf_matrix(world2cam_transf, theta_deg=cam_rot_aug)
        
        # convert joints to camera coordinates
        right_joints3d = np.einsum('ij, kj->ki', world2cam_transf[:3, :3], right_joints3d) + world2cam_transf[:3, 3]
        left_joints3d = np.einsum('ij, kj->ki', world2cam_transf[:3, :3], left_joints3d) + world2cam_transf[:3, 3]
        # convert vertices to camera coordinates
        right_vertices = np.einsum('ij, kj->ki', world2cam_transf[:3, :3], right_vertices) + world2cam_transf[:3, 3]
        left_vertices = np.einsum('ij, kj->ki', world2cam_transf[:3, :3], left_vertices) + world2cam_transf[:3, 3]

        # transform global pose of right hand to camera coordinates
        right_pose, right_transl = transform_pose(right_pose, right_transl, world2cam_transf)
        # transform global pose of left hand to camera coordinates
        left_pose, left_transl = transform_pose(left_pose, left_transl, world2cam_transf)

        if self.rot_cam:
            # change the intrx as well, this is because image is rotated by 90 degrees clockwise
            intrx = np.array([[focal[1], 0, principal[1]], [0, focal[0], principal[0]], [0, 0, 1]])

        # project 3d joints to 2D:
        right_joints2d = (intrx @ right_joints3d.T).T
        right_joints2d = right_joints2d[:, :2] / right_joints2d[:, 2:3]
        left_joints2d = (intrx @ left_joints3d.T).T
        left_joints2d = left_joints2d[:, :2] / left_joints2d[:, 2:3]

        # project 3d vertices to 2D:
        right_vertices2d = (intrx @ right_vertices.T).T
        right_vertices2d = right_vertices2d[:, :2] / right_vertices2d[:, 2:3]
        left_vertices2d = (intrx @ left_vertices.T).T
        left_vertices2d = left_vertices2d[:, :2] / left_vertices2d[:, 2:3]
        
        image_size = {"width": img_w, "height": img_h}
        bbox = [image_size['width'] / 2, image_size['height'] / 2, max(image_size['width'], image_size['height']) / 200] # original bbox
        # is_egocam = True
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

        vertices2d_r = pad_jts2d(right_vertices2d)
        vertices2d_l = pad_jts2d(left_vertices2d)
        vertices2d_r = data_utils.j2d_processing(
            vertices2d_r, center, scale, augm_dict, args.img_res
        )
        vertices2d_l = data_utils.j2d_processing(
            vertices2d_l, center, scale, augm_dict, args.img_res
        )

        imgname = self.get_imgname_from_index(seqname, index)
        cv_img, _ = read_img(imgname, (img_w, img_h, 3))
        if self.rot_cam:
            cv_img = np.rot90(cv_img, -1) # rotate the image to match the convention in this codebase

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
        
        meta_info["imgname"] = imgname
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
        meta_info['dataset'] = 'hot3d'
        # meta_info["sample_index"] = index

        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info['is_j2d_loss'] = 0 # args.get('finetune_2d', 0) # shouldn't matter since 3D labels are available
        meta_info['is_j3d_loss'] = 1
        meta_info['is_beta_loss'] = 1
        meta_info['is_pose_loss'] = 1
        meta_info['is_cam_loss'] = 1
        
        # check this once, hands too close to the camera causing issues
        if meta_info['is_j2d_loss'] == 1 and (right_transl[2] < 0.1 or left_transl[2] < 0.1):
            meta_info['is_cam_loss'] = 0

        r_valid = self.check_valid_joints2d(joints2d_r[..., :2])
        l_valid = self.check_valid_joints2d(joints2d_l[..., :2])

        # atleast 3 valid joints are needed for the hand to be valid
        r_v = int(np.sum(r_valid) >= 3)
        l_v = int(np.sum(l_valid) >= 3)
        
        targets['is_valid'] = r_v * l_v
        targets['right_valid'] = r_v
        targets['left_valid'] = l_v
        targets["joints_valid_r"] = r_valid
        targets["joints_valid_l"] = l_valid

        return inputs, targets, meta_info


