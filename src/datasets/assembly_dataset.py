import os
import pickle
import os.path as osp

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from glob import glob

import common.data_utils as data_utils
from src.datasets.dataset_utils import pad_jts2d
from src.datasets.base_dataset import BaseDataset
from src.datasets.assembly_utils import (
    cam2pixel,
    Camera,
    world2cam_assemblyhands as world2cam,
    CAM_ROT,
)
import json
from pycocotools.coco import COCO

ANNOT_VERSION = "v1-1"
N_DEBUG_SAMPLES = 400


class AssemblyDataset(BaseDataset):
    def __init__(self, args, split="train"):
        self.split = split.replace('mini','').replace('tiny','').replace('small','')  # train, test, val
        self.img_path = f"{os.environ['DOWNLOADS_DIR']}/assembly/images"
        self.annot_path = f"{os.environ['DOWNLOADS_DIR']}/assembly/annotations"
        self.modality = "ego"
        self.transform = transforms.Compose([
                transforms.ToTensor()
            ]) if 'train' in self.split else transforms.ToTensor()
        self.normalize_img = transforms.Normalize(mean=args.img_norm_mean, std=args.img_norm_std)
        self.args = args
        self.joint_num = 21  # single hand
        self.root_joint_idx = {"right": 20, "left": 41}
        
        # assembly to mano joint mapping
        self.joint_type = {
            "right": np.array([20, 7, 6, 5, 11, 10, 9, 19, 18, 17, 15, 14, 13, 3, 2, 1, 0, 4, 8, 12, 16]),
            "left": np.array([41, 28, 27, 26, 32, 31, 30, 40, 39, 38, 36, 35, 34, 24, 23, 22, 21, 25, 29, 33, 37]),
        }
        # mano to assembly joint mapping
        self.joint_type_inv = {'right': [], 'left': []}
        hand_ids = self.joint_type['right']
        for i, j in enumerate(hand_ids):
            self.joint_type_inv['right'].append(np.where(hand_ids == i)[0][0])
        self.joint_type_inv['right'] = np.array(self.joint_type_inv['right'])
        self.joint_type_inv['left'] = self.joint_type_inv['right'].copy() # concatenate right and left hand preds

        self.baseDir = f"{os.environ['DOWNLOADS_DIR']}/data/assembly"
        self.image_dir = os.path.join(self.baseDir, 'images/ego_images_rectified/{}/{}/{}.jpg')
        
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
        
        self.is_debug = args.debug
        self.aug_data = split.endswith("train")

        self.img_h, self.img_w = 480, 636

        # mean values of beta, computed from val set of arctic, used for datasets without MANO fits
        # can also use default beta values in MANO, either is fine as long as it is consistent across training
        self.mean_beta_r = [0.82747316,  0.13775729, -0.39435294, 0.17889787, -0.73901576, 0.7788163, -0.5702684, 0.4947751, -0.24890041, 1.5943261]
        self.mean_beta_l = [-0.19330633, -0.08867972, -2.5790455, -0.10344583, -0.71684015, -0.28285977, 0.55171007, -0.8403888, -0.8490544, -1.3397144]

    def handtype_str2array(self, hand_type):
        if hand_type == "right":
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == "left":
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == "interacting":
            return np.array([1, 1], dtype=np.float32)
        else:
            assert 0, print("Not supported hand type: " + hand_type)

    def _load_motion_data(self):
        # this is generated using _generate_motion_data above
        if self.args.get('use_fixed_length', False):
            data_file = glob(f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/assembly/fixed_{self.args.max_motion_length:03d}/assembly/{self.split}*.pkl')
        else:
            data_file = glob(f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/assembly/{self.split}*.pkl')
        data_file = sorted(data_file)[-1]
        if not os.path.exists(data_file):
            print(f"Data file {data_file} does not exist")
            exit(1)
        with open(data_file, 'rb') as f:
            self.motion_data = pickle.load(f) # dict of lists
        for ind in range(len(self.motion_data['names'])):
            st, end = self.motion_data['ranges'][ind]
            assert end > st, f"Invalid range: {st}, {end}"
            if not self.args.get('use_fixed_length', False):
                end = min(end, self.args.max_motion_length-2)
            self.motion_data['ranges'][ind] = (st, end)

    def _load_hand_labels(self):
        save_dir = f"{os.environ['DOWNLOADS_DIR']}/motion_splits/assembly"
        hand_label_file = glob(os.path.join(save_dir, f'hand_labels_{self.split}*.pkl'))
        hand_label_file = sorted(hand_label_file)[-1]
        self.hand_label_file = hand_label_file
        cam_info_file = os.path.join(save_dir, f'cam_info_{self.split}.pkl')
        if os.path.exists(hand_label_file) and os.path.exists(cam_info_file):
            with open(hand_label_file, 'rb') as f:
                self.hand_labels = pickle.load(f)
            with open(cam_info_file, 'rb') as f:
                self.cam_info = pickle.load(f)
            print ('Loaded hand labels from disk')
        
        else:
            # load annotation
            print(f"Load annotation from  {self.annot_path}, mode: {self.split}")
            data_mode = self.split
            db = COCO(
                osp.join(
                    self.annot_path,
                    data_mode,
                    "assemblyhands_"
                    + data_mode
                    + f"_{self.modality}_data_{ANNOT_VERSION}.json",
                )
            )
            with open(
                osp.join(
                    self.annot_path,
                    data_mode,
                    "assemblyhands_"
                    + data_mode
                    + f"_{self.modality}_calib_{ANNOT_VERSION}.json",
                )
            ) as f:
                cameras = json.load(f)["calibration"]
            with open(
                osp.join(
                    self.annot_path,
                    data_mode,
                    "assemblyhands_" + data_mode + f"_joint_3d_{ANNOT_VERSION}.json",
                )
            ) as f:
                joints = json.load(f)["annotations"]

            print("Processing groundtruth annotations")
            self.hand_labels = {}
            self.cam_info = {}
            annot_list = db.anns.keys()
            for i, aid in enumerate(tqdm(annot_list)):
                ann = db.anns[aid]
                image_id = ann["image_id"]
                img = db.loadImgs(image_id)[0]

                seq_name = img['seq_name']
                camera_name = img['camera']
                frame_idx = img["frame_idx"]
                video_name = osp.join(seq_name, camera_name)
                
                K = np.array(
                    cameras[seq_name]["intrinsics"][camera_name + "_mono10bit"],
                    dtype=np.float32,
                )

                Rt = np.array(
                    cameras[seq_name]["extrinsics"][f"{frame_idx:06d}"][
                        camera_name + "_mono10bit"
                    ],
                    dtype=np.float32,
                )
                # convert Rt to 4x4 matrix
                tf_mat = np.concatenate(
                    (Rt, np.array([[0, 0, 0, 1]], dtype=np.float32)), axis=0
                )
                # convert translation from mm to m
                tf_mat[:3, 3] = tf_mat[:3, 3] / 1000.0
                if video_name not in self.cam_info:
                    self.cam_info[video_name] = {}
                self.cam_info[video_name][frame_idx] = tf_mat
                self.cam_info[video_name]['intrx'] = K

                retval_camera = Camera(K, tf_mat[:3], dist=None, name=camera_name)
                campos, camrot, focal, princpt = retval_camera.get_params()

                joint_world = np.array(
                    joints[seq_name][f"{frame_idx:06d}"]["world_coord"], dtype=np.float32
                ) / 1000.0  # mm -> m
                joint_cam = world2cam(joint_world, camrot, campos)
                joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]

                joint_valid = np.array(ann["joint_valid"], dtype=np.float32).reshape(
                    self.joint_num * 2
                )
                # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
                joint_valid[:21] *= joint_valid[self.root_joint_idx['right']]
                joint_valid[21:] *= joint_valid[self.root_joint_idx['left']]

                abs_depth = {
                    "right": joint_cam[self.root_joint_idx["right"], 2],
                    "left": joint_cam[self.root_joint_idx["left"], 2],
                }

                if video_name not in self.hand_labels:
                    self.hand_labels[video_name] = {}
                if frame_idx not in self.hand_labels[video_name]:
                    self.hand_labels[video_name][frame_idx] = {}
                curr_labels = {}
                curr_labels["world_coord"] = joint_world
                curr_labels["cam_coord"] = joint_cam
                curr_labels["img_coord"] = joint_img
                curr_labels["valid"] = joint_valid
                curr_labels["abs_depth"] = abs_depth
                self.hand_labels[video_name][frame_idx] = curr_labels

            # save self.hand_labels and self.cam_info to disk in pickle format
            os.makedirs(save_dir, exist_ok=True)
            with open(hand_label_file, 'wb') as f:
                pickle.dump(self.hand_labels, f)
            with open(cam_info_file, 'wb') as f:
                pickle.dump(self.cam_info, f)

            print ('Saved hand labels to {}'.format(save_dir))
    
    def __len__(self):
        if self.args.debug:
            return 10
        return len(self.subsampled_keys)

    def get_fixed_length_sequence(self, imgname, history_size, prediction_horizon):
        splits = imgname.split('/')
        seqname = '/'.join(splits[-3:-1])
        img_idx = imgname.split("/")[-1].split(".")[0]
        
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
        seqname = '/'.join(splits[-3:-1])
        img_idx = imgname.split("/")[-1].split(".")[0]

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
        imgname = self.image_dir.format(self.split, video_name, index)
        assert os.path.exists(imgname), f"Image {imgname} does not exist"
        return imgname
    
    def get_img_data(self, imgname, load_rgb=True):
        splits = imgname.split('/')
        seqname = '/'.join(splits[-3:-1])
        index = splits[-1].split('.')[0]
        inputs, targets, meta_info = self.getitem(seqname, index)
        return inputs, targets, meta_info
    
    def get_future_data(self, imgname, indices):
        """
        Get future data for the given image name and future indices.
        """

        splits = imgname.split("/")
        seqname = "/".join(splits[-3:-1])
        curr_idx = imgname.split("/")[-1].split(".")[0]

        camname = seqname.split('/')[-1].split('_')[1]
        rot_angle = CAM_ROT.get(camname, 0.0)

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
                ind = range_indices[ind]
            
            if j == 0 or indices[j] != indices[j-1]:
                # hand_labels = self.get_hand_labels(seqname, ind)
                hand_labels = self.hand_labels[seqname.replace('_mono10bit', '')][int(ind)]
                
                joints = hand_labels["cam_coord"]
                right_joints = joints[self.joint_type["right"]].astype(np.float32)
                left_joints = joints[self.joint_type["left"]].astype(np.float32)
                
                if rot_angle != 0.0: # this only works for inplane rotation
                    # rotate joints if necessary
                    right_joints, left_joints = self.apply_rotation_to_joints3d(
                        right_joints, left_joints, rot_angle
                    )
                
                right_j2d, left_j2d = self.process_joints2d(hand_labels["img_coord"], rot=rot_angle)
                
                right_pose = np.zeros((48,))
                left_pose = np.zeros((48,))
                if 'pose_r' in hand_labels:
                    assert 'pose_l' in hand_labels
                    right_pose = hand_labels["pose_r"]
                    left_pose = hand_labels["pose_l"]
                right_betas = self.mean_beta_r
                left_betas = self.mean_beta_l
                r_valid = hand_labels["valid"][self.joint_type["right"]]
                l_valid = hand_labels["valid"][self.joint_type["left"]]
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

            egocam_pose = self.cam_info[seqname.replace('_mono10bit', '')][int(ind)]
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
        world2future = np.stack(world2future, axis=0).astype(np.float32)

        world2view = self.cam_info[seqname.replace('_mono10bit', '')][int(curr_idx)]
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
    
    def process_joints2d(self, joints2d, rot=0.0):
        j2d_r = joints2d[self.joint_type['right']]
        j2d_l = joints2d[self.joint_type['left']]

        # rotate joints2d if necessary
        if rot != 0.0:
            j2d_r, j2d_l = self.apply_rotation_to_joints2d(
                j2d_r, j2d_l, rot, self.img_w, self.img_h
            )

        j2d_r = pad_jts2d(j2d_r)
        j2d_l = pad_jts2d(j2d_l)
        bbox = [self.img_w//2, self.img_h//2, max(self.img_w, self.img_h)/200.0]
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

        j2d_r = data_utils.j2d_processing(
            j2d_r, center, scale, augm_dict, args.img_res
        )
        j2d_l = data_utils.j2d_processing(
            j2d_l, center, scale, augm_dict, args.img_res
        )
        j2d_r = j2d_r[:, :2]
        j2d_l = j2d_l[:, :2]

        return j2d_r, j2d_l
    
    def rotate_image(self, cv_img, rot_angle, img_w, img_h):
        """
        Apply rotation to an image.
        
        Args:
            cv_img: OpenCV image (numpy array)
            rot_angle: Rotation angle in degrees (must be 0, 90, 180, or 270)
            img_w: Image width
            img_h: Image height
            
        Returns:
            cv_img: Rotated image
        """
        # Check if rotation is needed
        if rot_angle == 0.0:
            return cv_img
        
        # Validate rotation angle
        if rot_angle not in [90, 180, 270]:
            raise ValueError("Rotation angle must be 0, 90, 180, or 270 degrees")
        
        # Apply rotation based on angle
        if rot_angle == 90:
            # Rotate image 90 degrees clockwise
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
        elif rot_angle == 180:
            # Rotate image 180 degrees
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_180)
        elif rot_angle == 270:
            # Rotate image 270 degrees clockwise (90 counterclockwise)
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return cv_img
    
    def apply_rotation_to_joints2d(self, joints2d_r, joints2d_l, rot_angle, img_w, img_h):
        """
        Apply rotation to 2D joints and camera intrinsics.
        
        Args:
            joints2d_r: Right joints 2D coordinates (numpy array of shape [N, 2])
            joints2d_l: Left joints 2D coordinates (numpy array of shape [N, 2])
            rot_angle: Rotation angle in degrees (must be 0, 90, 180, or 270)
            img_w: Image width
            img_h: Image height
            
        Returns:
            joints2d_r: Rotated right joints
            joints2d_l: Rotated left joints
        """
        # Check if rotation is needed
        if rot_angle == 0.0:
            return joints2d_r, joints2d_l
        
        # Validate rotation angle
        if rot_angle not in [90, 180, 270]:
            raise ValueError("Rotation angle must be 0, 90, 180, or 270 degrees")
        
        # Get a copy of joints to avoid modifying the original
        joints2d_r = joints2d_r.copy()
        joints2d_l = joints2d_l.copy()
        
        # Apply rotation based on angle
        if rot_angle == 90:
            # Rotate joints2d 90 degrees clockwise
            for i in range(joints2d_r.shape[0]):
                x, y = joints2d_r[i]
                joints2d_r[i] = [img_h - y, x]
            for i in range(joints2d_l.shape[0]):
                x, y = joints2d_l[i]
                joints2d_l[i] = [img_h - y, x]
            
        elif rot_angle == 180:
            # Rotate joints2d 180 degrees
            for i in range(joints2d_r.shape[0]):
                x, y = joints2d_r[i]
                joints2d_r[i] = [img_w - x, img_h - y]
            for i in range(joints2d_l.shape[0]):
                x, y = joints2d_l[i]
                joints2d_l[i] = [img_w - x, img_h - y]
            
        elif rot_angle == 270:
            # Rotate joints2d 270 degrees clockwise
            for i in range(joints2d_r.shape[0]):
                x, y = joints2d_r[i]
                joints2d_r[i] = [y, img_w - x]
            for i in range(joints2d_l.shape[0]):
                x, y = joints2d_l[i]
                joints2d_l[i] = [y, img_w - x]
        
        return joints2d_r, joints2d_l
    
    def apply_rotation_to_intrinsics(self, intrx, rot_angle, img_w, img_h):
        """
        Apply rotation to 2D joints and camera intrinsics.
        
        Args:
            intrx: Camera intrinsics matrix (3x3 numpy array)
            rot_angle: Rotation angle in degrees (must be 0, 90, 180, or 270)
            img_w: Image width
            img_h: Image height
            
        Returns:
            intrx: Updated camera intrinsics
        """
        # Check if rotation is needed
        if rot_angle == 0.0:
            return intrx
        
        # Validate rotation angle
        if rot_angle not in [90, 180, 270]:
            raise ValueError("Rotation angle must be 0, 90, 180, or 270 degrees")
        
        # Get a copy of joints to avoid modifying the original
        intrx = intrx.copy()
        
        # Apply rotation based on angle
        if rot_angle == 90:
            # Update intrinsics for 90-degree rotation
            fx, fy = intrx[0, 0], intrx[1, 1]
            cx, cy = intrx[0, 2], intrx[1, 2]
            intrx[0, 0], intrx[1, 1] = fy, fx
            intrx[0, 2], intrx[1, 2] = img_h - cy, cx
            
        elif rot_angle == 180:
            # Update intrinsics for 180-degree rotation
            cx, cy = intrx[0, 2], intrx[1, 2]
            intrx[0, 2], intrx[1, 2] = img_w - cx, img_h - cy
            
        elif rot_angle == 270:
            # Update intrinsics for 270-degree rotation
            fx, fy = intrx[0, 0], intrx[1, 1]
            cx, cy = intrx[0, 2], intrx[1, 2]
            intrx[0, 0], intrx[1, 1] = fy, fx
            intrx[0, 2], intrx[1, 2] = cy, img_w - cx
        
        return intrx
    
    def apply_rotation_to_joints3d(self, joints3d_r, joints3d_l, rot_angle):
        """
        Apply rotation to 3D joints around the z-axis.
        
        Args:
            joints3d_r: Right joints 3D coordinates (numpy array of shape [N, 3])
            joints3d_l: Left joints 3D coordinates (numpy array of shape [N, 3])
            rot_angle: Rotation angle in degrees (must be 0, 90, 180, or 270)
            
        Returns:
            joints3d_r: Rotated right joints 3D
            joints3d_l: Rotated left joints 3D
        """
        # Check if rotation is needed
        if rot_angle == 0.0:
            return joints3d_r, joints3d_l
        
        # Validate rotation angle
        if rot_angle not in [90, 180, 270]:
            raise ValueError("Rotation angle must be 0, 90, 180, or 270 degrees")
        
        # Create copies to avoid modifying the originals
        joints3d_r = joints3d_r.copy()
        joints3d_l = joints3d_l.copy()
        
        # Convert angle to radians
        theta = np.radians(rot_angle)
        
        # Create rotation matrix for z-axis rotation
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        R = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        
        # Apply rotation to the 3D joints
        # For each joint, apply the rotation matrix
        for i in range(joints3d_r.shape[0]):
            joints3d_r[i] = R @ joints3d_r[i]
        
        for i in range(joints3d_l.shape[0]):
            joints3d_l[i] = R @ joints3d_l[i]
        
        return joints3d_r, joints3d_l
    
    def __getitem__(self, idx):
        seqName, index = self.samples[idx]
        return self.getitem(seqName, index)

    def getitem(self, seqName, index):
        img_path = self.get_imgname_from_index(seqName, index)
        seqName = seqName.replace('_mono10bit', '')
        joint = self.hand_labels[seqName][int(index)]

        joint_world = joint["world_coord"].copy()
        joint_cam = joint["cam_coord"].copy()
        joint_img = joint["img_coord"].copy()
        joint_valid = joint["valid"].copy()

        data_cam = joint_cam
        data_2d = joint_img

        intrx = self.cam_info[seqName]['intrx'].copy()
        joints2d_r = data_2d[self.joint_type['right']]
        joints2d_l = data_2d[self.joint_type['left']]
        joints3d_r = data_cam[self.joint_type['right']] / 1000 # mm -> m
        joints3d_l = data_cam[self.joint_type['left']] / 1000 # mm -> m

        cv_img, img_status = data_utils.read_img(img_path, None)
        # load image
        img_h, img_w, _ = cv_img.shape
        bbox = [img_w//2, img_h//2, max(img_w, img_h)/200.0]
        is_egocam = True
        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        ######### transformations based on in-plane rotation #########
        # rotate these based on camera rotation to make them upright
        camname = seqName.split('/')[-1].split('_')[1]
        rot_angle = CAM_ROT.get(camname, 0.0)

        if rot_angle != 0.0:
            # Rotate image
            cv_img = self.rotate_image(cv_img, rot_angle, img_w, img_h)

            # Rotate joints intrinsics
            joints2d_r, joints2d_l = self.apply_rotation_to_joints2d(
                joints2d_r, joints2d_l, rot_angle, img_w, img_h
            )

            # Rotate intrinsics
            intrx = self.apply_rotation_to_intrinsics(intrx, rot_angle, img_w, img_h)

            # Rotate 3D joints
            joints3d_r, joints3d_l = self.apply_rotation_to_joints3d(
                joints3d_r, joints3d_l, rot_angle
            )
        ########### end of transformations #########

        joints2d_r = pad_jts2d(joints2d_r)
        joints2d_l = pad_jts2d(joints2d_l)

        args = self.args
        # augment parameters
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )

        
        use_gt_k = True
        
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

        # exporting starts
        inputs = {}
        targets = {}
        meta_info = {}
        inputs["img"] = norm_img
        meta_info["imgname"] = img_path

        # hands
        targets["mano.pose.r"] = torch.zeros((48,)) # dummy values
        targets["mano.pose.l"] = torch.zeros((48,)) # dummy values
        targets["mano.beta.r"] = torch.tensor(self.mean_beta_r)
        targets["mano.beta.l"] = torch.tensor(self.mean_beta_l)
        targets["mano.j2d.norm.r"] = torch.from_numpy(joints2d_r[:, :2]).float()
        targets["mano.j2d.norm.l"] = torch.from_numpy(joints2d_l[:, :2]).float()

        targets["mano.j3d.full.r"] = torch.FloatTensor(joints3d_r[:, :3])
        targets["mano.j3d.full.l"] = torch.FloatTensor(joints3d_l[:, :3])

        # dummy values
        meta_info["query_names"] = 'dummy'
        meta_info["window_size"] = torch.LongTensor(np.array([args.window_size]))

        # scale and center in the original image space
        scale_original = max([img_w, img_h]) / 200.0
        center_original = [img_w / 2.0, img_h / 2.0]
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

        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        meta_info["dist"] = torch.FloatTensor(torch.zeros(8)) # dummy value
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["rot_angle"] = np.float32(augm_dict["rot"])
        meta_info['loss_mask'] = 1
        meta_info['dataset'] = 'assembly'

        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info['is_j2d_loss'] = self.args.get('finetune_2d', 0) # og: 1
        meta_info['is_j3d_loss'] = self.args.get('finetune_3d', 0)
        meta_info['is_beta_loss'] = 0
        meta_info['is_pose_loss'] = 0
        meta_info['is_cam_loss'] = 0

        joint_valid_r = joint_valid[self.joint_type["right"]]
        joint_valid_l = joint_valid[self.joint_type["left"]]
        # root and at least 3 joints inside image
        right_valid = sum(joint_valid_r > 0) > 3
        left_valid = sum(joint_valid_l > 0) > 3
        targets["is_valid"] = (left_valid + right_valid) > 0
        targets["left_valid"] = left_valid
        targets["right_valid"] = right_valid
        targets["joints_valid_r"] = joint_valid_r
        targets["joints_valid_l"] = joint_valid_l
        
        return inputs, targets, meta_info

