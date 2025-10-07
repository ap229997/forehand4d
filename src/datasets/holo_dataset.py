import os
import pickle
import os.path as op

import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from torchvision.transforms import Normalize

import common.data_utils as data_utils
from common.data_utils import read_img
from src.datasets.dataset_utils import pad_jts2d
from src.datasets.holo_utils import get_hand_labels
from src.datasets.base_dataset import BaseDataset


class HoloDataset(BaseDataset):
    def __init__(self, args, split):
        super().__init__()
        if 'train' in split:
            self.split = 'train'
        elif 'val' in split:
            self.split = 'val'
        else:
            raise Exception('split not supported')
        
        self.args = args
        self.base_dir = f'{os.environ["DOWNLOADS_DIR"]}/data/holo'
        self.img_file = op.join(self.base_dir, 'video_pitch_shifted/{}/Export_py/Video/images_jpg/{}.jpg')
        self.label_file = op.join(self.base_dir, 'holo_hands/{}/{}.pkl')
        self.cam_file = op.join(self.base_dir, 'campose/{}_action.pkl')

        self._load_motion_data()

        self.samples = []
        self.video_info = {}
        for i in range(len(self.motion_data['names'])):
            video_name = self.motion_data['names'][i]
            max_frames = self.motion_data['max_frames'][i]
            if max_frames == 0:
                continue
            start = self.motion_data['ranges'][i][0]
            start_idx = str(start).zfill(6)
            self.samples.append((video_name, start_idx))
            if video_name not in self.video_info:
                self.video_info[video_name] = {'total_frames': max_frames}
        self.imgnames = self.samples

        self._load_hand_labels()

        self.img_h, self.img_w = 504, 896
        default_focal = 5000 # taken from hamer
        self.scaled_focal = default_focal / 256 * max(self.img_h, self.img_w)
        self.aug_data = split.endswith("train")
        self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)

        self.openpose_to_mano = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20] # hamer outputs follow openpose ordering
        
        # mean values of beta, computed from val set of arctic, used for datasets without MANO fits
        # can also use default beta values in MANO, either is fine as long as it is consistent across training
        self.mean_beta_r = [0.82747316,  0.13775729, -0.39435294, 0.17889787, -0.73901576, 0.7788163, -0.5702684, 0.4947751, -0.24890041, 1.5943261]
        self.mean_beta_l = [-0.19330633, -0.08867972, -2.5790455, -0.10344583, -0.71684015, -0.28285977, 0.55171007, -0.8403888, -0.8490544, -1.3397144]
        
    def _load_motion_data(self):
        if self.args.get('use_fixed_length', False):
            data_file = glob(f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/holo/fixed_{self.args.max_motion_length:03d}/{self.split}*.pkl')
        else:
            data_file = glob(f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/holo/{self.split}*.pkl')
        data_file = sorted(data_file)[-1]
        if not os.path.exists(data_file):
            print(f"Data file {data_file} does not exist")
            exit(1)
        with open(data_file, 'rb') as f:
            self.motion_data = pickle.load(f) # dict of lists
        # subtract 1 from range, indexing issue
        for i in range(len(self.motion_data['ranges'])):
            st, end = self.motion_data['ranges'][i]
            if not self.args.get('use_fixed_length', False):
                self.motion_data['ranges'][i] = (st, end-1)
    
    def _load_hand_labels(self):
        save_dir = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/holo'
        hand_label_file = glob(os.path.join(save_dir, f'hand_labels_{self.split}*.pkl'))
        hand_label_file = sorted(hand_label_file)[-1]
        self.hand_label_file = hand_label_file
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
            total_len = len(self.motion_data['names'])
            for ind in tqdm(range(total_len)):
                seqname = self.motion_data['names'][ind]
                st, end = self.motion_data['ranges'][ind]
                cam_info = self.cam_file.format(seqname)
                with open(cam_info, 'rb') as f:
                    cam_data = pickle.load(f)
                for img_idx in range(st, end+1):
                    imgname = str(img_idx).zfill(6)
                    label_file = self.label_file.format(seqname, imgname)
                    with open(label_file, 'rb') as f:
                        hand_label = pickle.load(f)
                    relevant_label = get_hand_labels(hand_label)
                    cam2world = cam_data['pose'][img_idx]['cam2world']
                    world2cam = np.linalg.inv(cam2world)
                    if seqname not in self.hand_labels:
                        self.hand_labels[seqname] = {}
                    if seqname not in self.ego_cam_pose:
                        self.ego_cam_pose[seqname] = {}
                    self.hand_labels[seqname][imgname] = relevant_label
                    self.ego_cam_pose[seqname][imgname] = world2cam

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
        splits = imgname.split('/')
        seqname = splits[-5]
        img_idx = int(imgname.split("/")[-1].split(".")[0])

        future_ind = (np.arange(prediction_horizon) + img_idx + 1).astype(np.int64)
        num_frames = self.video_info[seqname]['total_frames']
        future_ind[future_ind >= num_frames] = num_frames - 1
        
        past_ind = (np.arange(history_size) - history_size + img_idx + 1).astype(np.int64)
        past_ind[past_ind < 0] = 0
        return past_ind, future_ind
    
    def get_variable_length_sequence(self, imgname, history_size, curr_length, max_length):
        splits = imgname.split('/')
        seqname = splits[-5]
        img_idx = int(imgname.split("/")[-1].split(".")[0])

        future_ind = (np.arange(curr_length) + img_idx + 1).astype(np.int64)
        mask_ind = np.ones(curr_length)
        if curr_length < max_length:
            # repeat the last frame to make it max_length
            future_ind = np.concatenate([future_ind, np.repeat(future_ind[-1:], max_length - curr_length)])
            # add zero mask to indices
            mask_ind = np.concatenate([mask_ind, np.zeros(max_length - curr_length)])
        num_frames = self.video_info[seqname]['total_frames']
        future_ind[future_ind >= num_frames] = num_frames - 1

        past_ind = (np.arange(history_size) - history_size + img_idx + 1).astype(np.int64)
        past_ind[past_ind < 0] = 0
        past_ind[past_ind < img_idx] = img_idx # TODO: past index not supported, labels not available

        return past_ind, future_ind, mask_ind
    
    def get_imgname_from_index(self, seqname, index):
        if not isinstance(index, str):
            index = str(index).zfill(6)
        imgname = self.img_file.format(seqname, index)
        assert op.exists(imgname), f"Image {imgname} does not exist"
        return imgname

    def get_img_data(self, imgname, load_rgb=True):
        splits = imgname.split('/')
        seqname = splits[-5]
        index = splits[-1].split('.')[0]
        inputs, targets, meta_info = self.getitem(seqname, index)
        return inputs, targets, meta_info
    
    def get_future_data(self, imgname, indices):
        """
        Get future data for the given image name and future indices.
        """

        splits = imgname.split("/")
        seqname = splits[-5]
        curr_idx = imgname.split("/")[-1].split(".")[0]

        joints2d_r, joints2d_l = [], []
        verts2d_r, verts2d_l = [], []
        pose_r, pose_l = [], []
        betas_r, betas_l = [], []
        right_valid, left_valid = [], []
        world2future = []
        
        for j, ind in enumerate(indices):
            if not isinstance(ind, str):
                ind = str(ind).zfill(6)
            if j == 0 or indices[j] != indices[j-1]: 
                # compute lables for unique indices
                # hand_labels = self.get_hand_labels(seqname, ind)
                hand_labels = self.hand_labels[seqname][ind]
                if 'iter' in self.hand_label_file:
                    try:
                        right_joints2d, right_verts2d, r_valid, left_joints2d, left_verts2d, l_valid, \
                            right_pose, right_transl, left_pose, left_transl = self.process_hand_labels(hand_labels)
                    except:
                        # print (f"Error in processing hand labels for {seqname} {ind}")
                        right_joints2d = np.zeros((21, 2))
                        left_joints2d = np.zeros((21, 2))
                        right_verts2d = np.zeros((778, 2))
                        left_verts2d = np.zeros((778, 2))
                        r_valid = 0
                        l_valid = 0
                        pose_r = np.zeros((48,))
                        transl_r = np.zeros((3,))
                        pose_l = np.zeros((48,))
                        transl_l = np.zeros((3,))
                        right_pose = np.zeros((48,))
                        left_pose = np.zeros((48,))
                else:
                    right_joints2d, right_verts2d, r_valid, left_joints2d, left_verts2d, l_valid = self.process_hand_labels(hand_labels)
                    right_pose = np.zeros((48,))
                    left_pose = np.zeros((48,))
            else:
                # end of unique indices, repeat the last one for the rest
                pass
            joints2d_r.append(right_joints2d)
            joints2d_l.append(left_joints2d)
            verts2d_r.append(right_verts2d)
            verts2d_l.append(left_verts2d)
            pose_r.append(right_pose)
            pose_l.append(left_pose)
            right_valid.append(r_valid)
            left_valid.append(l_valid)
            betas_r.append(self.mean_beta_r)
            betas_l.append(self.mean_beta_l)

            # egocam_pose = self.get_egocam_pose(seqname, ind)
            egocam_pose = self.ego_cam_pose[seqname][ind]
            world2future.append(egocam_pose)

        joints2d_r = np.stack(joints2d_r, axis=0).astype(np.float32)
        joints2d_l = np.stack(joints2d_l, axis=0).astype(np.float32)
        verts2d_r = np.stack(verts2d_r, axis=0).astype(np.float32)
        verts2d_l = np.stack(verts2d_l, axis=0).astype(np.float32)
        pose_r = np.stack(pose_r, axis=0).astype(np.float32)
        pose_l = np.stack(pose_l, axis=0).astype(np.float32)
        betas_r = np.stack(betas_r, axis=0).astype(np.float32)
        betas_l = np.stack(betas_l, axis=0).astype(np.float32)
        right_valid = np.stack(right_valid, axis=0).astype(np.float32)
        left_valid = np.stack(left_valid, axis=0).astype(np.float32)
        # right_valid and left_valid are binary labels, extend them to 21
        # this is done to be consistent with other datasets
        right_valid = np.tile(right_valid[:, None], (1, 21)).astype(np.float32)
        left_valid = np.tile(left_valid[:, None], (1, 21)).astype(np.float32)
        world2future = np.stack(world2future, axis=0).astype(np.float32)

        world2view = self.ego_cam_pose[seqname][curr_idx].astype(np.float32)
        future2view = world2view @ np.linalg.inv(world2future)

        # dummy values for joints3d_r, joints3d_l, pose_r, pose_l so that mixed dataloader doesn't break
        joints3d_r = np.zeros(joints2d_r.shape[:2]+(3,)).astype(np.float32)
        joints3d_l = np.zeros(joints2d_l.shape[:2]+(3,)).astype(np.float32)
        
        # store in a dict
        future_data = {
            "future_joints3d_r": joints3d_r,
            "future_joints3d_l": joints3d_l,
            "future_pose_r": pose_r,
            "future_pose_l": pose_l,
            "future.j2d.norm.r": joints2d_r,
            "future.j2d.norm.l": joints2d_l,
            # "future.v2d.norm.r": verts2d_r,
            # "future.v2d.norm.l": verts2d_l,
            "future_betas_r": betas_r,
            "future_betas_l": betas_l,
            "future_valid_r": right_valid,
            "future_valid_l": left_valid,
            "future2view": future2view,
        }
        return future_data
    
    def process_hand_labels(self, hand_labels):
        if 'iter' in self.hand_label_file:
            try:
                right_joints2d, right_verts2d, r_valid, left_joints2d, left_verts2d, l_valid, pose_r, transl_r, pose_l, transl_l = hand_labels
            except:
                right_joints2d = np.zeros((21, 2))
                left_joints2d = np.zeros((21, 2))
                right_verts2d = np.zeros((778, 2))
                left_verts2d = np.zeros((778, 2))
                r_valid = 0
                l_valid = 0
                pose_r = np.zeros((48,))
                transl_r = np.zeros((3,))
                pose_l = np.zeros((48,))
                transl_l = np.zeros((3,))
        else:
            right_joints2d, right_verts2d, r_valid, left_joints2d, left_verts2d, l_valid = hand_labels
        right_joints2d = right_joints2d[self.openpose_to_mano]
        left_joints2d = left_joints2d[self.openpose_to_mano]

        args = self.args
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )
        
        image_size = {"width": self.img_w, "height": self.img_h}
        bbox = [image_size['width'] / 2, image_size['height'] / 2, max(image_size['width'], image_size['height']) / 200]
        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        joints2d_r = pad_jts2d(right_joints2d)
        joints2d_l = pad_jts2d(left_joints2d)
        verts2d_r = pad_jts2d(right_verts2d)
        verts2d_l = pad_jts2d(left_verts2d)

        joints2d_r = data_utils.j2d_processing(
            joints2d_r, center, scale, augm_dict, args.img_res
        )
        joints2d_l = data_utils.j2d_processing(
            joints2d_l, center, scale, augm_dict, args.img_res
        )
        verts2d_r = data_utils.j2d_processing(
            verts2d_r, center, scale, augm_dict, args.img_res
        )
        verts2d_l = data_utils.j2d_processing(
            verts2d_l, center, scale, augm_dict, args.img_res
        )

        joints2d_r = joints2d_r[:, :2]
        joints2d_l = joints2d_l[:, :2]
        verts2d_r = verts2d_r[:, :2]
        verts2d_l = verts2d_l[:, :2]

        if 'iter' in self.hand_label_file:
            return joints2d_r, verts2d_r, r_valid, joints2d_l, verts2d_l, l_valid, pose_r, transl_r, pose_l, transl_l
        else:
            return joints2d_r, verts2d_r, r_valid, joints2d_l, verts2d_l, l_valid

    def __getitem__(self, idx):
        seqname, index = self.samples[idx]
        data = self.getitem(seqname, index)
        return data
    
    def getitem(self, seqname, index):
        img_path = self.img_file.format(seqname, index)
        cv_img, _ = read_img(img_path, (2800, 2000, 3))

        # hand_labels = self.get_hand_labels(seqname, index)
        hand_labels = self.hand_labels[seqname][index]
        if 'iter' in self.hand_label_file:
            try:
                right_joints2d, right_verts2d, right_valid, left_joints2d, left_verts2d, left_valid, \
                    right_pose, right_transl, left_pose, left_transl = hand_labels
            except:
                # these are to handle few corrupted labels
                right_joints2d = np.zeros((21, 2))
                left_joints2d = np.zeros((21, 2))
                right_verts2d = np.zeros((778, 2))
                left_verts2d = np.zeros((778, 2))
                right_valid = 0
                left_valid = 0
                pose_r = np.zeros((48,))
                transl_r = np.zeros((3,))
                pose_l = np.zeros((48,))
                transl_l = np.zeros((3,))
        else:
            right_joints2d, right_verts2d, right_valid, left_joints2d, left_verts2d, left_valid = hand_labels
        right_joints2d = right_joints2d[self.openpose_to_mano]
        left_joints2d = left_joints2d[self.openpose_to_mano]

        image_size = {"width": cv_img.shape[1], "height": cv_img.shape[0]}
        bbox = [image_size['width'] / 2, image_size['height'] / 2, max(image_size['width'], image_size['height']) / 200] # original bbox
        center = [bbox[0], bbox[1]]
        scale = bbox[2]

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
        
        joints2d_r = pad_jts2d(right_joints2d)
        joints2d_l = pad_jts2d(left_joints2d)
        verts2d_r = pad_jts2d(right_verts2d)
        verts2d_l = pad_jts2d(left_verts2d)

        joints2d_r = data_utils.j2d_processing(
            joints2d_r, center, scale, augm_dict, args.img_res
        )
        joints2d_l = data_utils.j2d_processing(
            joints2d_l, center, scale, augm_dict, args.img_res
        )
        verts2d_r = data_utils.j2d_processing(
            verts2d_r, center, scale, augm_dict, args.img_res
        )
        verts2d_l = data_utils.j2d_processing(
            verts2d_l, center, scale, augm_dict, args.img_res
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
        
        meta_info["imgname"] = img_path

        targets = {}
        targets["mano.pose.r"] = torch.zeros((48,)) # dummy values
        targets["mano.pose.l"] = torch.zeros((48,)) # dummy values
        targets["mano.beta.r"] = torch.tensor(self.mean_beta_r)
        targets["mano.beta.l"] = torch.tensor(self.mean_beta_l)
        targets['mano.j2d.norm.r'] = torch.from_numpy(joints2d_r[:, :2]).float()
        targets['mano.j2d.norm.l'] = torch.from_numpy(joints2d_l[:, :2]).float()
        targets["mano.j3d.full.r"] = torch.zeros(joints2d_r[:, :3].shape) # dummy values
        targets["mano.j3d.full.l"] = torch.zeros(joints2d_l[:, :3].shape) # dummy values
        
        meta_info["query_names"] = '' # dummy value
        meta_info["window_size"] = torch.LongTensor(np.array([args.window_size]))

        # scale and center in the original image space
        scale_original = max([image_size["width"], image_size["height"]]) / 200.0
        center_original = [image_size["width"] / 2.0, image_size["height"] / 2.0]
        fixed_focal_length = args.focal_length
        intrx = np.array([[self.scaled_focal, 0, self.img_w / 2],
                                 [0, self.scaled_focal, self.img_h / 2],
                                 [0, 0, 1]], dtype=np.float32).astype(np.float32)
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
        meta_info["dist"] = torch.FloatTensor(torch.zeros(8)) # dummy value for distortion params
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["rot_angle"] = np.float32(augm_dict["rot"])
        meta_info['loss_mask'] = 1
        meta_info['dataset'] = 'holo'
        
        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info['is_j2d_loss'] = self.args.get('finetune_2d', 0) # this is set in the config file for optimizing labels
        meta_info['is_j3d_loss'] = 0
        meta_info['is_beta_loss'] = 0
        meta_info['is_pose_loss'] = 0
        meta_info['is_cam_loss'] = 0
        
        targets['is_valid'] = (left_valid + right_valid) > 0
        targets['left_valid'] = left_valid
        targets['right_valid'] = right_valid
        targets["joints_valid_r"] = np.ones(21) * targets["right_valid"]
        targets["joints_valid_l"] = np.ones(21) * targets["left_valid"]

        return inputs, targets, meta_info