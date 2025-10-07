import os
import pickle
import os.path as op

import numpy as np
from tqdm import tqdm
import torch
from torchvision.transforms import Normalize

import common.data_utils as data_utils
from common.data_utils import read_img
from src.datasets.dataset_utils import pad_jts2d
from src.datasets.epic_utils import get_hand_labels
from src.datasets.base_dataset import BaseDataset


class EPICDataset(BaseDataset):
    def __init__(self, args, split):
        super().__init__()
        if 'train' in split:
            self.split = 'train'
        elif 'val' in split:
            self.split = 'val'
        else:
            raise Exception('split not supported')
        
        self.args = args
        self.base_dir = f"{os.environ['DOWNLOADS_DIR']}/data/epic"
        self.img_file = op.join(self.base_dir, '{}/rgb_frames/{}/frame_{}.jpg')
        self.label_file = op.join(self.base_dir, 'epic_hands/{}/frame_{}.pkl')
        # self.cam_file = op.join(self.base_dir, 'campose/{}_action.pkl')

        self._load_motion_data()

        self.samples = []
        self.video_info = {}
        for i in range(len(self.motion_data['names'])):
            video_name = self.motion_data['names'][i]
            subsampled_indices = self.motion_data['subsampled_indices'][i]
            start_idx = 0
            self.samples.append((video_name, start_idx))
            total_frames = len(subsampled_indices)
            self.video_info[video_name] = {'indices': subsampled_indices, 'total_frames': total_frames}
        self.imgnames = self.samples

        self._load_hand_labels()

        self.img_h, self.img_w = 256, 456
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
            data_file = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/epic/fixed_{self.args.max_motion_length:03d}/epic/{self.split}.pkl'
        else:
            data_file = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/epic/{self.split}.pkl'
        with open(data_file, 'rb') as f:
            self.motion_data = pickle.load(f) # dict of lists

    def _load_hand_labels(self):
        save_dir = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/epic'
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
            total_len = len(self.motion_data['names'])
            for ind in tqdm(range(total_len)):
                seqname = self.motion_data['names'][ind]
                st, end = self.motion_data['ranges'][ind]
                indices = self.motion_data['subsampled_indices']
                
                for i in range(st, end+1):
                    img_idx = indices[ind][i]
                    imgname = str(img_idx).zfill(10)
                    curr_label_file = self.label_file.format(seqname, imgname)
                    with open(curr_label_file, 'rb') as f:
                        hand_label = pickle.load(f)
                    relevant_label = get_hand_labels(hand_label)

                    if seqname not in self.hand_labels:
                        self.hand_labels[seqname] = {}
                    if seqname not in self.ego_cam_pose:
                        self.ego_cam_pose[seqname] = {}
                    self.hand_labels[seqname][imgname] = relevant_label

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
        seqname = splits[-2]
        img_idx = int(imgname.split("/")[-1].split("_")[-1].split(".")[0])
        img_idx = self.video_info[seqname]['indices'].index(img_idx)

        future_ind = (np.arange(prediction_horizon) + img_idx).astype(np.int64)
        num_frames = self.video_info[seqname]['total_frames']
        future_ind[future_ind >= num_frames] = num_frames - 1
        
        past_ind = (np.arange(history_size) - history_size + img_idx + 1).astype(np.int64)
        past_ind[past_ind < 0] = 0
        return past_ind, future_ind
    
    def get_variable_length_sequence(self, imgname, history_size, curr_length, max_length):
        splits = imgname.split('/')
        seqname = splits[-2]
        img_idx = int(imgname.split("/")[-1].split("_")[-1].split(".")[0])
        img_idx = self.video_info[seqname]['indices'].index(img_idx)

        future_ind = (np.arange(curr_length) + img_idx).astype(np.int64)
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
            index = self.video_info[seqname]['indices'][index]
            index = str(index).zfill(10)
        pid = seqname.split('_')[0]
        imgname = self.img_file.format(pid, seqname, index)
        assert op.exists(imgname), f"Image {imgname} does not exist"
        return imgname

    def get_img_data(self, imgname, load_rgb=True):
        splits = imgname.split('/')
        seqname = splits[-2]
        index = int(imgname.split("/")[-1].split("_")[-1].split(".")[0])
        index = self.video_info[seqname]['indices'].index(index)
        inputs, targets, meta_info = self.getitem(seqname, index)
        return inputs, targets, meta_info
    
    def get_future_data(self, imgname, indices):
        """
        Get future data for the given image name and future indices.
        """

        splits = imgname.split("/")
        seqname = splits[-2]
        curr_idx = imgname.split("/")[-1].split("_")[-1].split(".")[0]

        joints2d_r, joints2d_l = [], []
        verts2d_r, verts2d_l = [], []
        betas_r, betas_l = [], []
        right_valid, left_valid = [], []
        world2future = []
        
        for j, ind in enumerate(indices):
            if not isinstance(ind, str):
                ind = self.video_info[seqname]['indices'][ind]
                ind = str(ind).zfill(10)
            if j == 0 or indices[j] != indices[j-1]: 
                # compute lables for unique indices
                # hand_labels = self.get_hand_labels(seqname, ind)
                hand_labels = self.hand_labels[seqname][ind]
                right_joints2d, right_verts2d, r_valid, left_joints2d, left_verts2d, l_valid = self.process_hand_labels(hand_labels)
            else:
                # end of unique indices, repeat the last one for the rest
                pass
            joints2d_r.append(right_joints2d)
            joints2d_l.append(left_joints2d)
            verts2d_r.append(right_verts2d)
            verts2d_l.append(left_verts2d)
            right_valid.append(r_valid)
            left_valid.append(l_valid)
            betas_r.append(self.mean_beta_r)
            betas_l.append(self.mean_beta_l)

            egocam_pose = self.ego_cam_pose[seqname][ind]
            world2future.append(egocam_pose)

        joints2d_r = np.stack(joints2d_r, axis=0).astype(np.float32)
        joints2d_l = np.stack(joints2d_l, axis=0).astype(np.float32)
        verts2d_r = np.stack(verts2d_r, axis=0).astype(np.float32)
        verts2d_l = np.stack(verts2d_l, axis=0).astype(np.float32)
        betas_r = np.stack(betas_r, axis=0).astype(np.float32)
        betas_l = np.stack(betas_l, axis=0).astype(np.float32)
        right_valid = np.stack(right_valid, axis=0).astype(np.float32)
        left_valid = np.stack(left_valid, axis=0).astype(np.float32)
        world2future = np.stack(world2future, axis=0).astype(np.float32)

        world2view = np.array(self.ego_cam_pose[seqname][curr_idx]).astype(np.float32)
        future2view = world2view @ np.linalg.inv(world2future)

        # dummy values for joints3d_r, joints3d_l, pose_r, pose_l so that mixed dataloader doesn't break
        joints3d_r = np.zeros(joints2d_r.shape[:2]+(3,)).astype(np.float32)
        joints3d_l = np.zeros(joints2d_l.shape[:2]+(3,)).astype(np.float32)
        pose_r = np.zeros((joints2d_r.shape[0], 48)).astype(np.float32)
        pose_l = np.zeros((joints2d_l.shape[0], 48)).astype(np.float32)

        # store in a dict
        future_data = {
            "future_joints3d_r": joints3d_r,
            "future_joints3d_l": joints3d_l,
            "future_pose_r": pose_r,
            "future_pose_l": pose_l,
            "future.j2d.norm.r": joints2d_r,
            "future.j2d.norm.l": joints2d_l,
            "future_betas_r": betas_r,
            "future_betas_l": betas_l,
            "future_valid_r": right_valid,
            "future_valid_l": left_valid,
            "future2view": future2view,
        }
        return future_data
    
    def process_hand_labels(self, hand_labels):
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

        return joints2d_r, verts2d_r, r_valid, joints2d_l, verts2d_l, l_valid

    def __getitem__(self, idx):
        seqname, index = self.samples[idx]
        data = self.getitem(seqname, index)
        return data
    
    def getitem(self, seqname, index):
        img_path = self.get_imgname_from_index(seqname, index)
        cv_img, _ = read_img(img_path, (2800, 2000, 3))

        # hand_labels = self.get_hand_labels(seqname, index)
        img_idx = self.video_info[seqname]['indices'][index]
        img_idx = str(img_idx).zfill(10)
        hand_labels = self.hand_labels[seqname][img_idx]
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
        meta_info["query_names"] = '' # dummy value
        
        targets = {}
        targets["mano.pose.r"] = torch.zeros((48,)) # dummy values
        targets["mano.pose.l"] = torch.zeros((48,)) # dummy values
        targets['mano.j2d.norm.r'] = torch.from_numpy(joints2d_r[:, :2]).float()
        targets['mano.j2d.norm.l'] = torch.from_numpy(joints2d_l[:, :2]).float()
        targets["mano.beta.r"] = torch.tensor(self.mean_beta_r)
        targets["mano.beta.l"] = torch.tensor(self.mean_beta_l)
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
        meta_info['dataset'] = 'epic'
        
        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info['is_j2d_loss'] = self.args.get('finetune_2d', 0)
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