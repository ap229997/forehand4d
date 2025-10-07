import os
import pickle
import os.path as op

import cv2
from glob import glob
import numpy as np
from tqdm import tqdm
import torch
from torchvision.transforms import Normalize

import common.data_utils as data_utils
from src.datasets.dataset_utils import pad_jts2d
from common.body_models import MODEL_DIR as MANO_DIR
from src.datasets.ho3d_utils import read_RGB_img, read_annotation, project_3D_points
from src.datasets.base_dataset import BaseDataset


class H2O3DDataset(BaseDataset):
    def __init__(self,args,split='train'):
        self.baseDir = f"{os.environ['DOWNLOADS_DIR']}/data/h2o3d"
        if 'train' in split:
            self.split = 'train'
        elif 'val' in split:
            self.split = 'train' # no val split available, set to train for code compatibility
        else:
            raise Exception('split not supported')
        self.args = args

        self.image_dir = os.path.join(self.baseDir, self.split, '{}/rgb/{}.jpg')

        self._load_motion_data()

        self.samples = []
        self.video_info = {}
        # every sequence starts with id 0000, skip that
        for i in range(len(self.motion_data['names'])):
            video_name = self.motion_data['names'][i]
            # start_idx = self.motion_data['ranges'][i][0]
            subsampled_indices = self.motion_data['subsampled_indices'][i][1:]
            start_idx = subsampled_indices[0]
            self.samples.append((video_name, start_idx))
            total_frames = len(subsampled_indices)
            self.video_info[video_name] = {'indices': subsampled_indices, 'total_frames': total_frames}
        self.imgnames = self.samples

        self._load_hand_labels()

        self._load_mano_defaults()

        self.img_h, self.img_w = 480, 640
        self.aug_data = split.endswith("train")
        self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)

    def _generate_motion_data(self):
        h2o3d_dir = f"{os.environ['DOWNLOADS_DIR']}/data/h2o3d"
        train_file = f"{h2o3d_dir}/train.txt"
        test_file = f"{h2o3d_dir}/evaluation.txt"
        local_train_file = f"{h2o3d_dir}/local_train.txt"
        local_val_file = f"{h2o3d_dir}/local_val.txt"

        # load all files
        def load_file(file):
            with open(file) as f:
                lines = f.readlines()
            lines = [line.strip() for line in lines]
            return lines

        train_data = load_file(train_file)
        test_data = load_file(test_file)
        local_train_data = load_file(local_train_file)
        local_val_data = load_file(local_val_file)

        video_dir = os.path.join(h2o3d_dir, 'train/*/rgb')
        video_files = glob(video_dir)
        video_files = sorted(video_files)
        video_lengths = {}
        for video_name in video_files:
            video_name = video_name.split('/')[-2]
            video_length = len(glob(os.path.join(h2o3d_dir, 'train', video_name, 'rgb', '*.jpg')))
            video_lengths[video_name] = video_length

        # hand motion is very slow so we can subsample the videos
        # subsample every 10 frames and store indices
        subsample_factor = 10
        subsampled_indices = {}
        for video_name, video_length in video_lengths.items():
            indices = list(range(0, video_length, subsample_factor))
            # convert to string
            indices = [str(i).zfill(4) for i in indices]
            subsampled_indices[video_name] = indices
            
        data_dict = {'names': [], 'subsampled_indices': [], 'ranges': []}
        for video_name, indices in subsampled_indices.items():
            first_idx = indices[0]
            data_dict['names'].append((video_name, first_idx))
            data_dict['subsampled_indices'].append(indices)
            data_dict['ranges'].append((0, len(indices)-1))

        # get video names from lcoal train and val files
        local_train_videos = set()
        for line in local_train_data:
            video_name = line.split('/')[-3]
            local_train_videos.add(video_name)
        local_val_videos = set()
        for line in local_val_data:
            video_name = line.split('/')[-3]
            local_val_videos.add(video_name)

        # # processing splits
        # def extract_dict(data_dict, video_names):
        #     new_dict = {'names': [], 'subsampled_indices': [], 'ranges': []}
        #     for i in range(len(data_dict['names'])):
        #         video_name = data_dict['names'][i][0]
        #         if video_name in video_names:
        #             new_dict['names'].append(data_dict['names'][i])
        #             new_dict['subsampled_indices'].append(data_dict['subsampled_indices'][i])
        #             new_dict['ranges'].append(data_dict['ranges'][i])
        #     return new_dict

        # local_train_dict = extract_dict(data_dict, local_train_videos)
        # local_val_dict = extract_dict(data_dict, local_val_videos)
        # print (len(local_train_dict['names']), len(local_val_dict['names']))
        # save_dir = f"{os.environ['DOWNLOADS_DIR']}/motion_splits/h2o3d"
        # os.makedirs(save_dir, exist_ok=True)
        # with open(os.path.join(save_dir, 'train.pkl'), 'wb') as f:
        #     pickle.dump(local_train_dict, f)
        # with open(os.path.join(save_dir, 'val.pkl'), 'wb') as f:
        #     pickle.dump(local_val_dict, f)

    def _load_motion_data(self):
        # this is generated using _generate_motion_data above
        if self.args.get('use_fixed_length', False):
            data_file = f"{os.environ['DOWNLOADS_DIR']}/motion_splits/h2o3d/fixed_{self.args.max_motion_length:03d}/h2o3d/{self.split}.pkl"
        else:
            data_file = f"{os.environ['DOWNLOADS_DIR']}/motion_splits/h2o3d/{self.split}.pkl"
        with open(data_file, 'rb') as f:
            self.motion_data = pickle.load(f) # dict of lists
        for ind in range(len(self.motion_data['names'])):
            name = self.motion_data['names'][ind]
            self.motion_data['names'][ind] = name[0]

    def _load_mano_defaults(self):
        # load mano_right and mano_left models and cache mean pose
        mano_right = op.join(MANO_DIR, 'MANO_RIGHT.pkl')
        mano_left = op.join(MANO_DIR, 'MANO_LEFT.pkl')
        if not op.exists(mano_right) or not op.exists(mano_left):
            raise Exception('MANO model missing! Please run setup_mano.py to setup mano folder')
        # load mano models using pickle, encoding latin1
        with open(mano_right, 'rb') as f:
            mano_right = pickle.load(f, encoding='latin1')
        with open(mano_left, 'rb') as f:
            mano_left = pickle.load(f, encoding='latin1')
        mano_right_mean = mano_right['hands_mean']
        mano_left_mean = mano_left['hands_mean']
        # add 0,0,0 for global pose
        self.mano_right_mean = np.concatenate([np.zeros((3)), mano_right_mean], axis=0)
        self.mano_left_mean = np.concatenate([np.zeros((3)), mano_left_mean], axis=0)
    
    def _load_hand_labels(self):
        save_dir = f"{os.environ['DOWNLOADS_DIR']}/motion_splits/h2o3d"
        hand_label_file = os.path.join(save_dir, f'hand_labels_{self.split}.pkl')
        if os.path.exists(hand_label_file):
            with open(hand_label_file, 'rb') as f:
                self.hand_labels = pickle.load(f)
            print ('Loaded hand labels from disk')
        
        else:
            self.hand_labels = {}
            for ind in tqdm(range(len(self.motion_data['names']))):
                seqname = self.motion_data['names'][ind]
                curr_indices = self.motion_data['subsampled_indices'][ind]
                for sample in curr_indices:
                    hand_label = read_annotation(self.baseDir, seqname, sample, self.split)
                    if seqname not in self.hand_labels:
                        self.hand_labels[seqname] = {}
                    if sample not in self.hand_labels[seqname]:
                        self.hand_labels[seqname][sample] = hand_label

            # save self.hand_labels and self.ego_cam_pose to disk in pickle format
            os.makedirs(save_dir, exist_ok=True)
            with open(hand_label_file, 'wb') as f:
                pickle.dump(self.hand_labels, f)

            print ('Saved hand labels to {}'.format(save_dir))

    def __len__(self):
        return len(self.samples)
    
    def get_fixed_length_sequence(self, imgname, history_size, prediction_horizon):
        seqname = imgname.split('/')[-3]
        img_idx = int(imgname.split("/")[-1].split(".")[0])

        future_ind = (np.arange(prediction_horizon) + img_idx + 1).astype(np.int64)
        num_frames = self.video_info[seqname]['total_frames']
        future_ind[future_ind >= num_frames] = num_frames - 1
        
        past_ind = (np.arange(history_size) - history_size + img_idx + 1).astype(np.int64)
        past_ind[past_ind < 0] = 0
        return past_ind, future_ind
    
    def get_variable_length_sequence(self, imgname, history_size, curr_length, max_length):
        splits = imgname.split('/')
        seqname = splits[-3]
        img_idx = imgname.split("/")[-1].split(".")[0]
        img_idx = self.video_info[seqname]['indices'].index(img_idx)

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

        return past_ind, future_ind, mask_ind
    
    def get_imgname_from_index(self, seqname, index):
        if not isinstance(index, str):
            index = self.video_info[seqname]['indices'][index]
        imgname = self.image_dir.format(seqname, index)
        assert op.exists(imgname), f"Image {imgname} does not exist"
        return imgname
    
    def get_img_data(self, imgname, load_rgb=True):
        splits = imgname.split('/')
        seqname = splits[-3]
        index = splits[-1].split('.')[0]
        # index = self.video_info[seqname]['indices'][index]
        inputs, targets, meta_info = self.getitem(seqname, index)
        return inputs, targets, meta_info
    
    def get_future_data(self, imgname, indices):
        """
        Get future data for the given image name and future indices.
        """

        splits = imgname.split("/")
        seqname = splits[-3]

        joints3d_r, joints3d_l = [], []
        joints2d_r, joints2d_l = [], []
        pose_r, pose_l = [], []
        betas_r, betas_l = [], []
        right_valid, left_valid = [], []
        future2view = []
        
        for ind in indices:
            index = self.video_info[seqname]['indices'][ind]
            anno = self.hand_labels[seqname][index]

            r3d = anno['rightHandJoints3D']
            l3d = anno['leftHandJoints3D']
            r3d = self.change_coordinate(r3d)
            l3d = self.change_coordinate(l3d)

            rp = anno['rightHandPose'].copy()
            lp = anno['leftHandPose'].copy()
            rp = rp - self.mano_right_mean
            lp = lp - self.mano_left_mean
            rp_global = self.change_rotation(rp[:3].copy())
            lp_global = self.change_rotation(lp[:3].copy())
            rp[:3] = rp_global
            lp[:3] = lp_global

            r2d, l2d = self.get_joints2d(anno)
            
            joints3d_r.append(r3d)
            joints3d_l.append(l3d)
            joints2d_r.append(r2d)
            joints2d_l.append(l2d)
            pose_r.append(rp)
            pose_l.append(lp)
            betas_r.append(anno['handBeta'])
            betas_l.append(anno['handBeta'])
            
            right_valid.append(anno['jointValidRight'])
            left_valid.append(anno['jointValidLeft'])
            future2view.append(np.eye(4).astype(np.float32)) # exocamera is fixed

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
    
    def get_joints2d(self, anno):
        intrx = anno['camMat'].copy()
        image_size = {"width": self.img_w, "height": self.img_h}
        bbox = [image_size['width'] / 2, image_size['height'] / 2, max(image_size['width'], image_size['height']) / 200] # original bbox
        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        args = self.args
        rightHandKps = project_3D_points(intrx, anno['rightHandJoints3D'], is_OpenGL_coords=True)
        leftHandKps = project_3D_points(intrx, anno['leftHandJoints3D'], is_OpenGL_coords=True)

        # augment parameters
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )
        augm_dict['rot'] = 0

        joints2d_r = pad_jts2d(rightHandKps)
        joints2d_l = pad_jts2d(leftHandKps)

        joints2d_r = data_utils.j2d_processing(
            joints2d_r, center, scale, augm_dict, args.img_res
        )
        joints2d_l = data_utils.j2d_processing(
            joints2d_l, center, scale, augm_dict, args.img_res
        )

        return joints2d_r[..., :2], joints2d_l[..., :2]
    
    def change_coordinate(self, joints3d):
        """
        Change the coordinate system of the given joints3d data.
        """
        coordChangMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        joints3d = joints3d.dot(coordChangMat.T)
        return joints3d
    
    def change_rotation(self, rot_aa):
        coordChangMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        cam_rot = cv2.Rodrigues(coordChangMat)[0].squeeze()
        RAbsMat = cv2.Rodrigues(cam_rot)[0].dot(cv2.Rodrigues(rot_aa)[0])
        rp_global = cv2.Rodrigues(RAbsMat)[0][:, 0]
        return rp_global

    def __getitem__(self, idx):
        seqName, index = self.samples[idx]
        return self.getitem(seqName, index)

    def getitem(self, seqName, idx):
        split = self.split
        cv_img = read_RGB_img(self.baseDir, seqName, idx, split)
        cv_img = cv_img[:,:,::-1]

        anno = self.hand_labels[seqName][idx]
        intrx = anno['camMat'].copy()

        image_size = {"width": cv_img.shape[1], "height": cv_img.shape[0]}
        bbox = [image_size['width'] / 2, image_size['height'] / 2, max(image_size['width'], image_size['height']) / 200] # original bbox
        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        args = self.args
        rightHandKps = project_3D_points(intrx, anno['rightHandJoints3D'], is_OpenGL_coords=True)
        leftHandKps = project_3D_points(intrx, anno['leftHandJoints3D'], is_OpenGL_coords=True)

        # augment parameters
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )
        augm_dict['rot'] = 0

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
        
        meta_info["imgname"] = self.get_imgname_from_index(seqName, idx)
        meta_info["query_names"] = '' # dummy value

        r3d = self.change_coordinate(anno['rightHandJoints3D'].copy())
        l3d = self.change_coordinate(anno['leftHandJoints3D'].copy())

        ########## this is due to different coordinate convention ##########
        rp = anno['rightHandPose'].copy()
        lp = anno['leftHandPose'].copy()
        # add mean pose
        rp  = rp - self.mano_right_mean
        lp  = lp - self.mano_left_mean
        rp_global = self.change_rotation(rp[:3].copy())
        lp_global = self.change_rotation(lp[:3].copy())
        rp[:3] = rp_global
        lp[:3] = lp_global

        targets = {}
        targets['mano.pose.r'] = torch.FloatTensor(rp)
        targets['mano.pose.l'] = torch.FloatTensor(lp)
        targets['mano.beta.r'] = torch.FloatTensor(anno['handBeta'])
        targets['mano.beta.l'] = torch.FloatTensor(anno['handBeta'])
        targets['mano.j2d.norm.r'] = torch.from_numpy(joints2d_r[:, :2]).float() # torch.FloatTensor(rightJoints2d)
        targets['mano.j2d.norm.l'] = torch.from_numpy(joints2d_l[:, :2]).float() # torch.FloatTensor(leftJoints2d)
        targets['mano.j3d.full.r'] = torch.FloatTensor(r3d)
        targets['mano.j3d.full.l'] = torch.FloatTensor(l3d)
        
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

        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        meta_info["dist"] = torch.FloatTensor(torch.zeros(8)) # dummy value
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["rot_angle"] = np.float32(augm_dict["rot"])
        meta_info['loss_mask'] = 1
        meta_info['dataset'] = 'h2o3d'

        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info['is_j2d_loss'] = args.get('finetune_2d', 0) # 'future_j2d' in self.args.get('cond_mode','no_cond') # shouldn't matter since 3D labels are available
        meta_info['is_j3d_loss'] = 1
        meta_info['is_beta_loss'] = 1
        meta_info['is_pose_loss'] = 1
        meta_info['is_cam_loss'] = 1

        left_valid = sum(anno['jointValidLeft'] > 0) > 3 # atleast 3 joints are valid
        right_valid = sum(anno['jointValidRight'] > 0) > 3 # atleast 3 joints are valid
        targets['is_valid'] = (left_valid + right_valid) > 0
        targets['left_valid'] = int(left_valid)
        targets['right_valid'] = int(right_valid)
        targets["joints_valid_r"] = anno['jointValidRight'] # np.ones(21) * targets["right_valid"]
        targets["joints_valid_l"] = anno['jointValidLeft'] # np.ones(21) * targets["left_valid"]

        return inputs, targets, meta_info


if __name__ == '__main__':
    from common.xdict import xdict
    args = xdict()
    args.img_res = 224
    args.img_norm_mean = [0.485, 0.456, 0.406]
    args.img_norm_std = [0.229, 0.224, 0.225]
    dat = H2O3DDataset(args)
    print(dat[0])
    
