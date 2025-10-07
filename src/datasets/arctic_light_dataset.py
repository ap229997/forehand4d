import json
import os
import random
import pickle
import os.path as op

import numpy as np
import torch
from loguru import logger
from torchvision.transforms import Normalize
import pytorch3d.transforms.rotation_conversions as rot_conv

import common.data_utils as data_utils
import src.datasets.dataset_utils as dataset_utils
from common.data_utils import read_img
from common.object_tensors import ObjectTensors
from src.datasets.dataset_utils import get_valid, pad_jts2d
from src.datasets.base_dataset import BaseDataset


class ArcticLightDataset(BaseDataset):
    def __getitem__(self, index):
        imgname = self.imgnames[index]
        data = self.getitem(imgname)
        return data

    def getitem(self, imgname, load_rgb=True):
        args = self.args
        # LOADING START
        speedup = args.speedup
        sid, seq_name, view_idx, image_idx = imgname.split("/")[-4:]
        obj_name = seq_name.split("_")[0]
        view_idx = int(view_idx)

        seq_data = self.data[f"{sid}/{seq_name}"]

        data_cam = seq_data["cam_coord"]
        data_2d = seq_data["2d"]
        data_bbox = seq_data["bbox"]
        data_params = seq_data["params"]

        vidx = int(image_idx.split(".")[0]) - self.ioi_offset[sid]
        vidx, is_valid, right_valid, left_valid = get_valid(
            data_2d, data_cam, vidx, view_idx, imgname
        )

        if view_idx == 0:
            intrx = data_params["K_ego"][vidx].copy()
        else:
            intrx = np.array(self.intris_mat[sid][view_idx - 1])

        # hands
        joints2d_r = pad_jts2d(data_2d["joints.right"][vidx, view_idx].copy())
        joints3d_r = data_cam["joints.right"][vidx, view_idx].copy()

        joints2d_l = pad_jts2d(data_2d["joints.left"][vidx, view_idx].copy())
        joints3d_l = data_cam["joints.left"][vidx, view_idx].copy()

        pose_r = data_params["pose_r"][vidx].copy()
        betas_r = data_params["shape_r"][vidx].copy()
        pose_l = data_params["pose_l"][vidx].copy()
        betas_l = data_params["shape_l"][vidx].copy()

        rot_r = data_cam["rot_r_cam"][vidx, view_idx].copy()
        rot_l = data_cam["rot_l_cam"][vidx, view_idx].copy()

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
        
        if self.setup == 'p2' and cam_rot_aug != 0 and \
            ('img' not in self.args.cond_mode) and ('spatial' not in self.args.cond_mode): # this only works if image is not used in conditioning
            # transform 3d quantities
            joints3d_r, joints3d_l, joints2d_r, joints2d_l, rot_r, rot_l = self.transform_3d_quantities(
                joints3d_r, joints3d_l, rot_r, rot_l, intrx, cam_rot_aug
            )
            joints2d_r = pad_jts2d(joints2d_r)
            joints2d_l = pad_jts2d(joints2d_l)
        ########### end of camera rotation augmentation #########

        # distortion parameters for egocam rendering
        dist = data_params["dist"][vidx].copy()
        # NOTE: kp2d, kp3d are in undistored space
        # thus, results for evaluation is in the undistorted space (non-curved)
        # dist parameters can be used for rendering in visualization

        image_size = self.image_sizes[sid][view_idx]
        image_size = {"width": image_size[0], "height": image_size[1]}

        bbox = data_bbox[vidx, view_idx]  # original bbox
        is_egocam = "/0/" in imgname

        (
            joints2d_r,
            joints2d_l,
            bbox,
        ) = dataset_utils.transform_2d_for_speedup_light(
            speedup,
            is_egocam,
            joints2d_r,
            joints2d_l,
            bbox,
            args.ego_image_scale,
        )

        if load_rgb:
            if imgname.startswith("./"):
                imgname = imgname.replace("./arctic_data", f"{os.environ['DOWNLOADS_DIR']}/data/arctic/")
            if speedup:
                imgname = imgname.replace("/images/", "/cropped_images/")
            cv_img, img_status = read_img(imgname, (2800, 2000, 3))
        else:
            norm_img = None

        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        use_gt_k = is_egocam

        joints2d_r = data_utils.j2d_processing(
            joints2d_r, center, scale, augm_dict, args.img_res
        )
        joints2d_l = data_utils.j2d_processing(
            joints2d_l, center, scale, augm_dict, args.img_res
        )
        
        # data augmentation: image
        if load_rgb:
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
        meta_info["imgname"] = imgname
        
        pose_r = np.concatenate((rot_r, pose_r), axis=0)
        pose_l = np.concatenate((rot_l, pose_l), axis=0)

        # hands
        targets["mano.pose.r"] = torch.from_numpy(
            data_utils.pose_processing(pose_r, augm_dict)
        ).float()
        targets["mano.pose.l"] = torch.from_numpy(
            data_utils.pose_processing(pose_l, augm_dict)
        ).float()
        targets["mano.beta.r"] = torch.from_numpy(betas_r).float()
        targets["mano.beta.l"] = torch.from_numpy(betas_l).float()
        targets["mano.j2d.norm.r"] = torch.from_numpy(joints2d_r[:, :2]).float()
        targets["mano.j2d.norm.l"] = torch.from_numpy(joints2d_l[:, :2]).float()

        # full image camera coord
        targets["mano.j3d.full.r"] = torch.FloatTensor(joints3d_r[:, :3])
        targets["mano.j3d.full.l"] = torch.FloatTensor(joints3d_l[:, :3])

        meta_info["query_names"] = obj_name
        meta_info["window_size"] = torch.LongTensor(np.array([args.window_size]))

        # scale and center in the original image space
        scale_original = max([image_size["width"], image_size["height"]]) / 200.0
        center_original = [image_size["width"] / 2.0, image_size["height"] / 2.0]
        intrx = data_utils.get_aug_intrix(
            intrx,
            args.focal_length,
            args.img_res,
            use_gt_k,
            center_original[0],
            center_original[1],
            augm_dict["sc"] * scale_original,
        )

        if cam_rot_aug != 0:
            # cache intrx to be used later in get_future_data()
            self.intrx = intrx

        if is_egocam and self.egocam_k is None:
            self.egocam_k = intrx
        elif is_egocam and self.egocam_k is not None:
            intrx = self.egocam_k

        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        if not is_egocam:
            dist = dist * float("nan")
        meta_info["dist"] = torch.FloatTensor(dist)
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info["rot_angle"] = np.float32(augm_dict["rot"])
        meta_info['loss_mask'] = 1
        if self.setup == 'p2':
            meta_info['dataset'] = 'arctic_ego' # this info is present in imgname as well
        else:
            meta_info['dataset'] = 'arctic_exo'

        meta_info['is_j2d_loss'] = args.get('finetune_2d', 0) # 'future_j2d' in self.args.get('cond_mode','no_cond') # this shouldn't matter since arctic has 3d labels already
        meta_info['is_j3d_loss'] = 1
        meta_info['is_beta_loss'] = 1
        meta_info['is_pose_loss'] = 1
        meta_info['is_cam_loss'] = 1

        # root and at least 3 joints inside image
        targets["is_valid"] = float(is_valid)
        targets["left_valid"] = float(left_valid) * float(is_valid)
        targets["right_valid"] = float(right_valid) * float(is_valid)
        targets["joints_valid_r"] = np.ones(21) * targets["right_valid"]
        targets["joints_valid_l"] = np.ones(21) * targets["left_valid"]

        return inputs, targets, meta_info
    
    def _process_imgnames(self, seq, split):
        imgnames = self.imgnames
        if seq is not None:
            imgnames = [imgname for imgname in imgnames if "/" + seq + "/" in imgname]
        assert len(imgnames) == len(set(imgnames))
        imgnames = dataset_utils.downsample(imgnames, split)
        self.imgnames = imgnames

    def _load_data(self, args, split, seq):
        self.args = args
        self.split = split
        self.aug_data = split.endswith("train")
        
        self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)

        if "train" in split:
            self.mode = "train"
        elif "val" in split:
            self.mode = "val"
        elif "test" in split:
            self.mode = "test"

        short_split = split.replace("mini", "").replace("tiny", "").replace("small", "")
        data_p = op.join(
            f"{os.environ['DOWNLOADS_DIR']}/data/arctic/data/splits/{self.setup}_{short_split}.npy"
        )
        logger.info(f"Loading {data_p}")
        data = np.load(data_p, allow_pickle=True).item()

        self.data = data["data_dict"]
        self.imgnames = data["imgnames"]

        with open(f"{os.environ['DOWNLOADS_DIR']}/data/arctic/data/meta/misc.json", "r") as f:
            misc = json.load(f)

        self._load_motion_data()

        self.use_arctic_pretrained_feats = False # TODO: config also has a key of the same name, fix this issue
        if self.use_arctic_pretrained_feats:
            self._load_pretrained_feats()

        # unpack
        subjects = list(misc.keys())
        intris_mat = {}
        world2cam = {}
        image_sizes = {}
        ioi_offset = {}
        for subject in subjects:
            world2cam[subject] = misc[subject]["world2cam"]
            intris_mat[subject] = misc[subject]["intris_mat"]
            image_sizes[subject] = misc[subject]["image_size"]
            ioi_offset[subject] = misc[subject]["ioi_offset"]

        self.world2cam = world2cam
        self.intris_mat = intris_mat
        self.image_sizes = image_sizes
        self.ioi_offset = ioi_offset

        object_tensors = ObjectTensors()
        self.kp3d_cano = object_tensors.obj_tensors["kp_bottom"]
        self.obj_names = object_tensors.obj_tensors["names"]
        self.egocam_k = None

    def _load_motion_data(self):
        self.img_dir = f'{os.environ["DOWNLOADS_DIR"]}/data/arctic/data/cropped_images'
        # setup = p2 for ego, p1 for exo
        if self.args.get('use_fixed_length', False):
            data_file = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/fixed_{self.args.max_motion_length:03d}/arctic/{self.setup}_{self.mode}.pkl'
        else:
            data_file = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/arctic/{self.setup}_{self.mode}.pkl'
        with open(data_file, 'rb') as f:
            self.motion_data = pickle.load(f) # dict of lists

    def _load_pretrained_feats(self):   
        if 'mdm' in self.args.method or 'motion' in self.args.method:
            data_p = f"{os.environ['DOWNLOADS_DIR']}/data/arctic/data/feat/{self.args.img_feat_version}/{self.setup}_{self.mode}.pt"
        else:
            data_p = f"{os.environ['DOWNLOADS_DIR']}/data/arctic/data/feat/{self.args.img_feat_version}/{self.setup}_{self.split}.pt"
        logger.info(f"Loading: {data_p}")
        data = torch.load(data_p)
        imgnames = data["imgnames"]
        vecs_list = data["feat_vec"]
        assert len(imgnames) == len(vecs_list)
        vec_dict = {}
        for imgname, vec in zip(imgnames, vecs_list):
            key = "/".join(imgname.split("/")[-4:])
            vec_dict[key] = vec
        self.vec_dict = vec_dict
        assert len(imgnames) == len(vec_dict.keys())
        
        self.imgnames = list(self.vec_dict.keys())

    def __init__(self, args, split, setup='p2', seq=None):
        self.setup = setup
        self._load_data(args, split, seq)
        self._process_imgnames(seq, split)
        logger.info(
            f"ImageDataset Loaded {self.split} split, num samples {len(self.imgnames)}"
        )

    def __len__(self):
        if self.args.debug:
            return 2
        return len(self.imgnames)
    
    def is_index_valid(self, index, num_frames):
        # skip first and last frames as they are not useful
        if index < self.skip_horizon:
            return False
        if index >= num_frames - self.skip_horizon - 1:
            return False
        # discard if there's not enough frames in future to form a window
        if index >= num_frames - self.prediction_horizon:
            return False
        # discard if there's not enough frames in past to form a window
        if index < self.history_size:
            return False
        
        return True
    
    def get_fixed_length_sequence(self, imgname, history_size, prediction_horizon):
        if imgname.startswith("./"):
            imgname = imgname.replace("./arctic_data", f"{os.environ['DOWNLOADS_DIR']}/data/arctic/")

        sid = imgname.split("/")[-4]
        img_idx = int(op.basename(imgname).split(".")[0])
        img_idx = img_idx - self.ioi_offset[sid]

        num_frames = self.data["/".join(imgname.split("/")[-4:-2])]["params"]["rot_r"].shape[0]
        # check if img_idx is within the range of num_frames
        while not self.is_index_valid(img_idx, num_frames):
            img_idx = random.choice(range(len(self.imgnames)))
            imgname = self.imgnames[img_idx]

        future_ind = (np.arange(prediction_horizon) + img_idx + 1).astype(np.int64)
        
        past_ind = (np.arange(history_size) - history_size + img_idx + 1).astype(np.int64)
        return past_ind, future_ind
    
    def get_variable_length_sequence(self, imgname, history_size, curr_length, max_length):
        if imgname.startswith("./"):
            imgname = imgname.replace("./arctic_data", f"{os.environ['DOWNLOADS_DIR']}/data/arctic/")

        sid = imgname.split("/")[-4]
        img_idx = int(op.basename(imgname).split(".")[0])
        img_idx = img_idx - self.ioi_offset[sid]

        future_ind = (np.arange(curr_length) + img_idx + 1).astype(np.int64)
        mask_ind = np.ones(curr_length)
        if curr_length < max_length:
            # repeat the last frame to make it max_length
            future_ind = np.concatenate([future_ind, np.repeat(future_ind[-1:], max_length - curr_length)])
            # add zero mask to indices
            mask_ind = np.concatenate([mask_ind, np.zeros(max_length - curr_length)])

        past_ind = (np.arange(history_size) - history_size + img_idx + 1).astype(np.int64)

        return past_ind, future_ind, mask_ind
    
    def get_imgname_from_index(self, video_name, index):
        if self.setup == "p2":
            imgname = op.join(self.img_dir, video_name, '0', "%05d.jpg" % index)
        else:
            imgname = op.join(self.img_dir, video_name, "%05d.jpg" % index)
        assert op.exists(imgname), f"Image {imgname} does not exist"
        return imgname

    def get_img_data(self, imgname, load_rgb=True):
        if self.use_arctic_pretrained_feats:
            img_folder = f"{os.environ['DOWNLOADS_DIR']}/arctic/data/images/"
            inputs, targets, meta_info = self.getitem(
                op.join(img_folder, imgname), load_rgb=load_rgb
            )
        else:
            inputs, targets, meta_info = self.getitem(imgname, load_rgb=load_rgb)

        return inputs, targets, meta_info
    
    def get_exocentric_joints(self, seq_data, vidx, view_idx):
        data_bbox = seq_data["bbox"]
        data_2d = seq_data["2d"]
        bbox = data_bbox[vidx, view_idx]
        joints2d_r = pad_jts2d(data_2d["joints.right"][vidx, view_idx].copy())
        joints2d_l = pad_jts2d(data_2d["joints.left"][vidx, view_idx].copy())
        (joints2d_r, joints2d_l, bbox) = dataset_utils.transform_2d_for_speedup_light(
            True, False, joints2d_r, joints2d_l, bbox, self.args.ego_image_scale)
        
        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        if hasattr(self, 'augm_dict'):
            augm_dict = self.augm_dict
        else:
            # augment parameters
            augm_dict = data_utils.augm_params(
                self.aug_data,
                self.args.flip_prob,
                self.args.noise_factor,
                self.args.rot_factor,
                self.args.scale_factor,
            )
            augm_dict['rot'] = 0

        joints2d_r = data_utils.j2d_processing(joints2d_r, center, scale, augm_dict, self.args.img_res)
        joints2d_l = data_utils.j2d_processing(joints2d_l, center, scale, augm_dict, self.args.img_res)

        return joints2d_r, joints2d_l
    
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
    
    def get_future_data(self, imgname, ind):
        """
        Get future data for the given image name and future indices.
        """
        sid, seq_name, view_idx, _ = imgname.split("/")[-4:]
        view_idx = int(view_idx)
        vidx = int(imgname.split("/")[-1].split(".")[0])

        seq_data = self.data[f"{sid}/{seq_name}"]

        data_cam = seq_data["cam_coord"]
        data_2d = seq_data["2d"]
        data_params = seq_data["params"]
        
        # check validity for each ind
        right_valid, left_valid = [], []
        joints2d_r, joints2d_l = [], []
        for j, future_ind in enumerate(ind):
            if j == 0 or ind[j] != ind[j-1]:
                _, c_valid, r_valid, l_valid = dataset_utils.get_valid(
                    data_2d, data_cam, future_ind, view_idx, imgname
                ) # imgname is not used in get_valid

                # also get these to be consistent with datasets with 2d labels only for future timesteps
                j2d_r, j2d_l = self.process_joints2d(data_2d, seq_data["bbox"], sid, view_idx, future_ind)
            else:
                # end of unique indices, repeat the last one for the rest
                pass
            right_valid.append(c_valid * r_valid)
            left_valid.append(c_valid * l_valid)
            joints2d_r.append(j2d_r)
            joints2d_l.append(j2d_l)
        
        right_valid = torch.tensor(right_valid).float()
        left_valid = torch.tensor(left_valid).float()
        # right_valid and left_valid are binary labels, extend them to 21
        # this is done to be consistent with other datasets
        right_valid = right_valid.unsqueeze(1).repeat(1, 21)
        left_valid = left_valid.unsqueeze(1).repeat(1, 21)
        joints2d_r = torch.tensor(np.stack(joints2d_r, axis=0)).float()
        joints2d_l = torch.tensor(np.stack(joints2d_l, axis=0)).float()

        # joints
        joints3d_r = data_cam["joints.right"][ind, view_idx].copy()
        joints3d_l = data_cam["joints.left"][ind, view_idx].copy()

        if view_idx == 0: # egocentric
            world2view = data_params["world2ego"][vidx][None] # 1 x 4 x 4
            world2future = data_params["world2ego"][ind] # B x 4 x 4
            # compute transformation from future to view using numpy
            future2view = world2view @ np.linalg.inv(world2future)
        else:
            # exo camera is fixed so repeat the same matrix
            future2view = np.eye(4).astype(np.float32)[None].astype(np.float32)
            future2view = np.repeat(future2view, len(ind), axis=0)
        
        # mano params
        pose_r = torch.from_numpy(data_params["pose_r"][ind].copy())
        betas_r = torch.from_numpy(data_params["shape_r"][ind].copy())
        pose_l = torch.from_numpy(data_params["pose_l"][ind].copy())
        betas_l = torch.from_numpy(data_params["shape_l"][ind].copy())

        # add global orientation to pose
        rot_r = data_cam["rot_r_cam"][ind, view_idx]
        rot_l = data_cam["rot_l_cam"][ind, view_idx]
        
        ######## this is added for camera rotation augmentation ########
        # modify 3D quantities as per the augmentation parameters
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
        
        if self.setup == 'p2' and cam_rot_aug != 0 and ('img' not in self.args.cond_mode) and ('spatial' not in self.args.cond_mode): # this only works if image is not used in conditioning
            # transform 3d quantities
            assert hasattr(self, 'intrx'), "intrinsics need to be set before calling this method"
            joints3d_r, joints3d_l, joints2d_r, joints2d_l, rot_r, rot_l = self.transform_3d_quantities(
                joints3d_r, joints3d_l, rot_r, rot_l, self.intrx, cam_rot_aug
            )
            # normalize 2D joints
            joints2d_r = 2 * joints2d_r / self.args.img_res - 1.0
            joints2d_l = 2 * joints2d_l / self.args.img_res - 1.0
        ######## camera rotation augmentation ends here ########
        
        pose_r = np.concatenate((rot_r, pose_r), axis=-1)
        pose_l = np.concatenate((rot_l, pose_l), axis=-1)

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
        
        if self.setup == 'p1': # arctic_exo:
            future_joints2d_r, future_joints2d_l = [], []
            for future_ind in ind:
                joints2d_r, joints2d_l = self.get_exocentric_joints(seq_data, future_ind, view_idx)
                future_joints2d_r.append(joints2d_r)
                future_joints2d_l.append(joints2d_l)
            future_joints2d_r = np.stack(future_joints2d_r, axis=0)
            future_joints2d_l = np.stack(future_joints2d_l, axis=0)
            future_data["future.j2d.norm.r"] = future_joints2d_r[..., :2]
            future_data["future.j2d.norm.l"] = future_joints2d_l[..., :2]
        
        return future_data
    
    def process_joints2d(self, data_2d, data_bbox, sid, view_idx, vidx):

        joints2d_r = pad_jts2d(data_2d["joints.right"][vidx, view_idx].copy())
        joints2d_l = pad_jts2d(data_2d["joints.left"][vidx, view_idx].copy())

        image_size = self.image_sizes[sid][view_idx]
        image_size = {"width": image_size[0], "height": image_size[1]}

        bbox = data_bbox[vidx, view_idx].copy()  # original bbox
        is_egocam = view_idx == 0

        args = self.args
        (
            joints2d_r,
            joints2d_l,
            bbox,
        ) = dataset_utils.transform_2d_for_speedup_light(
            args.speedup,
            is_egocam,
            joints2d_r,
            joints2d_l,
            bbox,
            args.ego_image_scale,
        )

        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        if hasattr(self, 'augm_dict'):
            augm_dict = self.augm_dict
        else:
            # augment parameters
            augm_dict = data_utils.augm_params(
                self.aug_data,
                self.args.flip_prob,
                self.args.noise_factor,
                self.args.rot_factor,
                self.args.scale_factor,
            )
            augm_dict['rot'] = 0

        joints2d_r = data_utils.j2d_processing(
            joints2d_r, center, scale, augm_dict, args.img_res
        )
        joints2d_l = data_utils.j2d_processing(
            joints2d_l, center, scale, augm_dict, args.img_res
        )

        return joints2d_r[..., :2], joints2d_l[..., :2]