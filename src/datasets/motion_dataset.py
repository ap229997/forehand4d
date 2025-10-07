import os.path as op
import random

import numpy as np
from tqdm import tqdm
import torch
from loguru import logger
from torch.utils.data import Dataset

import common.ld_utils as ld_utils
import common.torch_utils as torch_utils


class FixedLengthMotion(Dataset):
    """
    Dataset class for handling fixed length sequences
    """
    def _load_data(self, args, split):
        # self.use_arctic_pretrained_feats = False # load precomputed image features on arctic dataset
        # if self.use_arctic_pretrained_feats:
        #     if 'mdm' in args.method or 'motion' in args.method:
        #         data_p = f"{os.environ["DOWNLOADS_DIR"]}/data/arctic/data/feat/{args.img_feat_version}/{args.setup}_{self.mode}.pt"
        #     else:
        #         data_p = f"{os.environ["DOWNLOADS_DIR"]}/data/arctic/data/feat/{args.img_feat_version}/{args.setup}_{split}.pt"
        #     logger.info(f"Loading: {data_p}")
        #     data = torch.load(data_p)
        #     imgnames = data["imgnames"]
        #     vecs_list = data["feat_vec"]
        #     assert len(imgnames) == len(vecs_list)
        #     vec_dict = {}
        #     for imgname, vec in zip(imgnames, vecs_list):
        #         key = "/".join(imgname.split("/")[-4:])
        #         vec_dict[key] = vec
        #     self.vec_dict = vec_dict
        #     assert len(imgnames) == len(vec_dict.keys())
        
        self.prediction_horizon = args.max_motion_length
        self.history_size = args.window_size
        self.skip_horizon = 200
        
        self.aug_data = split.endswith("train")
        self.window_size = args.window_size

    def __init__(self, args, dataset, split, seq=None):
        self.args = args
        self.split = split
        self.dataset = dataset
        if hasattr(self.dataset, 'use_pretrained_feats'):
            self.use_arctic_pretrained_feats = self.dataset.use_pretrained_feats
        else:
            self.use_arctic_pretrained_feats = False

        self._load_data(args, split)

        # if self.use_arctic_pretrained_feats:
        #     self.imgnames = list(self.vec_dict.keys())
        # self.imgnames = dataset_utils.downsample(self.dataset.imgnames, split)

        logger.info(
            f"Motion dataset loaded {self.dataset.split} split, num samples {len(self.dataset.imgnames)}"
        )

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
    
    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        imgname = self.imgnames[index]

        past_ind, future_ind = self.dataset.get_fixed_length_sequence(imgname, self.history_size, self.prediction_horizon)
        imgnames = [op.join(op.dirname(imgname), "%05d.jpg" % (idx)) for idx in past_ind]

        targets_list = []
        meta_list = []
        img_feats = []
        inputs_list = []
        load_rgb = True # TODO: add support for no rgb loading
        for imgname in imgnames:
            # if self.use_arctic_pretrained_feats:
            #     img_folder = f"{os.environ["DOWNLOADS_DIR"]}/data/arctic/data/images/"
            #     inputs, targets, meta_info = self.dataset.getitem(
            #         op.join(img_folder, imgname), load_rgb=load_rgb
            #     )
            # else:
            #     inputs, targets, meta_info = self.dataset.getitem(imgname, load_rgb=load_rgb)

            inputs, targets, meta_info = self.dataset.get_img_data(imgname, load_rgb=load_rgb)
            
            if load_rgb:
                inputs_list.append(inputs)
            
            if self.use_arctic_pretrained_feats:
                img_feats.append(self.vec_dict[imgname].type(torch.FloatTensor))
            
            targets_list.append(targets)
            meta_list.append(meta_info)

        future_data = self.dataset.get_future_data(imgname, future_ind)

        inputs = {}
        if load_rgb:
            inputs_list = ld_utils.stack_dl(
                ld_utils.ld2dl(inputs_list), dim=0, verbose=False, squeeze=False
            )
            inputs.update({"img": inputs_list["img"]})
        
        if self.use_arctic_pretrained_feats:
            img_feats = torch.stack(img_feats, dim=0)
            inputs.update({"img_feat": img_feats})

        targets_list = ld_utils.stack_dl(
            ld_utils.ld2dl(targets_list), dim=0, verbose=False, squeeze=False
        )
        meta_list = ld_utils.stack_dl(ld_utils.ld2dl(meta_list), dim=0, verbose=False, squeeze=False)

        targets_list["is_valid"] = torch.FloatTensor(np.array(targets_list["is_valid"]))
        targets_list["left_valid"] = torch.FloatTensor(
            np.array(targets_list["left_valid"])
        )
        targets_list["right_valid"] = torch.FloatTensor(
            np.array(targets_list["right_valid"])
        )
        targets_list["joints_valid_r"] = torch.FloatTensor(
            np.array(targets_list["joints_valid_r"])
        )
        targets_list["joints_valid_l"] = torch.FloatTensor(
            np.array(targets_list["joints_valid_l"])
        )
        targets_list.update(future_data)

        meta_list["center"] = torch.FloatTensor(np.array(meta_list["center"]))
        meta_list["is_flipped"] = torch.FloatTensor(np.array(meta_list["is_flipped"]))
        meta_list["rot_angle"] = torch.FloatTensor(np.array(meta_list["rot_angle"]))
        
        # convert inputs, targets_list, meta_list to tensors
        inputs = torch_utils.to_tensor(inputs)
        targets_list = torch_utils.to_tensor(targets_list)
        meta_list = torch_utils.to_tensor(meta_list)
        
        return inputs, targets_list, meta_list


class VariableLengthMotion(Dataset):
    """
    Dataset class for handling variable length sequences
    """
    def __init__(self, args, dataset, split, seq=None):
        super().__init__()

        self.args = args
        self.split = split
        self.dataset = dataset
        if hasattr(self.dataset, 'use_arctic_pretrained_feats'): # TODO: only supported for arctic right now
            self.use_arctic_pretrained_feats = self.dataset.use_arctic_pretrained_feats
        else:
            self.use_arctic_pretrained_feats = False

        self.prediction_horizon = args.max_motion_length
        self.history_size = args.window_size
        self.skip_horizon = 200
        
        self.aug_data = split.endswith("train")
        self.window_size = args.window_size

        # motion_data is a dict of lists with 'names', 'ranges', 'tasks', 'texts'
        motion_data = self.dataset.motion_data
        video_names = motion_data['names'] # list of video names
        video_ranges = motion_data['ranges'] # list of tuples (start, end)

        self.pointer = 0
        self.max_motion_length = args.get('max_motion_length', 256)
        min_motion_len = 16 # TODO: move to config
        self.args.unit_length = 4 # TODO: move to config

        self.run_inference = self.args.get('inference', False)
        self.augment_length = self.args.get('augment_length', True) # True only during training
        self.use_fixed_length = self.args.get('use_fixed_length', False) # this is handled separately while creating the splits

        # filter out short or too long sequences
        name_list = []
        length_list = []
        range_list = []
        for idx, name in enumerate(tqdm(video_names)):
            start = video_ranges[idx][0]
            end = video_ranges[idx][1]
            length = end - start + 1
            if not self.use_fixed_length:
                if length < min_motion_len or length >= self.max_motion_length:
                    continue
            else:
                if length < self.max_motion_length: # max_motion_length is the fixed length
                    continue
            name_list.append(name)
            length_list.append(length)
            range_list.append(video_ranges[idx])

        name_list, length_list, range_list = zip(*sorted(zip(name_list, length_list, range_list), key=lambda x: x[1]))
        self.length_arr = np.array(length_list)
        self.name_list = name_list
        self.range_list = range_list

        logger.info(
            f"Motion dataset loaded {self.dataset.split} split, num samples {len(self.length_arr)}"
        )

    def __len__(self):
        if self.args.debug:
            return 10
        if self.run_inference:
            start_num = self.args.get('start_num', 0)
            end_num = self.args.get('end_num', 1)
            start_idx = int(start_num * len(self.length_arr))
            end_idx = int(end_num * len(self.length_arr))
            self.length_arr = self.length_arr[start_idx:end_idx]
            self.name_list = self.name_list[start_idx:end_idx]
            self.range_list = self.range_list[start_idx:end_idx]
            print (f"Running inference on {len(self.length_arr)} sequences from {start_idx} to {end_idx} samples")

        return len(self.length_arr)
    
    def __getitem__(self, item):
        video_name = self.name_list[item]
        m_length = self.length_arr[item]
        start, end = self.range_list[item]

        if self.use_fixed_length:
            fixed_length = self.args.max_motion_length
            if self.augment_length: # used during training, at eval st=0 is used always
                idx = random.randint(0, m_length - fixed_length)
                start = start + idx
        else: # this is for variable length
            if self.augment_length: # modify variations in lengths
                # Crop the motions in to times of 4, and introduce small variations
                if self.args.unit_length < 10:
                    coin2 = np.random.choice(['single', 'single', 'double'])
                else:
                    coin2 = 'single'

                if coin2 == 'double':
                    m_length = (m_length // self.args.unit_length - 1) * self.args.unit_length
                elif coin2 == 'single':
                    m_length = (m_length // self.args.unit_length) * self.args.unit_length
                idx = random.randint(0, self.length_arr[item] - m_length)
                start = start + idx
                end = start + m_length + 1

        imgname = self.dataset.get_imgname_from_index(video_name, start)

        if self.use_fixed_length:
            past_ind, future_ind, mask_ind = self.dataset.get_variable_length_sequence(imgname, self.history_size, self.max_motion_length, self.max_motion_length)
        else:    
            past_ind, future_ind, mask_ind = self.dataset.get_variable_length_sequence(imgname, self.history_size, m_length, self.max_motion_length)
        
        imgnames = [self.dataset.get_imgname_from_index(video_name, idx) for idx in past_ind]

        targets_list = []
        meta_list = []
        img_feats = []
        inputs_list = []
        load_rgb = True
        for imgname in imgnames:
            # if self.use_arctic_pretrained_feats:
            #     img_folder = f"{os.environ["DOWNLOADS_DIR"]}/data/arctic/data/images/"
            #     inputs, targets, meta_info = self.dataset.getitem(
            #         op.join(img_folder, imgname), load_rgb=load_rgb
            #     )
            # else:
            #     inputs, targets, meta_info = self.dataset.getitem(imgname, load_rgb=load_rgb)

            inputs, targets, meta_info = self.dataset.get_img_data(imgname, load_rgb=load_rgb)
            
            if load_rgb:
                inputs_list.append(inputs)
            
            if self.use_arctic_pretrained_feats:
                vec_dict_key = '/'.join(imgname.split("/")[-4:])
                img_feats.append(self.vec_dict[vec_dict_key].type(torch.FloatTensor))
            
            targets_list.append(targets)
            meta_list.append(meta_info)

        future_data = self.dataset.get_future_data(imgname, future_ind)
        if hasattr(self.dataset, 'augm_dict'): # should be reset for each sample
            # remove self.dataset.augm_dict and self.dataset.cam_rot_aug
            del self.dataset.augm_dict
            del self.dataset.cam_rot_aug
        if hasattr(self.dataset, 'intrx'): # this is added only for arctic dataset
            del self.dataset.intrx

        inputs = {}
        if load_rgb:
            inputs_list = ld_utils.stack_dl(
                ld_utils.ld2dl(inputs_list), dim=0, verbose=False, squeeze=False
            )
            inputs.update({"img": inputs_list["img"]})
        
        if self.use_arctic_pretrained_feats:
            img_feats = torch.stack(img_feats, dim=0)
            inputs.update({"img_feat": img_feats})

        if self.args.get('interpolate', False):
            # load last image as goal image
            last_imgname = self.dataset.get_imgname_from_index(video_name, future_ind[-1])
            last_inputs, _, _ = self.dataset.get_img_data(last_imgname, load_rgb=load_rgb)
            inputs.update({"goal_img": last_inputs['img'].unsqueeze(0)}) # TODO: add support for multiple goal images

        targets_list = ld_utils.stack_dl(
            ld_utils.ld2dl(targets_list), dim=0, verbose=False, squeeze=False
        )
        meta_list = ld_utils.stack_dl(ld_utils.ld2dl(meta_list), dim=0, verbose=False, squeeze=False)

        targets_list["is_valid"] = torch.FloatTensor(np.array(targets_list["is_valid"]))
        targets_list["left_valid"] = torch.FloatTensor(
            np.array(targets_list["left_valid"])
        )
        targets_list["right_valid"] = torch.FloatTensor(
            np.array(targets_list["right_valid"])
        )
        targets_list["joints_valid_r"] = torch.FloatTensor(
            np.array(targets_list["joints_valid_r"])
        )
        targets_list["joints_valid_l"] = torch.FloatTensor(
            np.array(targets_list["joints_valid_l"])
        )
        targets_list.update(future_data)

        meta_list['mask_timesteps'] = torch.FloatTensor(mask_ind).bool()
        if self.use_fixed_length:
            meta_list['lengths'] = torch.tensor(self.max_motion_length).int()
        else:
            meta_list['lengths'] = torch.tensor(m_length).int()

        meta_list["center"] = torch.FloatTensor(np.array(meta_list["center"]))
        meta_list["is_flipped"] = torch.FloatTensor(np.array(meta_list["is_flipped"]))
        meta_list["rot_angle"] = torch.FloatTensor(np.array(meta_list["rot_angle"]))

        meta_list["video_name"] = video_name[0] if type(video_name) in [tuple, list] else video_name
        meta_list["future_ids"] = future_ind
        
        # convert inputs, targets_list, meta_list to tensors
        inputs = torch_utils.to_tensor(inputs)
        targets_list = torch_utils.to_tensor(targets_list)
        meta_list = torch_utils.to_tensor(meta_list)

        return inputs, targets_list, meta_list
    

class VariableLengthMotion2D(VariableLengthMotion):
    """
    Dataset class for handling variable length sequences for 2D motion
    """
    def __init__(self, args, dataset, split, seq=None):
        super().__init__(args, dataset, split, seq)

    def __getitem__(self, item):
        video_name = self.name_list[item]
        m_length = self.length_arr[item]
        start, end = self.range_list[item]

        # Crop the motions in to times of 4, and introduce small variations
        if self.args.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.args.unit_length - 1) * self.args.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.args.unit_length) * self.args.unit_length
        idx = random.randint(0, self.length_arr[item] - m_length)
        start = start + idx
        end = start + m_length + 1

        imgname = self.dataset.get_imgname_from_index(video_name, start)

        past_ind, future_ind, mask_ind = self.dataset.get_variable_length_sequence(imgname, self.history_size, m_length, self.max_motion_length)
        
        imgnames = [self.dataset.get_imgname_from_index(video_name, idx) for idx in past_ind]

        targets_list = []
        meta_list = []
        img_feats = []
        inputs_list = []
        load_rgb = True
        for imgname in imgnames:
            # if self.use_arctic_pretrained_feats:
            #     img_folder = f"{os.environ["DOWNLOADS_DIR"]}/data/arctic/data/images/"
            #     inputs, targets, meta_info = self.dataset.getitem(
            #         op.join(img_folder, imgname), load_rgb=load_rgb
            #     )
            # else:
            #     inputs, targets, meta_info = self.dataset.getitem(imgname, load_rgb=load_rgb)

            inputs, targets, meta_info = self.dataset.get_img_data(imgname, load_rgb=load_rgb)
            
            if load_rgb:
                inputs_list.append(inputs)
            
            if self.use_arctic_pretrained_feats:
                vec_dict_key = '/'.join(imgname.split("/")[-4:])
                img_feats.append(self.vec_dict[vec_dict_key].type(torch.FloatTensor))
            
            targets_list.append(targets)
            meta_list.append(meta_info)

        future_data = self.dataset.get_future_data(imgname, future_ind)

        inputs = {}
        if load_rgb:
            inputs_list = ld_utils.stack_dl(
                ld_utils.ld2dl(inputs_list), dim=0, verbose=False, squeeze=False
            )
            inputs.update({"img": inputs_list["img"]})
        
        if self.use_arctic_pretrained_feats:
            img_feats = torch.stack(img_feats, dim=0)
            inputs.update({"img_feat": img_feats})

        targets_list = ld_utils.stack_dl(
            ld_utils.ld2dl(targets_list), dim=0, verbose=False, squeeze=False
        )
        meta_list = ld_utils.stack_dl(ld_utils.ld2dl(meta_list), dim=0, verbose=False, squeeze=False)

        targets_list["is_valid"] = torch.FloatTensor(np.array(targets_list["is_valid"]))
        targets_list["left_valid"] = torch.FloatTensor(
            np.array(targets_list["left_valid"])
        )
        targets_list["right_valid"] = torch.FloatTensor(
            np.array(targets_list["right_valid"])
        )
        targets_list["joints_valid_r"] = torch.FloatTensor(
            np.array(targets_list["joints_valid_r"])
        )
        targets_list["joints_valid_l"] = torch.FloatTensor(
            np.array(targets_list["joints_valid_l"])
        )
        targets_list.update(future_data)

        meta_list['mask_timesteps'] = torch.FloatTensor(mask_ind).bool()
        meta_list['lengths'] = torch.tensor(m_length).int()

        meta_list["center"] = torch.FloatTensor(np.array(meta_list["center"]))
        meta_list["is_flipped"] = torch.FloatTensor(np.array(meta_list["is_flipped"]))
        meta_list["rot_angle"] = torch.FloatTensor(np.array(meta_list["rot_angle"]))
        
        # convert inputs, targets_list, meta_list to tensors
        inputs = torch_utils.to_tensor(inputs)
        targets_list = torch_utils.to_tensor(targets_list)
        meta_list = torch_utils.to_tensor(meta_list)
        
        return inputs, targets_list, meta_list