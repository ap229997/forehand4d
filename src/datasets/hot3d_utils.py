import os
import os.path as op
import random
import argparse
import pickle

import numpy as np
from tqdm import tqdm
import torch
import pytorch3d.transforms as rot_conv

from common.torch_utils import reset_all_seeds
from common.body_models import build_mano_aa
from src.datasets.dexycb_utils import MANO_DIR


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# taken from https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/clip_util.py
def get_number_of_frames(tar) -> int:
    """Returns the number of frames in a clip.

    Args:
        tar: File handler of an open tar file with clip data.
    Returns:
        Number of frames in the given tar file.
    """

    max_frame_id = -1
    for x in tar.getnames():
        if x.endswith(".info.json"):
            frame_id = int(x.split(".info.json")[0])
            if frame_id > max_frame_id:
                max_frame_id = frame_id
    return max_frame_id + 1


HOT3D_DIR = f"{os.environ['DOWNLOADS_DIR']}/data/hot3d"

def load_mano_defaults(num_comps=45):
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
    
    hands_components_r = mano_right['hands_components'][:num_comps]
    hands_components_l = mano_left['hands_components'][:num_comps]
    
    # mano_right_mean = mano_right['hands_mean']
    # mano_left_mean = mano_left['hands_mean']
    # # add 0,0,0 for global pose
    # mano_right_mean = np.concatenate([np.zeros((3)), mano_right_mean], axis=0)
    # mano_left_mean = np.concatenate([np.zeros((3)), mano_left_mean], axis=0)

    return hands_components_r, hands_components_l

HAND_COMP_R, HAND_COMP_L = load_mano_defaults(num_comps=15) # matches the default in the original HOT3D codebase
MANO_R = build_mano_aa(True)
MANO_L = build_mano_aa(False)


def get_camera_intrx(camera_info, camera_name, img_type):
    camera = camera_info[camera_name][img_type]
    focal = camera['f']
    principal = camera['c']
    return focal, principal


def transform_pose(pose, transl, transform):
    transf = torch.eye(4)
    global_pose = torch.from_numpy(pose[:3].copy()).float()
    transf[:3, :3] = rot_conv.axis_angle_to_matrix(global_pose[None])[0].float()
    transf[:3, 3] = torch.from_numpy(transl.copy())[0].float()
    # apply world2cam_transf to the right hand
    tf = torch.from_numpy(transform.copy()).float()
    transf = tf @ transf
    global_pose = rot_conv.matrix_to_axis_angle(transf[:3, :3][None])[0].float().numpy()
    pose = np.concatenate([global_pose, pose[3:]], axis=0)
    transl = transf[:3, 3]
    return pose, transl


def get_mano_output(pose, beta, transl, mano):
    rp = torch.from_numpy(pose)[None]
    rb = torch.from_numpy(beta)[None]
    rt = torch.from_numpy(transl)[None]
    mano_output = mano(hand_pose=rp[:,3:], global_orient=rp[:,:3], betas=rb, transl=rt)
    joints = mano_output.joints.squeeze(0).numpy()
    vertices = mano_output.vertices.squeeze(0).numpy()
    return joints, vertices


def get_mano_params(labels, hand_comp):
    local_pose = labels['theta']
    local_pose = torch.from_numpy(local_pose).float()[None]
    hand_comp = torch.from_numpy(hand_comp).float()
    local_pose = torch.einsum('bi,ij->bj', [local_pose, hand_comp])
    local_pose = local_pose.squeeze(0).numpy()
    global_pose = labels['wrist_xform'][:3]
    pose = np.concatenate([global_pose, local_pose], axis=0)
    transl = labels['wrist_xform'][3:]
    beta = labels['beta']
    return pose, beta, transl


def get_hand_labels(hand_labels):
    """
    hand_labels = { 'left': 
                {"theta": hand_pose.mano.mano_theta, # this is 15-dim, uses num_pca_comps=15
                "beta": hand_shape.mano_beta,
                "wrist_xform": hand_pose.mano.wrist_xform,}
                'right': 
                {"theta": hand_pose.mano.mano_theta,
                "beta": hand_shape.mano_beta,
                "wrist_xform": hand_pose.mano.wrist_xform,}
                }
    """

    right_pose, right_beta, right_transl = get_mano_params(hand_labels['right'], HAND_COMP_R)
    left_pose, left_beta, left_transl = get_mano_params(hand_labels['left'], HAND_COMP_L)

    right_joints, right_vertices = get_mano_output(right_pose, right_beta, right_transl, MANO_R)
    left_joints, left_vertices = get_mano_output(left_pose, left_beta, left_transl, MANO_L)

    return right_pose, right_beta, right_transl, right_joints, right_vertices, \
              left_pose, left_beta, left_transl, left_joints, left_vertices


def generate_motion_data():
    # process hot3d videos again, but this with the new structure
    data_dict = {'names': [], 'ranges': []}
    video_dir = HOT3D_DIR
    video_names = sorted(os.listdir(video_dir))
    print (len(video_names))
    for vname in tqdm(video_names):
        curr_dir = os.path.join(video_dir, vname)
        meta_file = os.path.join(curr_dir, 'meta.pkl')
        if os.path.exists(meta_file):
            views = sorted(os.listdir(curr_dir))
            for view in views:
                img_dir = os.path.join(curr_dir, view)
                if not os.path.isdir(img_dir):
                    continue
                distorted = os.path.join(curr_dir, view, 'distorted')
                undistorted = os.path.join(curr_dir, view, 'undistorted')
                curr_name = os.path.join(vname, view)
                start_idx = str(0).zfill(6)
                # if os.path.exists(distorted):
                #     data_dict['names'].append((f'{curr_name}/distorted' , start_idx))
                #     data_dict['ranges'].append((0, len(os.listdir(distorted))-1)) # this should be 150
                if os.path.exists(undistorted):
                    data_dict['names'].append((f'{curr_name}/undistorted', start_idx))
                    data_dict['ranges'].append((0, len(os.listdir(undistorted))-1))
    print (len(data_dict['names']), len(data_dict['ranges']))

    # create 80-20 train-val split
    train_dict = {'names': [], 'ranges': []}
    val_dict = {'names': [], 'ranges': []}
    for i in range(len(data_dict['names'])):
        random_val = random.random()
        if random_val < 0.8:
            train_dict['names'].append(data_dict['names'][i])
            train_dict['ranges'].append(data_dict['ranges'][i])
        else:
            val_dict['names'].append(data_dict['names'][i])
            val_dict['ranges'].append(data_dict['ranges'][i])
    print (len(train_dict['names']), len(val_dict['names']))
    # save the new dicts
    save_dir = f"{os.environ['DOWNLOADS_DIR']}/motion_splits/hot3d"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train_dict, f)
    with open(os.path.join(save_dir, 'val.pkl'), 'wb') as f:
        pickle.dump(val_dict, f)


def generate_hand_labels(args):
    data_dir = HOT3D_DIR
    img_dir = os.path.join(data_dir, '{}/{}/{}/{}.jpg')
    label_dir = os.path.join(data_dir, '{}/meta.pkl')

    split = args.split
    data_file = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/hot3d/{split}.pkl'
    with open(data_file, 'rb') as f:
        motion_data = pickle.load(f)

    save_dir = f"{os.environ['DOWNLOADS_DIR']}/motion_splits/hot3d"
    hand_label_file = os.path.join(save_dir, f'hand_labels_{split}.pkl')
    
    hand_labels = {}
    for ind in tqdm(range(len(motion_data['names']))):
        start_idx, end_idx = motion_data['ranges'][ind]
        name = motion_data['names'][ind]
        seqname = name[0]
        label_file = label_dir.format(seqname.split('/')[0])
        try:
            with open(label_file, 'rb') as f:
                label_data = pickle.load(f)
        except:
            print ('Error loading label file: {}'.format(label_file))
            continue
        if seqname not in hand_labels:
            hand_labels[seqname] = {}

        for frame_idx in range(start_idx, end_idx + 1):
            frame_name = str(frame_idx).zfill(6)
            curr_labels = label_data[frame_name]['hands']
            curr_hand = get_hand_labels(curr_labels)
            hand_labels[seqname][frame_name] = curr_hand

    # save self.hand_labels to disk in pickle format
    os.makedirs(save_dir, exist_ok=True)
    with open(hand_label_file, 'wb') as f:
        pickle.dump(hand_labels, f)

    print ('Saved hand labels to {}'.format(save_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    reset_all_seeds(args.seed)

    generate_hand_labels(args)