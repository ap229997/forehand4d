import os
import yaml
import argparse
import pickle
import os.path as op

import numpy as np
import torch
from glob import glob
from tqdm import tqdm

from common.body_models import MODEL_DIR as MANO_DIR


_SUBJECTS = [
    '20200709-subject-01',
    '20200813-subject-02',
    '20200820-subject-03',
    '20200903-subject-04',
    '20200908-subject-05',
    '20200918-subject-06',
    '20200928-subject-07',
    '20201002-subject-08',
    '20201015-subject-09',
    '20201022-subject-10',
]

_SERIALS = [
    '836212060125',
    '839512060362',
    '840412060917',
    '841412060263',
    '932122060857',
    '932122060861',
    '932122061900',
    '932122062010',
]

_YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

_MANO_JOINTS = [
    'wrist',
    'thumb_mcp',
    'thumb_pip',
    'thumb_dip',
    'thumb_tip',
    'index_mcp',
    'index_pip',
    'index_dip',
    'index_tip',
    'middle_mcp',
    'middle_pip',
    'middle_dip',
    'middle_tip',
    'ring_mcp',
    'ring_pip',
    'ring_dip',
    'ring_tip',
    'little_mcp',
    'little_pip',
    'little_dip',
    'little_tip'
]

_MANO_JOINT_CONNECT = [
    [0,  1], [ 1,  2], [ 2,  3], [ 3,  4],
    [0,  5], [ 5,  6], [ 6,  7], [ 7,  8],
    [0,  9], [ 9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]

_BOP_EVAL_SUBSAMPLING_FACTOR = 4

DEXYCB_DIR = f"{os.environ['DOWNLOADS_DIR']}/data/dexycb"

dexycb_to_mano_ordering = np.array([ 0,  5,  6,  7,  9, 10, 11, 17, 18, 19, 13, 14, 15,  1,  2,  3,  4, 8, 12, 16, 20])
# dexycb_to_mano_ordering = np.array([_MANO_JOINTS.index('wrist'),
#                                          _MANO_JOINTS.index('index_mcp'), _MANO_JOINTS.index('index_pip'), _MANO_JOINTS.index('index_dip'),
#                                          _MANO_JOINTS.index('middle_mcp'), _MANO_JOINTS.index('middle_pip'), _MANO_JOINTS.index('middle_dip'),
#                                          _MANO_JOINTS.index('little_mcp'), _MANO_JOINTS.index('little_pip'), _MANO_JOINTS.index('little_dip'),
#                                          _MANO_JOINTS.index('ring_mcp'), _MANO_JOINTS.index('ring_pip'), _MANO_JOINTS.index('ring_dip'),
#                                          _MANO_JOINTS.index('thumb_mcp'), _MANO_JOINTS.index('thumb_pip'), _MANO_JOINTS.index('thumb_dip'),
#                                          _MANO_JOINTS.index('thumb_tip'), _MANO_JOINTS.index('index_tip'), _MANO_JOINTS.index('middle_tip'),
#                                          _MANO_JOINTS.index('ring_tip'), _MANO_JOINTS.index('little_tip')])



class DexYCBMotion():
  """DexYCB dataset."""
  ycb_classes = _YCB_CLASSES
  mano_joints = _MANO_JOINTS
  mano_joint_connect = _MANO_JOINT_CONNECT

  def __init__(self, setup='s3', split='train'):
    """Constructor.

    Args:
      setup: Setup name. 's0', 's1', 's2', or 's3'.
      split: Split name. 'train', 'val', or 'test'.
    """
    self._setup = setup
    self._split = split

    self._data_dir = DEXYCB_DIR
    self._calib_dir = os.path.join(self._data_dir, "calibration")
    self._model_dir = os.path.join(self._data_dir, "models")

    self._color_format = "color_{:06d}.jpg"
    self._depth_format = "aligned_depth_to_color_{:06d}.png"
    self._label_format = "labels_{:06d}.npz"
    self._h = 480
    self._w = 640

    self._obj_file = {
        k: os.path.join(self._model_dir, v, "textured_simple.obj")
        for k, v in _YCB_CLASSES.items()
    }

    # Seen subjects, camera views, grasped objects.
    if self._setup == 's0':
      if self._split == 'train':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i % 5 != 4]
      if self._split == 'val':
        subject_ind = [0, 1]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i % 5 == 4]
      if self._split == 'test':
        subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i % 5 == 4]

    # Unseen subjects.
    if self._setup == 's1':
      if self._split == 'train':
        subject_ind = [0, 1, 2, 3, 4, 5, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = list(range(100))
      if self._split == 'val':
        subject_ind = [6]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = list(range(100))
      if self._split == 'test':
        subject_ind = [7, 8]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = list(range(100))

    # Unseen camera views.
    if self._setup == 's2':
      if self._split == 'train':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5]
        sequence_ind = list(range(100))
      if self._split == 'val':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [6]
        sequence_ind = list(range(100))
      if self._split == 'test':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [7]
        sequence_ind = list(range(100))

    # Unseen grasped objects.
    if self._setup == 's3':
      if self._split == 'train':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [
            i for i in range(100) if i // 5 not in (3, 7, 11, 15, 19)
        ]
      if self._split == 'val':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i // 5 in (3, 19)]
      if self._split == 'test':
        subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
        sequence_ind = [i for i in range(100) if i // 5 in (7, 11, 15)]

    self._subjects = [_SUBJECTS[i] for i in subject_ind]

    self._serials = [_SERIALS[i] for i in serial_ind]
    self._intrinsics = []
    for s in self._serials:
      intr_file = os.path.join(self._calib_dir, "intrinsics",
                               "{}_{}x{}.yml".format(s, self._w, self._h))
      with open(intr_file, 'r') as f:
        intr = yaml.load(f, Loader=yaml.FullLoader)
      intr = intr['color']
      self._intrinsics.append(intr)

    self.data_dict = {'names': [], 'ranges': []}
    
    self._sequences = []
    self._mapping = []
    self._ycb_ids = []
    self._ycb_grasp_ind = []
    self._mano_side = []
    self._mano_betas = []
    offset = 0
    for n in tqdm(self._subjects):
      seq = sorted(os.listdir(os.path.join(self._data_dir, n)))
      seq = [os.path.join(n, s) for s in seq]
      assert len(seq) == 100
      seq = [seq[i] for i in sequence_ind]
      self._sequences += seq
      for i, q in enumerate(seq):
        for serial in self._serials: # these are different cameras
          video_name = os.path.join(q, serial)
          video_path = os.path.join(self._data_dir, video_name)
          images = sorted(glob(os.path.join(video_path, 'color_*.jpg')))
          indices = [int(x.split('_')[-1].split('.')[0]) for x in images]
          self.data_dict['names'].append(video_name)
          self.data_dict['ranges'].append((min(indices), max(indices)))

    print ('Loaded {} sequences from {}'.format(len(self.data_dict['names']), self._data_dir))

    save_dir = f"{os.environ['DOWNLOADS_DIR']}/motion_splits/dexycb"
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    with open(os.path.join(save_dir, f'{setup}_{split}.pkl'), 'wb') as f:
      pickle.dump(self.data_dict, f)

  def __len__(self):
    return len(self.data_dict['names'])


def generate_motion_data():
    _sets_seqs = {}
    for setup in ('s0', 's1', 's2', 's3'):
        for split in ('train', 'val', 'test'):
            name = '{}_{}_seqs'.format(setup, split)
            _sets_seqs[name] = (lambda setup=setup, split=split: DexYCBMotion(setup, split))


def get_dummy_hand_labels():
    pose = np.zeros(48)
    beta = np.zeros(10)
    transl = np.zeros(3)
    joints3d = np.ones((21, 3)) * -1
    joints2d = np.ones((21, 2)) * -1
    valid = False
    return pose, beta, transl, joints3d, joints2d, valid

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

hand_comp_r, hand_comp_l = load_mano_defaults()

def get_hand_labels(meta_file, label_file, use_pca=True):
    # load meta file
    with open(meta_file, 'r') as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)
    hand_type = meta['mano_sides'][0]
    mano_calib_file = os.path.join(DEXYCB_DIR, "calibration",
                                    "mano_{}".format(meta['mano_calib'][0]),
                                    "mano.yml")
    with open(mano_calib_file, 'r') as f:
        mano_calib = yaml.load(f, Loader=yaml.FullLoader)
    mano_betas = mano_calib['betas']
    
    # load label file
    labels = np.load(label_file)
    hand_pose = labels['pose_m'][0]
    mano_thetas = hand_pose[:48]
    mano_transl = hand_pose[-3:]

    joint3d = labels['joint_3d'][0]
    joint3d = joint3d[dexycb_to_mano_ordering]
    joint2d = labels['joint_2d'][0]
    joint2d = joint2d[dexycb_to_mano_ordering]
    joint_valid = ~((joint3d == -1).all())

    if 'right' in hand_type:
        right_pose = mano_thetas
        right_beta = mano_betas
        right_transl = mano_transl
        right_joints3d = joint3d
        right_joints2d = joint2d
        right_valid = joint_valid

        if use_pca:
            global_pose = right_pose[:3].copy()
            local_pose = right_pose[3:48].copy()
            local_pose = torch.from_numpy(local_pose).float()[None]
            hand_comp = torch.from_numpy(hand_comp_r).float()
            local_pose = torch.einsum('bi,ij->bj', [local_pose, hand_comp])
            local_pose = local_pose.squeeze(0).numpy()
            right_pose = np.concatenate([global_pose, local_pose], axis=0)

        # dummy values for left hand
        left_pose, left_beta, left_transl, left_joints3d, left_joints2d, left_valid = get_dummy_hand_labels()

    elif 'left' in hand_type:
        left_pose = mano_thetas
        left_beta = mano_betas
        left_transl = mano_transl
        left_joints3d = joint3d
        left_joints2d = joint2d
        left_valid = joint_valid

        if use_pca:
            global_pose = left_pose[:3].copy()
            local_pose = left_pose[3:48].copy()
            local_pose = torch.from_numpy(local_pose).float()[None]
            hand_comp = torch.from_numpy(hand_comp_l).float()
            local_pose = torch.einsum('bi,ij->bj', [local_pose, hand_comp])
            local_pose = local_pose.squeeze(0).numpy()
            left_pose = np.concatenate([global_pose, local_pose], axis=0)

        # dummy values for right hand
        right_pose, right_beta, right_transl, right_joints3d, right_joints2d, right_valid = get_dummy_hand_labels()

    return right_pose, right_beta, right_transl, right_joints3d, right_joints2d, right_valid, \
            left_pose, left_beta, left_transl, left_joints3d, left_joints2d, left_valid


def main(args):
    data_dir = f"{os.environ['DOWNLOADS_DIR']}/data/dexycb"
    calib_dir = op.join(data_dir, "calibration")
    model_dir = op.join(data_dir, "models")

    color_format = "color_{:06d}.jpg"
    depth_format = "aligned_depth_to_color_{:06d}.png"
    label_format = "labels_{:06d}.npz"
    img_h = 480
    img_w = 640
    
    split = args.split
    data_file = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/dexycb/s3_{split}.pkl'
    with open(data_file, 'rb') as f:
        motion_data = pickle.load(f)

    hand_dict = {}
    for ind in tqdm(range(len(motion_data['names']))):
        seqname = motion_data['names'][ind]
        start_idx, end_idx = motion_data['ranges'][ind]
        for frame_idx in range(start_idx, end_idx + 1):
            label_file = os.path.join(data_dir, seqname, label_format.format(frame_idx))
            meta_file = os.path.join(data_dir, '/'.join(seqname.split('/')[:2]), "meta.yml")
            hand_labels = get_hand_labels(meta_file, label_file)
            if seqname not in hand_dict:
                hand_dict[seqname] = {}
            hand_dict[seqname][frame_idx] = hand_labels

    save_dir = f"{os.environ['DOWNLOADS_DIR']}/motion_splits/dexycb"
    hand_label_file = os.path.join(save_dir, f'hand_labels_s3_{split}.pkl')
    # save self.hand_labels and self.ego_cam_pose to disk in pickle format
    os.makedirs(save_dir, exist_ok=True)
    with open(hand_label_file, 'wb') as f:
        pickle.dump(hand_dict, f)
    print ('Saved hand labels to {}'.format(save_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', help='split to process')
    args = parser.parse_args()
    main(args)