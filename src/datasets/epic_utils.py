import os
import argparse
import pickle
import json
import numpy as np
from tqdm import tqdm
from glob import glob


# These are taken from EPIC-Fields code: https://github.com/epic-kitchens/epic-Fields-code
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def get_w2c(img_data: list) -> np.ndarray:
    """
    Args:
        img_data: list, [qvec, tvec] of w2c
    
    Returns:
        w2c: np.ndarray, 4x4 world-to-camera matrix
    """
    
    w2c = np.eye(4)
    w2c[:3, :3] = qvec2rotmat(img_data[:4])
    w2c[:3, -1] = img_data[4:7]
    return w2c


def get_c2w(img_data: list) -> np.ndarray:
    """
    Args:
        img_data: list, [qvec, tvec] of w2c
    
    Returns:
        c2w: np.ndarray, 4x4 camera-to-world matrix
    """
    w2c = get_w2c(img_data)
    c2w = np.linalg.inv(w2c)
    return c2w


def get_dummy_joints():
    joints = np.zeros((21, 2), dtype=np.float32)
    verts = np.zeros((778, 2), dtype=np.float32)
    return joints, verts


def get_hand_labels(hand_label):
    right_joints2d = hand_label['right']['joints2d']
    left_joints2d = hand_label['left']['joints2d']
    right_verts2d = hand_label['right']['verts2d']
    left_verts2d = hand_label['left']['verts2d']
    if len(right_joints2d) == 0:
        right_joints2d, right_verts2d = get_dummy_joints()
        right_valid = 0
    else:
        right_joints2d = np.array(right_joints2d[0]).astype(np.float32)
        right_verts2d = np.array(right_verts2d[0]).astype(np.float32)
        right_valid = 1
    
    if len(left_joints2d) == 0:
        left_joints2d, left_verts2d = get_dummy_joints()
        left_valid = 0
    else:
        left_joints2d = np.array(left_joints2d[0]).astype(np.float32)
        left_verts2d = np.array(left_verts2d[0]).astype(np.float32)
        left_valid = 1
    
    return right_joints2d, right_verts2d, right_valid, left_joints2d, left_verts2d, left_valid


def get_egocam_pose():
    json_dir = f"{os.environ['DOWNLOADS_DIR']}/data/epic/campose"
    cam_files = sorted(glob(os.path.join(json_dir, '*.json')))
    
    egocam_pose = {}
    for file in tqdm(cam_files):
        seqname = file.split('/')[-1].split('.')[0]
        with open(file, 'r') as f:
            model = json.load(f)
        camera = model['camera']
        cam_h, cam_w = camera['height'], camera['width']
        egocam_pose[seqname] = {}
        for k, v in model['images'].items():
            name = k.split('_')[-1].split('.')[0]
            w2c = get_w2c(v)
            egocam_pose[seqname][name] = w2c.tolist()

    # save campose
    save_dir = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/epic'
    train_file = f'{save_dir}/train.pkl'
    val_file = f'{save_dir}/val.pkl'
    # load the pkl files
    with open(train_file, 'rb') as f:
        train_dict = pickle.load(f)
    with open(val_file, 'rb') as f:
        val_dict = pickle.load(f)
    train_video_names = set(train_dict['names'])
    val_video_names = set(val_dict['names'])

    # set the camera poses in the dicts
    train_camposes = {}
    val_camposes = {}
    for k, v in egocam_pose.items():
        if k in train_video_names:
            train_camposes[k] = v
        if k in val_video_names:
            val_camposes[k] = v

    # save using pickle
    train_name = f'{save_dir}/ego_cam_pose_train.pkl'
    val_name = f'{save_dir}/ego_cam_pose_val.pkl'
    with open(train_name, 'wb') as f:
        pickle.dump(train_camposes, f)
    with open(val_name, 'wb') as f:
        pickle.dump(val_camposes, f)

    print(f'Saved camposes to {save_dir}')


def main(args):
    base_dir = f"{os.environ['DOWNLOADS_DIR']}/data/epic"
    img_file = os.path.join(base_dir, '{}/rgb_frames/{}/frame_{}.jpg')
    label_file = os.path.join(base_dir, 'epic_hands/{}/frame_{}.pkl')
    # cam_file = os.path.join(base_dir, 'campose/{}_action.pkl')

    data_file = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/epic/{args.split}.pkl'
    with open(data_file, 'rb') as f:
        motion_data = pickle.load(f) # dict of lists

    save_dir = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/epic'
    hand_label_file = os.path.join(save_dir, f'hand_labels_{args.split}.pkl')
    cam_pose_file = os.path.join(save_dir, f'ego_cam_pose_{args.split}.pkl')
    hand_labels = {}
    ego_cam_pose = {}
    total_len = len(motion_data['names'])
    for ind in tqdm(range(total_len)):
        seqname = motion_data['names'][ind]
        st, end = motion_data['ranges'][ind]
        indices = motion_data['subsampled_indices']
        
        for i in range(st, end+1):
            img_idx = indices[ind][i]
            imgname = str(img_idx).zfill(10)
            curr_label_file = label_file.format(seqname, imgname)
            with open(curr_label_file, 'rb') as f:
                hand_label = pickle.load(f)
            relevant_label = get_hand_labels(hand_label)

            if seqname not in hand_labels:
                hand_labels[seqname] = {}
            if seqname not in ego_cam_pose:
                ego_cam_pose[seqname] = {}
            hand_labels[seqname][imgname] = relevant_label
            ego_cam_pose[seqname][imgname] = np.eye(4) # dummpy value for now

    # save hand_labels and ego_cam_pose to disk in pickle format
    os.makedirs(save_dir, exist_ok=True)
    with open(hand_label_file, 'wb') as f:
        pickle.dump(hand_labels, f)
    with open(cam_pose_file, 'wb') as f:
        pickle.dump(ego_cam_pose, f)

    print ('Saved hand labels to {}'.format(save_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', help='train or val')
    args = parser.parse_args()

    get_egocam_pose()