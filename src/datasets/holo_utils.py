import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm


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


def main(args):
    base_dir = f'{os.environ["DOWNLOADS_DIR"]}/data/holo'
    img_file = os.path.join(base_dir, 'video_pitch_shifted/{}/Export_py/Video/images_jpg/{}.jpg')
    label_file = os.path.join(base_dir, 'holo_hands/{}/{}.pkl')
    cam_file = os.path.join(base_dir, 'campose/{}_action.pkl')

    data_file = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/holo/{args.split}.pkl'
    with open(data_file, 'rb') as f:
        motion_data = pickle.load(f) # dict of lists

    save_dir = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/holo'
    hand_label_file = os.path.join(save_dir, f'hand_labels_{args.split}.pkl')
    cam_pose_file = os.path.join(save_dir, f'ego_cam_pose_{args.split}.pkl')
    hand_labels = {}
    ego_cam_pose = {}
    total_len = len(motion_data['names'])
    for ind in tqdm(range(total_len)):
        seqname = motion_data['names'][ind]
        st, end = motion_data['ranges'][ind]
        cam_info = cam_file.format(seqname)
        with open(cam_info, 'rb') as f:
            cam_data = pickle.load(f)
        for img_idx in range(st, end+1):
            imgname = str(img_idx).zfill(6)
            curr_label_file = label_file.format(seqname, imgname)
            with open(curr_label_file, 'rb') as f:
                hand_label = pickle.load(f)
            relevant_label = get_hand_labels(hand_label)

            cam2world = cam_data['pose'][img_idx]['cam2world']
            world2cam = np.linalg.inv(cam2world)
            if seqname not in hand_labels:
                hand_labels[seqname] = {}
            if seqname not in ego_cam_pose:
                ego_cam_pose[seqname] = {}
            hand_labels[seqname][imgname] = relevant_label
            ego_cam_pose[seqname][imgname] = world2cam

    # save hand_labels and ego_cam_pose to disk in pickle format
    os.makedirs(save_dir, exist_ok=True)
    with open(hand_label_file, 'wb') as f:
        pickle.dump(hand_labels, f)
    with open(cam_pose_file, 'wb') as f:
        pickle.dump(ego_cam_pose, f)

    print ('Saved hand labels to {}'.format(save_dir))


def load_motion_data(args):
    data_file = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/holo/{args.split}.pkl'
    with open(data_file, 'rb') as f:
        motion_data = pickle.load(f) # dict of lists
    return motion_data

def load_hand_labels(args):
    save_dir = args.save_dir
    print ('Loading hand labels from', save_dir)
    hand_label_file = os.path.join(save_dir, f'hand_labels_{args.split}.pkl')
    cam_pose_file = os.path.join(save_dir, f'ego_cam_pose_{args.split}.pkl')
    if os.path.exists(hand_label_file) and os.path.exists(cam_pose_file):
        with open(hand_label_file, 'rb') as f:
            hand_labels = pickle.load(f)
        with open(cam_pose_file, 'rb') as f:
            ego_cam_pose = pickle.load(f)
        print ('Loaded hand labels from disk')
    return hand_labels, ego_cam_pose

def load_estimated_labels(args, motion_data, hand_labels):
    label_dir = args.label_dir
    new_motion_data = {}
    for k in motion_data:
        new_motion_data[k] = []
    for i in tqdm(range(len(motion_data['names']))):
        curr_name = motion_data['names'][i]
        curr_range = motion_data['ranges'][i]
        st, end = curr_range
        indices = list(range(st+1, end+1))
        
        # check if there are estimated labels for this sequence
        video_label_dir = os.path.join(label_dir, curr_name)
        if not os.path.exists(video_label_dir):
            continue
        
        valid_range = True
        for idx in indices:
            if isinstance(idx, int):
                ind = str(idx).zfill(6)
            else:
                ind = idx
            label_file = os.path.join(video_label_dir, f'{ind}.npz')
            if not os.path.exists(label_file):
                valid_range = False
                break
            # load labels
            label = np.load(label_file)
            label_dict = {key: label[key] for key in label}
            existing_labels = hand_labels[curr_name][ind]
            pose_r = label_dict['pose_r']
            transl_r = label_dict['transl_r']
            pose_l = label_dict['pose_l']
            transl_l = label_dict['transl_l']
            add_labels = (pose_r, transl_r, pose_l, transl_l)
            hand_labels[curr_name][ind] = existing_labels + add_labels

        if valid_range:
            # add to new motion data
            for k in motion_data:
                new_motion_data[k].append(motion_data[k][i])

    return hand_labels, new_motion_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--max_motion_length', type=int, default=256)
    parser.add_argument('--iter', type=int, default=1)
    args = parser.parse_args()

    args.label_dir = f'{os.environ["DOWNLOADS_DIR"]}/lifted_labels/holo_preds_iter_{args.iter:02d}'
    args.save_dir = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/holo'

    motion_data = load_motion_data(args)
    hand_labels, cam_info = load_hand_labels(args)
    hand_labels, new_motion_data = load_estimated_labels(args, motion_data, hand_labels)

    # save the new_motion_data and hand_labels in args.save_dir with suffix iter at the end
    save_file = os.path.join(args.save_dir, f'{args.split}_iter_{str(args.iter).zfill(2)}.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(new_motion_data, f)
    print ('Saved new motion data in', save_file)

    save_file = os.path.join(args.save_dir, f'hand_labels_{args.split}_iter_{str(args.iter).zfill(2)}.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(hand_labels, f)
    print ('Saved hand labels in', save_file)