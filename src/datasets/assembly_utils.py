# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os, math, random
import pickle
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from scipy import linalg


# define clockwise rotation angles for cameras to make them upright
CAM_ROT = {
    '84346135': 0,
    '84347414': 180,
    '84355350': 270,
    '84358933': 270,
    '21110305': 270,
    '21176623': 180,
    '21176875': 0,
    '21179183': 270,
}


class BaseConfig(object):
    def __init__(self):
        """
        Base config for model and heatmap generation
        """
        ## input, output
        self.input_img_shape = (256, 256)
        self.output_hm_shape = (64, 64, 64)  # (depth, height, width)
        self.sigma = 2.5
        self.bbox_3d_size = 400  # depth axis
        self.bbox_3d_size_root = 400  # depth axis
        self.output_root_hm_shape = 64  # depth axis

        ## model
        self.resnet_type = 50  # 18, 34, 50, 101, 152

    def print_config(self):
        """
        Print configuration
        """
        print(">>> Configuration:")
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
cfg = BaseConfig()

def load_img(path, order='RGB'):
    
    # load
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def update_params_after_crop(bbox, pts_2d, joint_world, joint_valid, retval_camera, img_size, dataset='assemblyhands'):
    (img_h, img_w) = img_size
    x0, y0, w, h = bbox
    box_l = max(int(max(w, h)), 100) # at least 100px
    x0 = int((x0 + (x0 + w)) * 0.5 - box_l * 0.5)
    y0 = int((y0 + (y0 + h)) * 0.5 - box_l * 0.5)
    x1, y1 = x0 + box_l, y0 + box_l
        
    # change coordinates
    bbox = [0, 0, box_l, box_l]
    pts_2d[:, 0] -= x0
    pts_2d[:, 1] -= y0
    # 2d visibility check
    joint_valid = (joint_valid > 0 & (pts_2d[:, :2].max(axis=1) < box_l))
    joint_valid = joint_valid.astype(int)    
    
    retval_camera.update_after_crop([x0, y0, x1, y1])    
    
    campos, camrot, focal, princpt = retval_camera.get_params()
    if dataset == 'assemblyhands':
        joint_cam = world2cam_assemblyhands(joint_world, camrot, campos)
    else:
        joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
    
    return (x0, y0, x1, y1), bbox, pts_2d, joint_cam, joint_valid, retval_camera

def load_img(path, order='RGB'):
    
    # load
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def pad_img_square(img):
    # pad img_og with zeros to make square image
    height, width = img.shape[0], img.shape[1]
    c_x, c_y = width//2, height//2
    size = max(height, width)
    img_pad = np.zeros((size, size, 3), dtype=np.uint8)
    img_pad[size//2-height//2:size//2+height//2, size//2-width//2:size//2+width//2] = img
    diff_c_x, diff_c_y = size//2-c_x, size//2-c_y
    return img_pad, (diff_c_x, diff_c_y)

def crop_and_pad(img, bbox, img_res, scale=1.5): # only for square images
    # crop image around bbox
    if bbox is None:
        img_crop, _, _ = generate_patch_image(img, [0, 0, img.shape[0], img.shape[1]], do_flip=False, scale=1.0, rot=0.0, out_shape=(img_res, img_res))
        new_bbox = np.array([0, 0, img_res-1, img_res-1])
        return img_crop, new_bbox.astype(np.int16)

    x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])
    x_mid, y_mid, width, height = (x0+x1)//2, (y0+y1)//2, x1-x0, y1-y0 
    bb_size = max(width, height)
    new_bbox = np.array([x_mid-(bb_size*scale)//2, y_mid-(bb_size*scale)//2, x_mid+(bb_size*scale)//2, y_mid+(bb_size*scale)//2]).astype(np.int16)
    
    new_img = Image.fromarray(img.astype(np.uint8))
    new_img = new_img.crop(list(new_bbox))
    new_img = np.asarray(new_img).astype(np.float32)
    img_crop, _, _ = generate_patch_image(new_img, [0, 0, new_img.shape[0], new_img.shape[1]], do_flip=False, scale=1.0, rot=0.0, out_shape=(img_res, img_res))

    scale = img_res / img.shape[0]
    new_bbox = new_bbox * scale

    return img_crop, new_bbox.astype(np.int16)

def load_crop_img(path, bbox, pts_2d, joint_world, joint_valid, retval_camera, order='RGB'):
    
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    
    (x0, y0, x1, y1), bbox, pts_2d, joint_cam, joint_valid, retval_camera = update_params_after_crop(bbox, pts_2d, joint_world, joint_valid, retval_camera, img.shape[:2])
    assert sum(joint_valid) >= 10, f"sum(joint_valid): {sum(joint_valid)}, path: {path}, joint_valid: {joint_valid}"
    # crop    
    img = Image.fromarray(img.astype(np.uint8))
    img = img.crop([x0, y0, x1, y1])
    img = np.asarray(img)
    assert img.shape[0] > 50 and img.shape[1] > 50, f"img.shape: {img.shape}, path: {path}, bbox: {x0}, {y0}, {bbox[2]}"
    
    return img, bbox, pts_2d, joint_cam, joint_valid, retval_camera

def load_skeleton(path, joint_num):

    # load joint_world info (name, parent_id)
    skeleton = [{} for _ in range(joint_num)]
    with open(path) as fp:
        for line in fp:
            if line[0] == '#': continue
            splitted = line.split(' ')
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            skeleton[joint_id]['name'] = joint_name
            skeleton[joint_id]['parent_id'] = joint_parent_id
    # save child_id
    for i in range(len(skeleton)):
        joint_child_id = []
        for j in range(len(skeleton)):
            if skeleton[j]['parent_id'] == i:
                joint_child_id.append(j)
        skeleton[i]['child_id'] = joint_child_id
    
    return skeleton

def get_aug_config():
    trans_factor = 0.15
    scale_factor = 0.25
    rot_factor = 45
    color_factor = 0.2
    
    trans = [np.random.uniform(-trans_factor, trans_factor), np.random.uniform(-trans_factor, trans_factor)]
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    do_flip = random.random() <= 0.5
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])

    return trans, scale, rot, do_flip, color_scale

def augmentation(img, bbox, joint_coord, joint_valid, hand_type, mode, joint_type, input_img_shape, no_aug=False):
    img = img.copy() 
    joint_coord = joint_coord.copy()
    hand_type = hand_type.copy()

    original_img_shape = img.shape
    joint_num = len(joint_coord)
    
    if mode == 'train' and not no_aug:
        trans, scale, rot, do_flip, color_scale = get_aug_config()
    else:
        trans, scale, rot, do_flip, color_scale = [0,0], 1.0, 0.0, False, np.array([1,1,1])
        # trans, scale, rot, do_flip, color_scale = get_aug_config()
    
    bbox[0] = bbox[0] + bbox[2] * trans[0]
    bbox[1] = bbox[1] + bbox[3] * trans[1]
    img, trans, inv_trans = generate_patch_image(img, bbox, do_flip, scale, rot, input_img_shape)
    img = np.clip(img * color_scale[None,None,:], 0, 255)
    
    if do_flip:
        joint_coord[:,0] = original_img_shape[1] - joint_coord[:,0] - 1
        joint_coord[joint_type['right']], joint_coord[joint_type['left']] = joint_coord[joint_type['left']].copy(), joint_coord[joint_type['right']].copy()
        joint_valid[joint_type['right']], joint_valid[joint_type['left']] = joint_valid[joint_type['left']].copy(), joint_valid[joint_type['right']].copy()
        hand_type[0], hand_type[1] = hand_type[1].copy(), hand_type[0].copy()
    for i in range(joint_num):
        joint_coord[i,:2] = trans_point2d(joint_coord[i,:2], trans)
        joint_valid[i] = joint_valid[i] * (joint_coord[i,0] >= 0) * (joint_coord[i,0] < input_img_shape[1]) * (joint_coord[i,1] >= 0) * (joint_coord[i,1] < input_img_shape[0])

    return img, joint_coord, joint_valid, hand_type, inv_trans

def transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, root_valid, root_joint_idx, joint_type):
    # transform to output heatmap space
    joint_coord = joint_coord.copy()
    joint_valid = joint_valid.copy()
    
    joint_coord[:,0] = joint_coord[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
    joint_coord[:,1] = joint_coord[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
    joint_coord[joint_type['right'],2] = joint_coord[joint_type['right'],2] - joint_coord[root_joint_idx['right'],2]
    joint_coord[joint_type['left'],2] = joint_coord[joint_type['left'],2] - joint_coord[root_joint_idx['left'],2]
  
    joint_coord[:,2] = (joint_coord[:,2] / (cfg.bbox_3d_size/2) + 1)/2. * cfg.output_hm_shape[0]
    joint_valid = joint_valid * ((joint_coord[:,2] >= 0) * (joint_coord[:,2] < cfg.output_hm_shape[0])).astype(np.float32)
    rel_root_depth = (rel_root_depth / (cfg.bbox_3d_size_root/2) + 1)/2. * cfg.output_root_hm_shape
    root_valid = root_valid * ((rel_root_depth >= 0) * (rel_root_depth < cfg.output_root_hm_shape)).astype(np.float32)
    
    return joint_coord, joint_valid, rel_root_depth, root_valid

def get_bbox(joint_img, joint_valid):
    x_img = joint_img[:,0][joint_valid==1]
    y_img = joint_img[:,1][joint_valid==1]
    xmin = min(x_img)
    ymin = min(y_img)
    xmax = max(x_img)
    ymax = max(y_img)

    x_center = (xmin+xmax)/2.
    width = xmax-xmin
    xmin = x_center - 0.5*width*1.2
    xmax = x_center + 0.5*width*1.2
    
    y_center = (ymin+ymax)/2.
    height = ymax-ymin
    ymin = y_center - 0.5*height*1.2
    ymax = y_center + 0.5*height*1.2

    bbox = np.array([xmin, ymin, xmax-xmin, ymax-ymin]).astype(np.float32)
    return bbox

def process_bbox(bbox, input_img_shape, scale=1.25):

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = input_img_shape[1]/input_img_shape[0]
    if w > aspect_ratio * h:
        h = w * aspect_ratio # this should be h = w * aspect_ratio, og code: h = w / aspect_ratio is wrong
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * scale
    bbox[3] = h * scale
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.

    return bbox

def generate_patch_image(cvimg, bbox, do_flip, scale, rot, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1
    
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    try:
        img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    except:
        print(img.shape, (int(out_shape[1]), int(out_shape[0])))
        raise Exception()
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord

def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord

def world2cam_assemblyhands(pts_3d, R, t):
    pts_cam = np.dot(R, pts_3d.T).T + t
    return pts_cam

def cam2world_assemblyhands(pts_cam_3d, R, t):
    inv_R = np.linalg.inv(R)
    pts_3d = np.dot(inv_R, (pts_cam_3d - t).T).T
    
    return pts_3d

def world2pixel(pts_3d, KRT):
    assert pts_3d.shape[1] == 3, f"shape error: {pts_3d.shape}"
    _pts_3d = np.concatenate((pts_3d[:, :3], np.ones((pts_3d.shape[0], 1))), axis=-1)
    pts_2d = np.matmul(_pts_3d, KRT.T)
    pts_2d /= pts_2d[:, 2:3]
    
    return pts_2d

def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)


def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
                            dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    assert flipped.device == tensor.device
    assert flipped.requires_grad == tensor.requires_grad
    return flipped

class Camera(object):
    def __init__(self, K, Rt, dist=None, name=""):
        # Rotate first then translate
        self.K = np.array(K).copy()
        assert self.K.shape == (3, 3)

        self.Rt = np.array(Rt).copy()
        assert self.Rt.shape == (3, 4)

        self.dist = dist
        if self.dist is not None:
            self.dist = np.array(self.dist).copy().flatten()

        self.name = name

    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_height, new_width = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_cx, new_cy

    @property
    def projection(self):
        return np.dot(self.K, self.Rt)

    def factor(self):
        """  Factorize the camera matrix into K,R,t as P = K[R|t]. """

        # factor first 3*3 part
        K,R = linalg.rq(self.projection[:, :3])

        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:
            T[1,1] *= -1

        K = np.dot(K,T)
        R = np.dot(T,R) # T is its own inverse
        t = np.dot(linalg.inv(self.K), self.projection[:,3])

        return K, R, t

    def get_params(self):
        K, R, t = self.factor()
        campos, camrot = t, R
        focal = [K[0, 0], K[1, 1]]
        princpt = [K[0, 2], K[1, 2]]
        return campos, camrot, focal, princpt
    

def load_motion_data(args):
    # this is generated using _generate_motion_data above
    data_file = os.path.join(args.save_dir, f'{args.split}.pkl')
    with open(data_file, 'rb') as f:
        motion_data = pickle.load(f) # dict of lists
    for ind in range(len(motion_data['names'])):
        st, end = motion_data['ranges'][ind]
        assert end > st, f"Invalid range: {st}, {end}"
        end = min(end, args.max_motion_length-2)
        motion_data['ranges'][ind] = (st, end)
    return motion_data


def load_hand_labels(args):
    save_dir = args.save_dir
    hand_label_file = os.path.join(save_dir, f'hand_labels_{args.split}.pkl')
    cam_info_file = os.path.join(save_dir, f'cam_info_{args.split}.pkl')
    if os.path.exists(hand_label_file) and os.path.exists(cam_info_file):
        with open(hand_label_file, 'rb') as f:
            hand_labels = pickle.load(f)
        with open(cam_info_file, 'rb') as f:
            cam_info = pickle.load(f)
        print ('Loaded hand labels from disk')
    return hand_labels, cam_info


def load_estimated_labels(args, motion_data, hand_labels):
    label_dir = args.label_dir
    print ('Loading estimated labels from', label_dir)
    new_motion_data = {'names': [], 'ranges': [], 'subsampled_indices': []}
    for i in tqdm(range(len(motion_data['names']))):
        curr_name = motion_data['names'][i]
        video_name = curr_name[0].replace('_mono10bit', '')
        curr_range = motion_data['ranges'][i]
        st, end = curr_range
        if 'subsampled_indices' in motion_data:
            # predictions are done only for future indices, so start from st+1
            indices = motion_data['subsampled_indices'][i][st+1:end+1]
        else:
            indices = list(range(st+1, end+1))
        
        # check if there are estimated labels for this sequence
        video_label_dir = os.path.join(label_dir, curr_name[0])
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
            existing_labels = hand_labels[video_name][int(idx)]
            existing_labels.update(label_dict)
            hand_labels[video_name][int(idx)] = existing_labels

        if valid_range:
            # add to new motion data
            new_motion_data['names'].append(motion_data['names'][i])
            new_motion_data['ranges'].append(motion_data['ranges'][i])
            new_motion_data['subsampled_indices'].append(motion_data['subsampled_indices'][i])

    return hand_labels, new_motion_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--max_motion_length', type=int, default=256)
    parser.add_argument('--iter', type=int, default=1)
    args = parser.parse_args()

    args.label_dir = f'{os.environ["DOWNLOADS_DIR"]}/lifted_labels/assembly_preds_iter_{args.iter:02d}'
    args.save_dir = f'{os.environ["DOWNLOADS_DIR"]}/motion_splits/assembly'

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