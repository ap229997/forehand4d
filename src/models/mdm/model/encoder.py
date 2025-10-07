import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vit import vit


class ImageEncoder(nn.Module):
    def __init__(self, args, feat_dim):
        super().__init__()

        if feat_dim == 'vit':
            # Create backbone feature extractor
            self.vit_input_size = (256, 192) # for loading pretrained weights
            self.backbone = vit(args, img_size=self.vit_input_size)
        else:
            raise NotImplementedError('Only ViT-B/32 is supported for now.')
        
        if args.use_pretrained_feats:
            # load pretrained weights from hamer backbone
            ckpt = os.path.join(os.environ['DOWNLOADS_DIR'], 'model/hamer/hamer.ckpt')
            assert ckpt is not None, f'Pretrained HaMeR weights not found at {ckpt}'
            state_dict = torch.load(ckpt, map_location='cpu')['state_dict']
            # extract backbone weights and load into model
            state_dict = {k[9:]: v for k, v in state_dict.items() if k.startswith('backbone.')}
            self.backbone.load_state_dict(state_dict, strict=True)

            if args.freeze_pretrained_feats:
                for param in self.backbone.parameters():
                    param.requires_grad = False
        
    def forward(self, x):
        x = F.interpolate(x, size=max(self.vit_input_size), mode='bilinear', align_corners=False)
        x = self.backbone(x[:,:,:,32:-32])
        return x


class JointConditioner(nn.Module):
    def __init__(self, args, latent_dim, **kwargs):
        super(JointConditioner, self).__init__()
        self.args = args
        self.cond_mode = args.get('cond_mode', 'no_cond')
        inp_dim = 0
        if 'pose' in self.cond_mode:
            pose_dim = kwargs.get('pose_dim', None)
            assert pose_dim is not None
            inp_dim += pose_dim

        if 'img' in self.cond_mode:
            img_feat = kwargs.get('img_feat', None)
            assert img_feat is not None
            self.feat_extractor = ImageEncoder(args, img_feat)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            inp_dim += self.feat_extractor.backbone.num_features

        if 'cam' in self.cond_mode:
            cam_dim = self.args.max_motion_length * 4 * 4 # camera matrix (4x4) for each frame
            inp_dim += cam_dim

        self.net = nn.Sequential(
            nn.Linear(inp_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, cond):
        all_conds = []
        if 'pose' in cond:
            all_conds.append(cond['pose'])
        
        if 'img' in cond:
            img_feat = self.feat_extractor(cond['img'])
            # if img_feat is (bz, ch, h, w), take avgpool across last 2 dim
            # TODO: need to modify to better to preserve spatial info
            if len(img_feat.shape) == 4:
                img_feat = self.avgpool(img_feat).reshape(img_feat.shape[0], -1)
            
            if self.args.get('interpolate', False):
                goal_img_feat = self.feat_extractor(cond['goal_img'])
                if len(goal_img_feat.shape) == 4:
                    goal_img_feat = self.avgpool(goal_img_feat).reshape(goal_img_feat.shape[0], -1)
                img_feat = img_feat + goal_img_feat
            
            all_conds.append(img_feat)

        if 'cam' in cond:
            all_conds.append(cond['cam'])

        out = self.net(torch.cat(all_conds, dim=-1))

        return out
    

class SpatialConditioner(nn.Module):
    def __init__(self, args, latent_dim, **kwargs):
        super().__init__()

        img_feat = kwargs.get('img_feat', None)
        assert img_feat is not None
        self.feat_extractor = ImageEncoder(args, img_feat)
        
        inp_dim = self.feat_extractor.backbone.num_features
        self.net = nn.Sequential(
            nn.Linear(inp_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, img):
        img_feat = self.feat_extractor(img) # (bz, ch, h, w)
        bz, c = img_feat.shape[:2]
        img_feat = img_feat.reshape(bz, c, -1).transpose(1,2)  # (bz, h*w, ch)
        img_feat = self.net(img_feat)
        return img_feat
    

class KpsConditioner(nn.Module):
    def __init__(self, args, latent_dim, **kwargs):
        super().__init__()
        self.args = args
        self.cond_mode = args.get('cond_mode', 'no_cond')
        if 'j2d' in self.cond_mode:
            inp_dim = 21 * 2 * 2  # 21 joints, each with x and y coordinates for both hands
        if 'j3d' in self.cond_mode:
            inp_dim = 21 * 3 * 2

        self.net = nn.Sequential(
            nn.Linear(inp_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, kps):
        out = self.net(kps)
        return out
    

class CameraConditioner(nn.Module):
    def __init__(self, args, latent_dim, **kwargs):
        super().__init__()
        self.args = args
        self.cond_mode = args.get('cond_mode', 'no_cond')
        # cam_dim = self.args.max_motion_length * 4 * 4 # camera matrix (4x4) for each frame
        cam_dim = 4 * 4
        if 'future_plucker' in self.cond_mode and kwargs.get('use_plucker', False):
            cam_dim = 2 * 21 * 6 # 21 joints in both hands, each with 6 plucker coordinates
        if 'future_kpe' in self.cond_mode and kwargs.get('use_kpe', False):
            freq = self.args.get('kpe_freq', 4)
            cam_dim = 2 * 21 * 2 * freq * 2 # sin, cos for each joint, 21 joints, 2 dimensions (x,y)
        if 'current_kpe' in self.cond_mode and kwargs.get('use_kpe', False):
            freq = self.args.get('kpe_freq', 4)
            cam_dim = 2 * 21 * 2 * freq * 2 # sin, cos for each joint, 21 joints, 2 dimensions (x,y)
        
        inp_dim = cam_dim

        self.net = nn.Sequential(
            nn.Linear(inp_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, cam):
        out = self.net(cam)
        return out