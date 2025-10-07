import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import torch.nn.functional as F

import torch.optim as optim

import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from utils.eval_t2m import evaluation_vqvae
from utils.utils import print_current_loss
from utils.metrics import compute_mpjpe, compute_mpjpe_ra, compute_mpjpe_pa, compute_mpjpe_pa_first, binary_classification_metrics
import common.rotation_conversions as rot
from common.quaternion import batch_determinant

import os
import sys
import pickle

def def_value():
    return 0.0


class RVQTokenizerTrainer:
    def __init__(self, args, vq_model):
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device

        if args.is_train:
            if not args.inference:
                self.logger = SummaryWriter(args.log_dir)
            if args.recons_loss == 'l1':
                self.l1_criterion = torch.nn.L1Loss(reduction='none')
            elif args.recons_loss == 'l1_smooth':
                self.l1_criterion = torch.nn.SmoothL1Loss(reduction='none')

            if args.contact_map:
                self.cross_entropy = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(5.0)) # check if  this is helpful
                # self.cross_entropy = torch.nn.BCEWithLogitsLoss(reduction='none')

        # self.critic = CriticWrapper(self.opt.dataset_name, self.opt.device)

    def forward(self, batch_data, mode='train'):

        if isinstance(batch_data, dict):
            motions = batch_data['motion'].detach().to(self.device).float()
            # motions = batch_data['motion'].to(self.device).float()
        else:
            motions = batch_data.detach().to(self.device).float()

        if self.opt.pred_cam:
            if self.opt.coord_sys == 'contact':
                if self.opt.joints_loss:
                    cam_rot_nondiff = motions[..., -6:].detach().to(self.device).float()
                    cam_rot = motions[..., -6:].to(self.device).float()
                    if mode == 'train':
                        cam_rot.requires_grad = True
                    mano_rot = motions[..., :3].to(self.device).float()
                    if mode == 'train':
                        mano_rot.requires_grad = True
                else:
                    # ###### this is originally ######
                    # cam_rot = motions[..., -6:].detach().to(self.device).float()
                    # cam_rot_nondiff = cam_rot
                    # motions = motions[..., :-6]

                    ###### this is added ######
                    cam_rot = motions[..., -9:-3].detach().to(self.device).float()
                    cam_rot_nondiff = cam_rot
                    cam_transl = motions[..., -3:].detach().to(self.device).float()
                    motions = motions[..., :-9]
            else:
                cam_rot = motions[..., -9:-3].detach().to(self.device).float()
                cam_rot_nondiff = cam_rot
                cam_transl = motions[..., -3:].detach().to(self.device).float()
                motions = motions[..., :-9]
        
        more_dict = {}
        if self.opt.video_feats:
            video_feats = batch_data['video_feats'].detach().to(self.device).float()
            more_dict['video_feats'] = video_feats
        if self.opt.text_feats:
            text_feats = batch_data['text_feats'].detach().to(self.device).float()
            more_dict['text_feats'] = text_feats
        if self.opt.contact_grid is not None:
            contact = batch_data['contact'].detach().to(self.device).float()
            more_dict['contact'] = contact
            grid = batch_data['grid'].detach().to(self.device).float()
            more_dict['grid'] = grid
            contact_mask = batch_data['contact_mask'].detach().to(self.device).float()
            more_dict['contact_mask'] = contact_mask
            known_mask = contact_mask.reshape(-1,)
            self.known_mask = known_mask
            grid_mask = batch_data['grid_mask'].detach().to(self.device).float()
            more_dict['grid_mask'] = grid_mask

            if self.opt.coord_sys == 'contact':
                cam_t_contact = batch_data['cam_t_contact'].detach().to(self.device).float()
                bz, ts = cam_t_contact.shape[:2]
                known_mask = contact_mask.reshape(-1,)
                cam_t_contact = cam_t_contact.reshape(-1, 3)
                # replace unknown cam_t_contact with learnable parameters
                cam_t_contact[known_mask == 0] = self.vq_model.unknown_cam_t_contact
                cam_t_contact = cam_t_contact.reshape(bz, ts, 3)
                more_dict['cam_t_contact'] = cam_t_contact
                # replace last 3 values of motion with learnable parameters
                motions = torch.cat([motions[..., :61], cam_t_contact], dim=-1)
                self.known_mask = known_mask

                cam_t_contact_ref = batch_data['cam_t_contact_ref'].detach().to(self.device).float()
                bz, ts = cam_t_contact_ref.shape[:2]
                known_mask_ref = batch_data['contact_ref_mask'].detach().to(self.device).float().reshape(-1,)
                cam_t_contact_ref = cam_t_contact_ref.reshape(-1, 3)
                # replace unknown cam_t_contact with learnable parameters
                cam_t_contact_ref[known_mask_ref == 0] = self.vq_model.unknown_cam_t_contact_ref
                cam_t_contact_ref = cam_t_contact_ref.reshape(bz, ts, 3)
                more_dict['cam_t_contact_ref'] = cam_t_contact_ref
                # replace last 3 values of motion with learnable parameters
                motions = torch.cat([motions[..., :64], cam_t_contact_ref], dim=-1)
                self.known_mask_ref = known_mask_ref

        known_cam_mask = batch_data['cam_mask'].detach().to(self.device).float().reshape(-1,)
        self.known_cam_mask = known_cam_mask

        bz, ts = motions.shape[:2]
        trainable_mask = torch.ones((bz, ts, self.opt.dim_pose)).to(self.device)
        if self.opt.pred_cam:
            if not self.opt.coord_sys == 'contact':
                cam_transf = torch.cat([cam_rot_nondiff, cam_transl], dim=-1)
            else:
                # cam_transf = cam_rot_nondiff # this is originally used
                cam_transf = torch.cat([cam_rot_nondiff, cam_transl], dim=-1) # this is added
            
            # # cam_transf is 0:3 + 58:61 from motions - this doesn't work well
            # cam_transf = torch.cat([motions[..., :3], motions[..., 58:61]], dim=-1)
            
            bz, ts = cam_transf.shape[:2]
            cam_transf = cam_transf.reshape(bz*ts, -1)
            
            # # replace unknown cam_transf with learnable parameters
            cam_transf[known_cam_mask == 0] = self.vq_model.unknown_cam_transf

            # append cam_transf to motions as last 9 values
            cam_transf = cam_transf.reshape(bz, ts, -1)
            
            # # replace 0:3 + 58:61 values of motion with learnable parameters - this doesn't work well
            # motions[..., :3] = cam_transf[..., :3]
            # motions[..., 58:61] = cam_transf[..., 3:]
            
            motions = torch.cat([motions, cam_transf], dim=-1)

            # trainable_mask[..., :3] = 0
            # trainable_mask[..., -6:] = 0
            # trainable_mask[..., 58:61] = 0
            if self.opt.joints_loss:
                # add trainable rotation parameters separately
                # motions = torch.cat([mano_rot, motions[..., 3:-9], cam_rot, cam_transf[...,-3:]], dim=-1) # check this
                trainable_mask[..., :3] = 0
                if not self.opt.coord_sys == 'contact':
                    trainable_mask[..., -9:-3] = 0
                    motions = torch.cat([mano_rot, motions[..., 3:-9], cam_rot, cam_transf[...,-3:]], dim=-1)
                else:
                    trainable_mask[..., -6:] = 0
                    trainable_mask[..., 58:61] = 0
                    motions = torch.cat([mano_rot, motions[..., 3:-6], cam_rot], dim=-1)

        trainable_mask = trainable_mask * known_cam_mask.reshape(bz, ts, -1)
        if self.opt.coord_sys == 'contact':
            trainable_mask = trainable_mask * known_mask.reshape(bz, ts, -1)* known_mask_ref.reshape(bz, ts, -1)
        
        if self.opt.contact_map:
            gt_contact_map = batch_data['contact_map'].detach().to(self.device).float()
            motions = torch.cat([motions, gt_contact_map], dim=-1)

        if self.opt.decoder_only:
            more_dict['code_idx'] = batch_data['code_idx'].detach().to(self.device).long()
        
        pred_motion, loss_commit, perplexity, more_outs = self.vq_model(motions, **more_dict)
        more_outs['pred_motion'] = pred_motion.clone() # return pred_motion for visualizations and evaluation

        if self.opt.text2hoi:
            self.t2h_motion = batch_data['text2hoi_preds'].to(self.device) # B X 4 X T X 21 X 3

        more_loss = {}
        if self.opt.contact_map:
            pred_contact_map = pred_motion[..., -778:]
            loss_contact_map = self.cross_entropy(pred_contact_map, gt_contact_map)
            loss_contact_map = torch.mean(loss_contact_map * known_mask.reshape(bz, ts, -1))
            more_loss['contact_map'] = loss_contact_map

            # compute classification metrics
            logits = pred_contact_map.detach().clone()
            labels = gt_contact_map.detach().clone()
            metric_mask = known_mask.reshape(bz, ts, 1) # .repeat(1, 1, logits.shape[-1])
            precision, recall, f1 = binary_classification_metrics(logits, labels, mask=metric_mask)
            more_loss['precision'] = precision
            more_loss['recall'] = recall
            more_loss['f1'] = f1
            more_loss['logits'] = logits
            more_loss['labels'] = labels

            motions = motions[..., :-778]
            pred_motion = pred_motion[..., :-778]

        if self.opt.diffusion:
            diff_loss = more_outs['diff_mse']
            if self.opt.latent:
                diff_mask = (trainable_mask==1).all(dim=-1)
                more_loss['diff_loss'] = torch.mean(diff_loss * diff_mask.reshape(bz, ts, -1))
            else:
                diff_motion_loss = diff_loss[..., :self.opt.dim_pose]
                diff_mask = (trainable_mask==1).all(dim=-1)
                diff_motion_loss = torch.mean(diff_motion_loss * diff_mask.reshape(bz, ts, -1))
                more_loss['diff_loss'] = diff_motion_loss
                if self.opt.contact_map:
                    diff_contact_loss = diff_loss[..., -778:]
                    diff_contact_loss = torch.mean(diff_contact_loss * known_mask.reshape(bz, ts, -1))
                    more_loss['diff_loss'] += diff_contact_loss

        loss_rec = self.l1_criterion(pred_motion, motions)
        loss_rec = torch.mean(loss_rec * trainable_mask)

        if self.opt.dataset_name == 'holo' or self.opt.dataset_name == 'arctic': # only using joint positions for now
            if 'mano' in self.opt.motion_type:
                # compute cam_t loss
                # loss_explicit = torch.mean(torch.abs(pred_motion[...,58:] - motions[...,58:]))
                # loss_explicit = self.l1_criterion(pred_motion[...,58:], motions[...,58:])
                loss_explicit = self.l1_criterion(pred_motion, motions)
            else:
                loss_explicit = self.l1_criterion(pred_motion, motions)
        else:
            pred_local_pos = pred_motion[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
            local_pos = motions[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
            loss_explicit = self.l1_criterion(pred_local_pos, local_pos)
        # loss_explicit = torch.mean(loss_explicit * trainable_mask)
        # loss_explicit = torch.mean(loss_explicit[..., 3:58])
        # add rot and transl separately, this works better than above
        loss_explicit = torch.mean(loss_explicit[..., :61]) + torch.mean(loss_explicit[..., -9:]) # + torch.mean(loss_explicit[..., 61:64]*trainable_mask[...,0:1]) + torch.mean(loss_explicit[..., 64:67]*trainable_mask[...,0:1])

        loss = loss_rec + self.opt.loss_vel * loss_explicit + self.opt.commit * loss_commit
        if self.opt.contact_map:
            loss = loss + 0.1*loss_contact_map # check what weight to use

        if 'joints' in self.opt.motion_type:
            if self.opt.pred_cam:
                self.motions = motions[...,:-9]
                self.pred_motion = pred_motion[...,:-9]
                self.gt_cam_rot = motions[...,-9:-3]
                self.gt_cam_transl = motions[...,-3:]
                self.pred_cam_rot = pred_motion[...,-9:-3]
                self.pred_cam_transl = pred_motion[...,-3:]
            else:
                self.motions = motions
                self.pred_motion = pred_motion
        elif 'mano' in self.opt.motion_type:
            bz, T, _ = motions.shape
            thetas_ = motions[..., :48].reshape(-1, 48)
            betas_ = motions[..., 48:58].reshape(-1, 10)
            joints_mano_ = self.vq_model.mano(global_orient=thetas_[:, :3], hand_pose=thetas_[:, 3:], betas=betas_)
            joints_mano = joints_mano_.joints.reshape(bz, T, -1, 3)
            joints_mano = joints_mano[:, :, self.vq_model.mano_to_openpose, :]
            gt_cam_t = motions[..., 58:61].unsqueeze(2) * self.mano_std[58:61].reshape(1,1,1,-1) + self.mano_mean[58:61].reshape(1,1,1,-1) # unnormalize
            joints_cam = joints_mano + gt_cam_t # motions[..., 58:].unsqueeze(2)
            self.motions = joints_cam
            self.gt_cam_t = gt_cam_t

            # repeat for pred_motion
            pred_thetas_ = pred_motion[...,:48].reshape(-1, 48)
            pred_betas_ = pred_motion[..., 48:58].reshape(-1, 10)
            pred_joints_mano_ = self.vq_model.mano(global_orient=pred_thetas_[:, :3], hand_pose=pred_thetas_[:, 3:], betas=pred_betas_)
            pred_joints_mano = pred_joints_mano_.joints.reshape(bz, T, -1, 3)
            pred_joints_mano = pred_joints_mano[:, :, self.vq_model.mano_to_openpose, :]
            pred_cam_t = pred_motion[..., 58:61].unsqueeze(2) * self.mano_std[58:61].reshape(1,1,1,-1) + self.mano_mean[58:61].reshape(1,1,1,-1) # unnormalize
            pred_joints_cam = pred_joints_mano + pred_cam_t # pred_motion[..., 58:].unsqueeze(2)
            self.pred_motion = pred_joints_cam
            self.pred_cam_t = pred_cam_t

            if self.opt.coord_sys == 'contact':
                gt_cam_t_contact = motions[..., 61:64].unsqueeze(2) * self.mano_std[61:64].reshape(1,1,1,-1) + self.mano_mean[61:64].reshape(1,1,1,-1) # unnormalize
                joints_cam_contact = joints_mano + gt_cam_t_contact
                self.motions_contact = joints_cam_contact
                self.gt_cam_t_contact = gt_cam_t_contact

                pred_cam_t_contact = pred_motion[..., 61:64].unsqueeze(2) * self.mano_std[61:64].reshape(1,1,1,-1) + self.mano_mean[61:64].reshape(1,1,1,-1) # unnormalize
                pred_joints_cam_contact = pred_joints_mano + pred_cam_t_contact
                self.pred_motion_contact = pred_joints_cam_contact
                self.pred_cam_t_contact = pred_cam_t_contact

                self.gt_cam_t_contact_ref = motions[..., 64:67].unsqueeze(2) * self.mano_std[64:67].reshape(1,1,1,-1) + self.mano_mean[64:67].reshape(1,1,1,-1) # unnormalize
                self.pred_cam_t_contact_ref = pred_motion[..., 64:67].unsqueeze(2) * self.mano_std[64:67].reshape(1,1,1,-1) + self.mano_mean[64:67].reshape(1,1,1,-1) # unnormalize
                # # transform to contact centroid at reference timestep
                # rel_c2c_ref = batch_data['rel_c2c'].detach().to(self.device).float()
                # # rotate pred_cam_t_contact by rel_c2c_ref[:3,:3]
                # pred_cam_t_contact_ref = pred_cam_t_contact_ref + torch.matmul(pred_cam_t_contact, rel_c2c_ref[:3,:3].transpose(0,1))
                # pred_joints_cam_contact_ref = pred_joints_mano + pred_cam_t_contact_ref
                # self.pred_motion_contact_ref = pred_joints_cam_contact_ref

                # transform joints to reference coordinate frame
                bz, ts = motions.shape[:2]
                gt_joints_ref = batch_data['joints_ref'].detach().to(self.device).float()
                gt_joints_contact_ref = gt_joints_ref.reshape(bz, ts, self.opt.joints_num, 3) # - self.gt_cam_t_contact_ref # cam_ref to contact_ref
                # repeat for pred_motion
                
                ###### this is originally used ######
                # pred_rot_contact_ref = rot.rotation_6d_to_matrix(pred_motion[..., -6:])
                # pred_joints_contact_ref = torch.matmul(pred_joints_cam_contact.reshape(-1, self.opt.joints_num, 3), pred_rot_contact_ref.reshape(-1, 3, 3).transpose(1,2)) + self.pred_cam_t_contact_ref.reshape(-1, 1, 3)
                
                ###### this is added ######
                pred_rot_contact_ref = rot.rotation_6d_to_matrix(pred_motion[..., -9:-3])
                pred_joints_contact_ref = torch.matmul(pred_rot_contact_ref.reshape(-1, 3, 3), pred_joints_cam.reshape(-1, self.opt.joints_num, 3).transpose(1,2)).transpose(1,2)
                pred_joints_contact_ref = pred_joints_contact_ref + pred_motion[..., -3:].reshape(-1, 1, 3)
                pred_joints_contact_ref = pred_joints_contact_ref - batch_data['contact_point'][:,0:1].to(self.device).float().repeat(1, ts, 1).reshape(-1, 1, 3)

                # pred_joints_contact_ref = torch.matmul(pred_rot_contact_ref.reshape(-1, 3, 3), pred_joints_cam_contact.reshape(-1, self.opt.joints_num, 3).transpose(1,2)).transpose(1,2) + self.pred_cam_t_contact_ref.reshape(-1, 1, 3)
                pred_joints_contact_ref = pred_joints_contact_ref.reshape(bz, ts, self.opt.joints_num, 3)
                self.gt_joints_frame_ref = gt_joints_contact_ref
                self.pred_joints_frame_ref = pred_joints_contact_ref
                joints_ref_mask = batch_data['joints_ref_mask'].detach().to(self.device).float().reshape(-1,)
                combined_mask = joints_ref_mask * self.known_mask_ref * self.known_mask
                self.frame_ref_mask = combined_mask

                if self.opt.joints_loss:
                    loss_joints = F.smooth_l1_loss(pred_joints_contact_ref, gt_joints_contact_ref, reduction='none')
                    # loss_joints = F.l1_loss(pred_joints_contact_ref, gt_joints_contact_ref, reduction='none')
                    loss_joints = torch.mean(loss_joints.reshape(bz*ts,-1) * combined_mask.reshape(-1, 1))
                    loss = loss + 0.5*loss_joints # 0.1 works well, 1 is too high
                    more_loss['joints_ref'] = loss_joints
                    
                    # add regularization term for pred_rot_contact_ref and pred_rot_mano_contact
                    reg_rot_contact_ref = torch.norm(pred_rot_contact_ref - torch.eye(3).to(self.device), dim=(2,3))
                    reg_rot_contact_ref = torch.mean(reg_rot_contact_ref * combined_mask.reshape(bz, ts))
                    pred_rot_mano_contact = rot.axis_angle_to_matrix(pred_motion[..., :3])
                    reg_rot_mano_contact = torch.norm(pred_rot_mano_contact - torch.eye(3).to(self.device), dim=(2,3))
                    reg_rot_mano_contact = torch.mean(reg_rot_mano_contact * combined_mask.reshape(bz, ts))
                    reg_loss = reg_rot_contact_ref + reg_rot_mano_contact

                    # # regularize determinant of rotation matrix to be 1
                    # det_rot_contact_ref = batch_determinant(pred_rot_contact_ref)
                    # det_rot_mano_contact = batch_determinant(pred_rot_mano_contact)
                    # reg_det_rot_contact_ref = torch.mean(torch.abs(det_rot_contact_ref - 1))
                    # reg_det_rot_mano_contact = torch.mean(torch.abs(det_rot_mano_contact - 1))
                    # reg_loss = reg_loss + reg_det_rot_contact_ref + reg_det_rot_mano_contact

                    # # transpose of rotation matrix should be inverse
                    # inv_rot_contact_ref = torch.inverse(pred_rot_contact_ref)
                    # inv_rot_mano_contact = torch.inverse(pred_rot_mano_contact)
                    # reg_inv_rot_contact_ref = torch.mean(torch.abs(inv_rot_contact_ref - pred_rot_contact_ref.transpose(-2,-1)))
                    # reg_inv_rot_mano_contact = torch.mean(torch.abs(inv_rot_mano_contact - pred_rot_mano_contact.transpose(-2,-1)))
                    # reg_loss = reg_loss + reg_inv_rot_contact_ref + reg_inv_rot_mano_contact

                    loss = loss + 0.1*reg_loss
            else:
                gt_joints_ref = batch_data['joints_ref'].detach().to(self.device).float()
                gt_joints_cam_ref = gt_joints_ref.reshape(bz, T, self.opt.joints_num, 3)
                if self.opt.pred_cam:
                    pred_rot_cam_ref = rot.rotation_6d_to_matrix(pred_motion[..., -9:-3])
                    pred_transl_cam_ref = pred_motion[..., -3:]
                    # pred_joints_cam_ref = torch.matmul(pred_joints_cam.reshape(-1, self.opt.joints_num, 3), pred_rot_cam_ref.reshape(-1, 3, 3).transpose(1,2)) + pred_transl_cam_ref.reshape(-1, 1, 3)
                    pred_joints_cam_ref = torch.matmul(pred_rot_cam_ref.reshape(-1, 3, 3), pred_joints_cam.reshape(-1, self.opt.joints_num, 3).transpose(1,2)).transpose(1,2) + pred_transl_cam_ref.reshape(-1, 1, 3)
                else:
                    gt_rot_cam_ref = batch_data['cam_rot_ref'].detach().to(self.device).float()
                    gt_rot_cam_ref = rot.rotation_6d_to_matrix(gt_rot_cam_ref)
                    gt_transl_cam_ref = batch_data['cam_transl_ref'].detach().to(self.device).float()
                    pred_joints_cam_ref = torch.matmul(pred_joints_cam.reshape(-1, self.opt.joints_num, 3), gt_rot_cam_ref.reshape(-1, 3, 3).transpose(1,2)) + gt_transl_cam_ref.reshape(-1, 1, 3)
                
                pred_joints_cam_ref = pred_joints_cam_ref.reshape(bz, T, self.opt.joints_num, 3)
                self.gt_joints_frame_ref = gt_joints_cam_ref
                self.pred_joints_frame_ref = pred_joints_cam_ref
                joints_ref_mask = batch_data['joints_ref_mask'].detach().to(self.device).float().reshape(-1,)
                combined_mask = joints_ref_mask * self.known_cam_mask
                self.frame_ref_mask = combined_mask

            if self.opt.pred_cam:
                if not self.opt.coord_sys == 'contact':
                    self.gt_cam_rot = motions[..., -9:-3]
                    self.gt_cam_transl = motions[..., -3:]
                    self.pred_cam_rot = pred_motion[..., -9:-3]
                    self.pred_cam_transl = pred_motion[..., -3:]
                else:
                    ###### this is originally used ######
                    # self.gt_cam_rot = motions[..., -6:]
                    # self.pred_cam_rot = pred_motion[..., -6:]

                    ###### this is added ######
                    self.gt_cam_rot = motions[..., -9:-3]
                    self.gt_cam_transl = motions[..., -3:]
                    self.pred_cam_rot = pred_motion[..., -9:-3]
                    self.pred_cam_transl = pred_motion[..., -3:]
        else:
            raise KeyError('Motion Type not recognized')

        more_loss.update(more_outs)
        return loss, loss_rec, loss_explicit, loss_commit, perplexity, more_loss
        # if self.opt.joints_loss:
        #     return loss, loss_rec, loss_explicit, loss_commit, perplexity, more_loss
        # else:
        #     return loss, loss_rec, loss_explicit, loss_commit, perplexity


    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep, total_it):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None, eval_loader=None):
        self.vq_model.to(self.device)

        # if 'holo' in self.opt.dataset_name:
        #     self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr)
        # else:
        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        
        if eval_val_loader is not None:
            print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(eval_val_loader)))
        else:
            print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        
        # val_loss = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        current_lr = self.opt.lr
        logs = defaultdict(def_value, OrderedDict())

        # sys.exit()
        if eval_wrapper is not None:
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_vqvae(
                self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=1000,
                best_div=100, best_top1=0,
                best_top2=0, best_top3=0, best_matching=100,
                eval_wrapper=eval_wrapper, save=False)
            
        if 'mano' in self.opt.motion_type:
            self.mano_mean = torch.from_numpy(train_loader.dataset.mean).to(self.device).float()
            self.mano_std = torch.from_numpy(train_loader.dataset.std).to(self.device).float()

        # while epoch < self.opt.max_epoch:
        for epoch in tqdm(range(epoch, self.opt.max_epoch)):
            self.vq_model.train()
            for i, batch_data in enumerate(tqdm(train_loader)):
                it += 1
                
                if it < self.opt.warm_up_iter and ('holo' not in self.opt.dataset_name and 'arctic' not in self.opt.dataset_name):
                    current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                
                # if self.opt.joints_loss:
                #     loss, loss_rec, loss_vel, loss_commit, perplexity, loss_joints = self.forward(batch_data)
                # else:
                #     loss, loss_rec, loss_vel, loss_commit, perplexity = self.forward(batch_data)
                
                loss, loss_rec, loss_vel, loss_commit, perplexity, more_loss = self.forward(batch_data)

                self.opt_vq_model.zero_grad()
                loss.backward()
                self.opt_vq_model.step()

                if it >= self.opt.warm_up_iter and ('holo' not in self.opt.dataset_name and 'arctic' not in self.opt.dataset_name):
                    self.scheduler.step()
                
                logs['loss'] += loss.item()
                logs['loss_rec'] += loss_rec.item()
                # Note it not necessarily velocity, too lazy to change the name now
                logs['loss_vel'] += loss_vel.item()
                logs['loss_commit'] += loss_commit.item()
                logs['perplexity'] += perplexity.item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']

                if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
                    if self.opt.joints_loss:
                        loss_joints = more_loss['joints_ref']
                        logs['loss_joints'] += loss_joints.item()

                    if self.opt.contact_map:
                        loss_contact_map = more_loss['contact_map']
                        logs['loss_contact_map'] += loss_contact_map.item()

                        logs['contact_precision'] += more_loss['precision']
                        logs['contact_recall'] += more_loss['recall']
                        logs['contact_f1'] += more_loss['f1']

                    if self.opt.diffusion:
                        loss_diff = more_loss['diff_loss']
                        logs['loss_diff'] += loss_diff.item()

                    # ########## what is the right mask for metrics ##########
                    # self.known_cam_mask = self.frame_ref_mask
                    # ########################################################
                    
                    curr_mpjpe = compute_mpjpe(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                               self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                               valid = self.known_cam_mask.reshape(-1))
                    logs['mpjpe'] += curr_mpjpe.item()
                    curr_mpjpe_ra = compute_mpjpe_ra(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                                     self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                                     valid = self.known_cam_mask.reshape(-1))
                    logs['mpjpe_ra'] += curr_mpjpe_ra.item()
                    curr_mpjpe_pa = compute_mpjpe_pa(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                                     self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                                     valid = self.known_cam_mask.reshape(-1))
                    logs['mpjpe_pa'] += curr_mpjpe_pa.item()
                    curr_mpjpe_ref = compute_mpjpe(self.pred_joints_frame_ref.detach().reshape(-1, self.opt.joints_num, 3), 
                                                   self.gt_joints_frame_ref.detach().reshape(-1, self.opt.joints_num, 3),
                                                   valid = self.frame_ref_mask.reshape(-1))
                    logs['mpjpe_ref'] += curr_mpjpe_ref.item()
                    l1_cam_t = torch.linalg.norm(self.pred_cam_t - self.gt_cam_t, dim=-1)
                    l1_cam_t = torch.nanmean(l1_cam_t.reshape(-1, 1) * self.known_cam_mask.reshape(-1, 1))
                    logs['l1_cam_t'] += l1_cam_t.item()

                    bz = self.pred_joints_frame_ref.shape[0]
                    # procrustes alignment at the global level
                    curr_mpjpe_pa_g = compute_mpjpe_pa(self.pred_joints_frame_ref.detach().reshape(bz, -1, 3),
                                                            self.gt_joints_frame_ref.detach().reshape(bz, -1, 3),
                                                            valid = self.frame_ref_mask.reshape(bz, -1))
                    logs['mpjpe_pa_g'] += curr_mpjpe_pa_g.item()
                    
                    # procrustes alignment at the first frame
                    curr_mpjpe_pa_f = compute_mpjpe_pa_first(self.pred_joints_frame_ref.detach().reshape(bz, self.opt.max_motion_length, -1, 3),
                                                            self.gt_joints_frame_ref.detach().reshape(bz, self.opt.max_motion_length, -1, 3),
                                                            valid = self.frame_ref_mask)
                    logs['mpjpe_pa_f'] += curr_mpjpe_pa_f.item()

                    if self.opt.text2hoi:
                        bz, num_preds = self.t2h_motion.shape[:2]
                        all_mpjpe_pas = []
                        for b in range(bz):
                            for j in range(num_preds):
                                curr_pred = self.t2h_motion[b,j]
                                # compute mpjpe for text2hoi predictions
                                try:
                                    # c_mpjpe_pa = compute_mpjpe_pa(curr_pred.detach().reshape(1, -1, 3),
                                    #                             self.gt_joints_frame_ref[b].detach().reshape(1, -1, 3),
                                    #                             valid = self.frame_ref_mask.reshape(bz, -1)[b])
                                    c_mpjpe_pa = compute_mpjpe_pa_first(curr_pred.detach().reshape(1, self.opt.max_motion_length, -1, 3),
                                                                self.gt_joints_frame_ref[b].detach().reshape(1, self.opt.max_motion_length, -1, 3),
                                                                valid = self.frame_ref_mask.reshape(bz, -1)[b])
                                    # c_mpjpe_pa = compute_mpjpe(curr_pred.detach().reshape(-1, self.opt.joints_num, 3),
                                    #                             self.gt_joints_frame_ref[b].detach().reshape(-1, self.opt.joints_num, 3),
                                    #                             valid = self.frame_ref_mask.reshape(bz,-1)[b])
                                    all_mpjpe_pas.append(c_mpjpe_pa.item())
                                except:
                                    all_mpjpe_pas.append(np.nan)
                        c_value = np.nanmin(all_mpjpe_pas)
                        if c_value is not np.nan:
                            logs['g_mpjpe_pa_t2h'] += c_value

                        # c_mpjpe_pa_ours = compute_mpjpe_pa(self.pred_joints_frame_ref.detach().reshape(bz, -1, 3),
                        #                                     self.gt_joints_frame_ref.detach().reshape(bz, -1, 3),
                        #                                     valid = self.frame_ref_mask)
                        c_mpjpe_pa_ours = compute_mpjpe_pa_first(self.pred_joints_frame_ref.detach().reshape(bz, self.opt.max_motion_length, -1, 3),
                                                            self.gt_joints_frame_ref.detach().reshape(bz, self.opt.max_motion_length, -1, 3),
                                                            valid = self.frame_ref_mask)
                        logs['g_mpjpe_pa_ours'] += c_mpjpe_pa_ours.item()

                    if self.opt.pred_cam:
                        # compute l1 error for cam_rot and cam_transl
                        l1_cam_rot = torch.abs(self.pred_cam_rot - self.gt_cam_rot)
                        l1_cam_rot = torch.nanmean(l1_cam_rot.reshape(-1, 6) * self.known_cam_mask.reshape(-1, 1))
                        logs['l1_cam_rot_ref'] += l1_cam_rot.item()
                        
                        if not self.opt.coord_sys == 'contact':
                            l1_cam_transl = torch.linalg.norm(self.pred_cam_transl - self.gt_cam_transl, dim=-1)
                            l1_cam_transl = torch.nanmean(l1_cam_transl.reshape(-1, 1) * self.known_cam_mask.reshape(-1, 1))
                            logs['l1_cam_transl_ref'] += l1_cam_transl.item()

                    if self.opt.coord_sys == 'contact':
                        curr_mpjpe_contact = compute_mpjpe(self.pred_motion_contact.detach().reshape(-1, self.opt.joints_num, 3), 
                                                           self.motions_contact.detach().reshape(-1, self.opt.joints_num, 3), 
                                                           valid = self.known_mask.reshape(-1))
                        logs['mpjpe_contact'] += curr_mpjpe_contact.item()
                        curr_mpjpe_ra_contact = compute_mpjpe_ra(self.pred_motion_contact.detach().reshape(-1, self.opt.joints_num, 3), 
                                                                 self.motions_contact.detach().reshape(-1, self.opt.joints_num, 3),
                                                                 valid = self.known_mask.reshape(-1))
                        logs['mpjpe_ra_contact'] += curr_mpjpe_ra_contact.item()
                        curr_mpjpe_pa_contact = compute_mpjpe_pa(self.pred_motion_contact.detach().reshape(-1, self.opt.joints_num, 3), 
                                                                 self.motions_contact.detach().reshape(-1, self.opt.joints_num, 3),
                                                                 valid = self.known_mask.reshape(-1),
                                                                 )
                        logs['mpjpe_pa_contact'] += curr_mpjpe_pa_contact
                        
                        # compute l1 error for cam_t_contact
                        # l1_cam_t_contact = torch.linalg.norm(self.pred_cam_t_contact - self.gt_cam_t_contact, dim=-1)
                        # l1_cam_t_contact = torch.nanmean(l1_cam_t_contact.reshape(-1, 1) * self.known_mask.reshape(-1, 1))
                        # l1_cam_t_contact = torch.nanmean(l1_cam_t_contact.reshape(-1, 1) * self.frame_ref_mask.reshape(-1, 1)) # masking is not correct
                        
                        # check this for correct masking
                        l1_cam_t_contact = torch.linalg.norm(self.pred_cam_t_contact - self.gt_cam_t_contact, dim=-1).flatten()
                        l1_cam_t_contact[self.frame_ref_mask==0] = float('nan')
                        l1_cam_t_contact = torch.nanmean(l1_cam_t_contact)

                        logs['l1_cam_t_contact'] += l1_cam_t_contact.item()

                        # compute l1 error for cam_t_contact_ref
                        # l1_cam_t_contact_ref = torch.linalg.norm(self.pred_cam_t_contact_ref - self.gt_cam_t_contact_ref, dim=-1)
                        # l1_cam_t_contact_ref = torch.nanmean(l1_cam_t_contact_ref.reshape(-1, 1) * self.known_mask_ref.reshape(-1, 1))
                        # l1_cam_t_contact_ref = torch.nanmean(l1_cam_t_contact_ref.reshape(-1, 1) * self.frame_ref_mask.reshape(-1, 1))
                        
                        # repeat for cam_t_contact_ref
                        l1_cam_t_contact_ref = torch.linalg.norm(self.pred_cam_t_contact_ref - self.gt_cam_t_contact_ref, dim=-1).flatten()
                        l1_cam_t_contact_ref[self.frame_ref_mask==0] = float('nan')
                        l1_cam_t_contact_ref = torch.nanmean(l1_cam_t_contact_ref)

                        logs['l1_cam_t_contact_ref'] += l1_cam_t_contact_ref.item()

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    # print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

                if it % self.opt.viz_every_t == 0:
                    num_viz = 4
                    data = torch.cat([self.motions[:num_viz], self.pred_motion[:num_viz]], dim=0).detach().cpu().numpy()
                    # np.save(pjoin(self.opt.eval_dir, 'E%04d.npy' % (epoch)), data)
                    save_dir = pjoin(self.opt.eval_dir, 'E%07d' % (it))
                    # add val prefix to names
                    names = [f'train_{n}' for n in batch_data['name'][:num_viz]]
                    names += names
                    os.makedirs(save_dir, exist_ok=True)
                    plot_eval(data, save_dir, names=names)

                if self.opt.debug: break

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            # epoch += 1
            # if epoch % self.opt.save_every_e == 0:
            #     self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            val_epoch = epoch if self.opt.debug else epoch + 1
            if val_epoch % self.opt.val_every_e == 0:
                print('Validation time:')
                self.vq_model.eval()
                val_loss_rec = []
                val_loss_vel = []
                val_loss_commit = []
                val_loss = []
                val_perpexity = []

                if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
                    val_mpjpe = []
                    val_mpjpe_ra = []
                    val_mpjpe_pa = []
                    val_mpjpe_pa_f = []
                    val_mpjpe_pa_g = []

                    val_mpjpe_ref = []
                    val_l1_cam_t = []

                    if self.opt.text2hoi:
                        val_g_mpjpe_pa_t2h = []
                        val_g_mpjpe_pa_ours = []

                    if self.opt.pred_cam:
                        val_l1_cam_rot_ref = []
                        val_l1_cam_transl_ref = []

                    if self.opt.coord_sys == 'contact':
                        val_mpjpe_contact = []
                        val_mpjpe_ra_contact = []
                        val_mpjpe_pa_contact = []

                        val_l1_cam_t_contact = []
                        val_l1_cam_t_contact_ref = []

                    if self.opt.joints_loss:
                        val_loss_joints = []
                    
                    if self.opt.contact_map:
                        val_loss_contact_map = []
                        val_contact_logits = []
                        val_contact_labels = []
                        val_contact_masks = []

                    if self.opt.diffusion:
                        val_loss_diff = []
                
                indices_out = {}
                with torch.no_grad():
                    for ival, batch_data in enumerate(tqdm(val_loader)):
                        loss, loss_rec, loss_vel, loss_commit, perplexity, more_loss = self.forward(batch_data, mode='val')

                        # move to cpu and detach
                        loss = loss.cpu().detach()
                        loss_rec = loss_rec.cpu().detach()
                        loss_vel = loss_vel.cpu().detach()
                        loss_commit = loss_commit.cpu().detach()
                        perplexity = perplexity.cpu().detach()
                        for k, v in more_loss.items():
                            if isinstance(v, torch.Tensor):
                                more_loss[k] = v.cpu().detach()

                        # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                        # val_loss_emb += self.embedding_loss.item()
                        val_loss.append(loss.item())
                        val_loss_rec.append(loss_rec.item())
                        val_loss_vel.append(loss_vel.item())
                        val_loss_commit.append(loss_commit.item())
                        val_perpexity.append(perplexity.item())
                        
                        if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
                            if self.opt.joints_loss:
                                loss_joints = more_loss['joints_ref']
                                # logs['loss_joints'] += loss_joints.item()
                                val_loss_joints.append(loss_joints.item())

                            if self.opt.contact_map:
                                loss_contact_map = more_loss['contact_map']
                                # logs['loss_contact_map'] += loss_contact_map.item()
                                val_loss_contact_map.append(loss_contact_map.item())

                                val_contact_logits.append(more_loss['logits'])
                                val_contact_labels.append(more_loss['labels'])
                                val_contact_masks.append(self.known_mask.reshape(-1, self.opt.max_motion_length, 1).cpu())

                            if self.opt.diffusion:
                                loss_diff = more_loss['diff_loss']
                                val_loss_diff.append(loss_diff.item())
                            
                            # ########## what is the right mask for metrics ##########
                            # self.known_cam_mask = self.frame_ref_mask
                            # ########################################################
                            
                            curr_mpjpe = compute_mpjpe(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                               self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                               valid = self.known_cam_mask.reshape(-1))
                            # logs['mpjpe'] += curr_mpjpe.item()
                            val_mpjpe.append(curr_mpjpe.item())
                            curr_mpjpe_ra = compute_mpjpe_ra(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                                            self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                                            valid = self.known_cam_mask.reshape(-1))
                            # logs['mpjpe_ra'] += curr_mpjpe_ra.item()
                            val_mpjpe_ra.append(curr_mpjpe_ra.item())
                            curr_mpjpe_pa = compute_mpjpe_pa(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                                            self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                                            valid = self.known_cam_mask.reshape(-1))
                            # logs['mpjpe_pa'] += curr_mpjpe_pa.item()
                            val_mpjpe_pa.append(curr_mpjpe_pa)
                            curr_mpjpe_ref = compute_mpjpe(self.pred_joints_frame_ref.detach().reshape(-1, self.opt.joints_num, 3), 
                                                   self.gt_joints_frame_ref.detach().reshape(-1, self.opt.joints_num, 3),
                                                   valid = self.frame_ref_mask.reshape(-1))
                            # logs['mpjpe_ref'] += curr_mpjpe_ref.item()
                            val_mpjpe_ref.append(curr_mpjpe_ref.item())
                            
                            l1_cam_t = torch.linalg.norm(self.pred_cam_t - self.gt_cam_t, dim=-1)
                            l1_cam_t = torch.nanmean(l1_cam_t.reshape(-1, 1) * self.known_cam_mask.reshape(-1, 1))
                            # logs['l1_cam_t'] += l1_cam_t.item()
                            val_l1_cam_t.append(l1_cam_t.item())

                            bz = self.pred_joints_frame_ref.shape[0]
                            # procrustes alignment at the global level
                            curr_mpjpe_pa_g = compute_mpjpe_pa(self.pred_joints_frame_ref.detach().reshape(bz, -1, 3),
                                                            self.gt_joints_frame_ref.detach().reshape(bz, -1, 3),
                                                            valid = self.frame_ref_mask.reshape(bz, -1))
                            # logs['mpjpe_pa_g'] += curr_mpjpe_pa_g.item()  
                            val_mpjpe_pa_g.append(curr_mpjpe_pa_g.item())

                            # procrustes alignment only at the first frame
                            curr_mpjpe_pa_f = compute_mpjpe_pa_first(self.pred_joints_frame_ref.detach().reshape(bz, self.opt.max_motion_length, -1, 3),
                                                            self.gt_joints_frame_ref.detach().reshape(bz, self.opt.max_motion_length, -1, 3),
                                                            valid = self.frame_ref_mask)
                            # logs['mpjpe_pa_f'] += curr_mpjpe_pa_f.item()
                            val_mpjpe_pa_f.append(curr_mpjpe_pa_f.item())

                            if self.opt.text2hoi:
                                bz, num_preds = self.t2h_motion.shape[:2]
                                all_mpjpe_pas = []
                                for b in range(bz):
                                    for j in range(num_preds):
                                        curr_pred = self.t2h_motion[b,j]
                                        # compute mpjpe for text2hoi predictions
                                        try:
                                            # c_mpjpe_pa = compute_mpjpe_pa(curr_pred.detach().reshape(1, -1, 3),
                                            #                             self.gt_joints_frame_ref[b].detach().reshape(1, -1, 3),
                                            #                             valid = self.frame_ref_mask.reshape(bz,-1)[b])
                                            c_mpjpe_pa = compute_mpjpe_pa_first(curr_pred.detach().reshape(1, self.opt.max_motion_length, -1, 3),
                                                                self.gt_joints_frame_ref[b].detach().reshape(1, self.opt.max_motion_length, -1, 3),
                                                                valid = self.frame_ref_mask.reshape(bz, -1)[b])
                                            # c_mpjpe_pa = compute_mpjpe(curr_pred.detach().reshape(-1, self.opt.joints_num, 3),
                                            #                             self.gt_joints_frame_ref[b].detach().reshape(-1, self.opt.joints_num, 3),
                                            #                             valid = self.frame_ref_mask.reshape(bz,-1)[b])
                                            all_mpjpe_pas.append(c_mpjpe_pa.item())
                                        except:
                                            all_mpjpe_pas.append(np.nan)
                                val_g_mpjpe_pa_t2h.append(np.nanmin(all_mpjpe_pas))

                                # c_mpjpe_pa_ours = compute_mpjpe_pa(self.pred_joints_frame_ref.detach().reshape(bz, -1, 3),
                                #                                 self.gt_joints_frame_ref.detach().reshape(bz, -1, 3),
                                #                                 valid = self.frame_ref_mask)
                                c_mpjpe_pa_ours = compute_mpjpe_pa_first(self.pred_joints_frame_ref.detach().reshape(bz, self.opt.max_motion_length, -1, 3),
                                                            self.gt_joints_frame_ref.detach().reshape(bz, self.opt.max_motion_length, -1, 3),
                                                            valid = self.frame_ref_mask)
                                val_g_mpjpe_pa_ours.append(c_mpjpe_pa_ours.item())

                            if self.opt.pred_cam:
                                # compute l1 error for cam_rot and cam_transl
                                l1_cam_rot = torch.abs(self.pred_cam_rot - self.gt_cam_rot)
                                l1_cam_rot = torch.nanmean(l1_cam_rot.reshape(-1, 6) * self.known_cam_mask.reshape(-1, 1))
                                # logs['l1_cam_rot_ref'] += l1_cam_rot.item()
                                val_l1_cam_rot_ref.append(l1_cam_rot.item())
                                
                                if not self.opt.coord_sys == 'contact':
                                    l1_cam_transl = torch.linalg.norm(self.pred_cam_transl - self.gt_cam_transl, dim=-1)
                                    l1_cam_transl = torch.nanmean(l1_cam_transl.reshape(-1, 1) * self.known_cam_mask.reshape(-1, 1))
                                    # logs['l1_cam_transl_ref'] += l1_cam_transl.item()
                                    val_l1_cam_transl_ref.append(l1_cam_transl.item())

                            if self.opt.coord_sys == 'contact':
                                curr_mpjpe_contact = compute_mpjpe(self.pred_motion_contact.reshape(-1, self.opt.joints_num, 3), 
                                                                   self.motions_contact.reshape(-1, self.opt.joints_num, 3),
                                                                   valid = self.known_mask.reshape(-1))
                                val_mpjpe_contact.append(curr_mpjpe_contact.item())
                                curr_mpjpe_ra_contact = compute_mpjpe_ra(self.pred_motion_contact.reshape(-1, self.opt.joints_num, 3), 
                                                                         self.motions_contact.reshape(-1, self.opt.joints_num, 3),
                                                                         valid = self.known_mask.reshape(-1))
                                val_mpjpe_ra_contact.append(curr_mpjpe_ra_contact.item())
                                curr_mpjpe_pa_contact = compute_mpjpe_pa(self.pred_motion_contact.reshape(-1, self.opt.joints_num, 3), 
                                                                         self.motions_contact.reshape(-1, self.opt.joints_num, 3),
                                                                         valid = self.known_mask.reshape(-1))
                                if not np.isnan(curr_mpjpe_pa_contact).any(): # check this
                                    val_mpjpe_pa_contact.append(curr_mpjpe_pa_contact)

                                # compute l1 error for cam_t_contact
                                # l1_cam_t_contact = torch.linalg.norm(self.pred_cam_t_contact - self.gt_cam_t_contact, dim=-1)
                                # l1_cam_t_contact = torch.nanmean(l1_cam_t_contact.reshape(-1, 3) * self.known_mask.reshape(-1, 1))
                                # l1_cam_t_contact = torch.nanmean(l1_cam_t_contact.reshape(-1, 1) * self.frame_ref_mask.reshape(-1, 1))
                                
                                # check this for correct masking
                                l1_cam_t_contact = torch.linalg.norm(self.pred_cam_t_contact - self.gt_cam_t_contact, dim=-1).flatten()
                                l1_cam_t_contact[self.frame_ref_mask==0] = float('nan')
                                l1_cam_t_contact = torch.nanmean(l1_cam_t_contact)

                                val_l1_cam_t_contact.append(l1_cam_t_contact.item())

                                # compute l1 error for cam_t_contact_ref
                                # l1_cam_t_contact_ref = torch.linalg.norm(self.pred_cam_t_contact_ref - self.gt_cam_t_contact_ref, dim=-1)
                                # l1_cam_t_contact_ref = torch.nanmean(l1_cam_t_contact_ref.reshape(-1, 3) * self.known_mask_ref.reshape(-1, 1))
                                # l1_cam_t_contact_ref = torch.nanmean(l1_cam_t_contact_ref.reshape(-1, 1) * self.frame_ref_mask.reshape(-1, 1))
                                
                                # repeat for cam_t_contact_ref
                                l1_cam_t_contact_ref = torch.linalg.norm(self.pred_cam_t_contact_ref - self.gt_cam_t_contact_ref, dim=-1).flatten()
                                l1_cam_t_contact_ref[self.frame_ref_mask==0] = float('nan')
                                l1_cam_t_contact_ref = torch.nanmean(l1_cam_t_contact_ref)

                                val_l1_cam_t_contact_ref.append(l1_cam_t_contact_ref.item())

                            if self.opt.return_indices:
                                names = batch_data['name']
                                ranges = batch_data['range']
                                indices = more_loss['code_idx']
                                starts = batch_data['start']
                                ends = batch_data['end']
                                bz = indices.shape[0]
                                for b in range(bz):
                                    curr_dict = {'range': (ranges[0][b], ranges[1][b]), 'indices': indices[b].reshape(-1).detach().cpu().numpy()}
                                    if names[b] not in indices_out:
                                        indices_out[names[b]] = {}
                                    indices_out[names[b]][f'{starts[b]}_{ends[b]}'] = curr_dict

                        if self.opt.debug and ival > 0:
                            break

                if self.opt.return_indices:
                    file_name = pjoin(self.opt.model_dir, 'indices_E%04d.pkl' % (val_epoch))
                    # save as pickle
                    with open(file_name, 'wb') as f:
                        pickle.dump(indices_out, f)

                # val_loss = val_loss_rec / (len(val_dataloader) + 1)
                # val_loss = val_loss / (len(val_dataloader) + 1)
                # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
                # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)
                self.logger.add_scalar('Val/loss', np.nanmean(val_loss), epoch)
                self.logger.add_scalar('Val/loss_rec', np.nanmean(val_loss_rec), epoch)
                self.logger.add_scalar('Val/loss_vel', np.nanmean(val_loss_vel), epoch)
                self.logger.add_scalar('Val/loss_commit', np.nanmean(val_loss_commit), epoch)
                self.logger.add_scalar('Val/loss_perplexity', np.nanmean(val_perpexity), epoch)

                if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
                    self.logger.add_scalar('Val/mpjpe', np.nanmean(val_mpjpe), epoch)
                    self.logger.add_scalar('Val/mpjpe_ra', np.nanmean(val_mpjpe_ra), epoch)
                    self.logger.add_scalar('Val/mpjpe_pa', np.nanmean(val_mpjpe_pa), epoch)
                    self.logger.add_scalar('Val/mpjpe_pa_f', np.nanmean(val_mpjpe_pa_f), epoch)
                    self.logger.add_scalar('Val/mpjpe_pa_g', np.nanmean(val_mpjpe_pa_g), epoch)

                    # self.logger.add_scalar('Val/mpjpe_ref', sum(val_mpjpe_ref) / len(val_mpjpe_ref), epoch)
                    self.logger.add_scalar('Val/mpjpe_ref', np.nanmean(val_mpjpe_ref), epoch)
                    self.logger.add_scalar('Val/l1_cam_t', np.nanmean(val_l1_cam_t), epoch)

                    if self.opt.text2hoi:
                        self.logger.add_scalar('Val/g_mpjpe_pa_t2h', np.nanmean(val_g_mpjpe_pa_t2h), epoch)
                        self.logger.add_scalar('Val/g_mpjpe_pa_ours', np.nanmean(val_g_mpjpe_pa_ours), epoch)

                    if self.opt.pred_cam:
                        self.logger.add_scalar('Val/l1_cam_rot_ref', np.nanmean(val_l1_cam_rot_ref), epoch)
                        if not self.opt.coord_sys == 'contact':
                            self.logger.add_scalar('Val/l1_cam_transl_ref', np.nanmean(val_l1_cam_transl_ref), epoch)

                    if self.opt.joints_loss:
                        self.logger.add_scalar('Val/loss_joints', np.nanmean(val_loss_joints), epoch)
                        
                    if self.opt.contact_map:
                        self.logger.add_scalar('Val/loss_contact_map', np.nanmean(val_loss_contact_map), epoch)

                        all_logits = torch.cat(val_contact_logits, dim=0)
                        all_labels = torch.cat(val_contact_labels, dim=0)
                        all_masks = torch.cat(val_contact_masks, dim=0)
                        # all_masks = all_masks.repeat(1, 1, all_logits.shape[-1])
                        precision, recall, f1 = binary_classification_metrics(all_logits, all_labels, all_masks)
                        self.logger.add_scalar('Val/contact_precision', precision, epoch)
                        self.logger.add_scalar('Val/contact_recall', recall, epoch)
                        self.logger.add_scalar('Val/contact_f1', f1, epoch)

                    if self.opt.diffusion:
                        self.logger.add_scalar('Val/loss_diff', np.nanmean(val_loss_diff), epoch)

                    if self.opt.coord_sys == 'contact':
                        self.logger.add_scalar('Val/mpjpe_contact', np.nanmean(val_mpjpe_contact), epoch)
                        self.logger.add_scalar('Val/mpjpe_ra_contact', np.nanmean(val_mpjpe_ra_contact), epoch)
                        self.logger.add_scalar('Val/mpjpe_pa_contact', np.nanmean(val_mpjpe_pa_contact), epoch)

                        self.logger.add_scalar('Val/l1_cam_t_contact', np.nanmean(val_l1_cam_t_contact), epoch)
                        self.logger.add_scalar('Val/l1_cam_t_contact_ref', np.nanmean(val_l1_cam_t_contact_ref), epoch)


                if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
                    # print('Validation Loss: %.5f Reconstruction: %.5f, Velocity: %.5f, Commit: %.5f, MPJPE: %.5f' %
                    #     (sum(val_loss)/len(val_loss), sum(val_loss_rec)/len(val_loss), 
                    #     sum(val_loss_vel)/len(val_loss), sum(val_loss_commit)/len(val_loss), sum(val_mpjpe)/len(val_mpjpe)))
                    print('Validation Loss: %.5f Reconstruction: %.5f, Velocity: %.5f, Commit: %.5f, MPJPE: %.5f' %
                        (np.nanmean(val_loss), np.nanmean(val_loss_rec), 
                        np.nanmean(val_loss_vel), np.nanmean(val_loss_commit), np.nanmean(val_mpjpe_ref)))
                else:
                    print('Validation Loss: %.5f Reconstruction: %.5f, Velocity: %.5f, Commit: %.5f' %
                        (sum(val_loss)/len(val_loss), sum(val_loss_rec)/len(val_loss), 
                        sum(val_loss_vel)/len(val_loss), sum(val_loss_commit)/len(val_loss)))
                
                # if val_epoch % self.opt.viz_every_e == 0:
                #     num_viz = 4
                #     data = torch.cat([self.motions[:num_viz], self.pred_motion[:num_viz]], dim=0).detach().cpu().numpy()
                #     # np.save(pjoin(self.opt.eval_dir, 'E%04d.npy' % (epoch)), data)
                #     save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                #     # add val prefix to names
                #     names = [f'val_{n}' for n in batch_data['name'][:num_viz]]
                #     names += names # 2 for gt and pred
                #     os.makedirs(save_dir, exist_ok=True)
                #     try:
                #         plot_eval(data, save_dir, names=names)
                #     except:
                #         pass

                curr_val_loss = np.nanmean(val_loss)
                if curr_val_loss is not np.nan and curr_val_loss < min_val_loss:
                    min_val_loss = curr_val_loss
                    # if sum(val_loss_vel) / len(val_loss_vel) < min_val_loss:
                    #     min_val_loss = sum(val_loss_vel) / len(val_loss_vel)
                    min_val_epoch = epoch
                    self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                    print('Best Validation Model So Far!~')

            # custom evaluation code for different settings of holoassist
            eval_epoch = epoch if self.opt.debug else epoch + 1
            if eval_epoch % self.opt.eval_every_e == 0 and eval_loader is not None:
                print('Evaluation time:')
                self.vq_model.eval()
                eval_loss_rec = []
                eval_loss_vel = []
                eval_loss_commit = []
                eval_loss = []
                eval_perpexity = []

                if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
                    eval_mpjpe = []
                    eval_mpjpe_ra = []
                    eval_mpjpe_pa = []
                    eval_mpjpe_pa_f = []
                    eval_mpjpe_pa_g = []

                    eval_mpjpe_ref = []
                    eval_l1_cam_t = []

                    if self.opt.text2hoi:
                        eval_g_mpjpe_pa_t2h = []
                        eval_g_mpjpe_pa_ours = []

                    if self.opt.pred_cam:
                        eval_l1_cam_rot_ref = []
                        eval_l1_cam_transl_ref = []

                    if self.opt.coord_sys == 'contact':
                        eval_mpjpe_contact = []
                        eval_mpjpe_ra_contact = []
                        eval_mpjpe_pa_contact = []

                        eval_l1_cam_t_contact = []
                        eval_l1_cam_t_contact_ref = []

                    if self.opt.joints_loss:
                        eval_loss_joints = []
                    
                    if self.opt.contact_map:
                        eval_loss_contact_map = []
                        eval_contact_logits = []
                        eval_contact_labels = []
                        eval_contact_masks = []

                    if self.opt.diffusion:
                        eval_loss_diff = []
                
                indices_out = {}
                with torch.no_grad():
                    for ieval, batch_data in enumerate(tqdm(eval_loader)):
                        loss, loss_rec, loss_vel, loss_commit, perplexity, more_loss = self.forward(batch_data, mode='eval')
                        
                        # move to cpu and detach
                        loss = loss.cpu().detach()
                        loss_rec = loss_rec.cpu().detach()
                        loss_vel = loss_vel.cpu().detach()
                        loss_commit = loss_commit.cpu().detach()
                        perplexity = perplexity.cpu().detach()
                        for k, v in more_loss.items():
                            if isinstance(v, torch.Tensor):
                                more_loss[k] = v.cpu().detach()
                        
                        # eval_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                        # eval_loss_emb += self.embedding_loss.item()
                        eval_loss.append(loss.item())
                        eval_loss_rec.append(loss_rec.item())
                        eval_loss_vel.append(loss_vel.item())
                        eval_loss_commit.append(loss_commit.item())
                        eval_perpexity.append(perplexity.item())
                        
                        if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
                            if self.opt.joints_loss:
                                loss_joints = more_loss['joints_ref']
                                # logs['loss_joints'] += loss_joints.item()
                                eval_loss_joints.append(loss_joints.item())

                            if self.opt.contact_map:
                                loss_contact_map = more_loss['contact_map']
                                # logs['loss_contact_map'] += loss_contact_map.item()
                                eval_loss_contact_map.append(loss_contact_map.item())

                                eval_contact_logits.append(more_loss['logits'])
                                eval_contact_labels.append(more_loss['labels'])
                                eval_contact_masks.append(self.known_mask.reshape(-1, self.opt.max_motion_length, 1).cpu())

                            if self.opt.diffusion:
                                loss_diff = more_loss['diff_loss']
                                eval_loss_diff.append(loss_diff.item())
                            
                            # ########## what is the right mask for metrics ##########
                            # self.known_cam_mask = self.frame_ref_mask
                            # ########################################################
                            
                            curr_mpjpe = compute_mpjpe(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                               self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                               valid = self.known_cam_mask.reshape(-1))
                            # logs['mpjpe'] += curr_mpjpe.item()
                            eval_mpjpe.append(curr_mpjpe.item())
                            curr_mpjpe_ra = compute_mpjpe_ra(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                                            self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                                            valid = self.known_cam_mask.reshape(-1))
                            # logs['mpjpe_ra'] += curr_mpjpe_ra.item()
                            eval_mpjpe_ra.append(curr_mpjpe_ra.item())
                            curr_mpjpe_pa = compute_mpjpe_pa(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                                            self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                                            valid = self.known_cam_mask.reshape(-1))
                            # logs['mpjpe_pa'] += curr_mpjpe_pa.item()
                            eval_mpjpe_pa.append(curr_mpjpe_pa)
                            curr_mpjpe_ref = compute_mpjpe(self.pred_joints_frame_ref.detach().reshape(-1, self.opt.joints_num, 3), 
                                                   self.gt_joints_frame_ref.detach().reshape(-1, self.opt.joints_num, 3),
                                                   valid = self.frame_ref_mask.reshape(-1))
                            # logs['mpjpe_ref'] += curr_mpjpe_ref.item()
                            eval_mpjpe_ref.append(curr_mpjpe_ref.item())
                            
                            l1_cam_t = torch.linalg.norm(self.pred_cam_t - self.gt_cam_t, dim=-1)
                            l1_cam_t = torch.nanmean(l1_cam_t.reshape(-1, 1) * self.known_cam_mask.reshape(-1, 1))
                            # logs['l1_cam_t'] += l1_cam_t.item()
                            eval_l1_cam_t.append(l1_cam_t.item())

                            bz = self.pred_joints_frame_ref.shape[0]
                            # procrustes alignment at the global level
                            curr_mpjpe_pa_g = compute_mpjpe_pa(self.pred_joints_frame_ref.detach().reshape(bz, -1, 3),
                                                               self.gt_joints_frame_ref.detach().reshape(bz, -1, 3),
                                                               valid=self.frame_ref_mask.reshape(bz, -1))
                            # logs['mpjpe_pa_g'] += curr_mpjpe_pa_g.item()
                            eval_mpjpe_pa_g.append(curr_mpjpe_pa_g.item())

                            # procrustes alignment only at the first frame
                            curr_mpjpe_pa_f = compute_mpjpe_pa_first(self.pred_joints_frame_ref.detach().reshape(bz, self.opt.max_motion_length, -1, 3),
                                                            self.gt_joints_frame_ref.detach().reshape(bz, self.opt.max_motion_length, -1, 3),
                                                            valid = self.frame_ref_mask)
                            # logs['mpjpe_pa_f'] += curr_mpjpe_pa_f.item()
                            eval_mpjpe_pa_f.append(curr_mpjpe_pa_f.item())

                            if self.opt.text2hoi:
                                bz, num_preds = self.t2h_motion.shape[:2]
                                all_mpjpe_pas = []
                                for b in range(bz):
                                    for j in range(num_preds):
                                        curr_pred = self.t2h_motion[b,j]
                                        # compute mpjpe for text2hoi predictions
                                        try:
                                            # c_mpjpe_pa = compute_mpjpe_pa(curr_pred.detach().reshape(1, -1, 3),
                                            #                             self.gt_joints_frame_ref[b].detach().reshape(1, -1, 3),
                                            #                             valid = self.frame_ref_mask.reshape(bz,-1)[b])
                                            c_mpjpe_pa = compute_mpjpe_pa_first(curr_pred.detach().reshape(1, self.opt.max_motion_length, -1, 3),
                                                                self.gt_joints_frame_ref[b].detach().reshape(1, self.opt.max_motion_length, -1, 3),
                                                                valid = self.frame_ref_mask.reshape(bz, -1)[b])
                                            # c_mpjpe_pa = compute_mpjpe(curr_pred.detach().reshape(-1, self.opt.joints_num, 3),
                                            #                             self.gt_joints_frame_ref[b].detach().reshape(-1, self.opt.joints_num, 3),
                                            #                             valid = self.frame_ref_mask.reshape(bz,-1)[b])
                                            all_mpjpe_pas.append(c_mpjpe_pa.item())
                                        except:
                                            all_mpjpe_pas.append(np.nan)
                                eval_g_mpjpe_pa_t2h.append(np.nanmin(all_mpjpe_pas))

                                # c_mpjpe_pa_ours = compute_mpjpe_pa(self.pred_joints_frame_ref.detach().reshape(bz, -1, 3),
                                #                                 self.gt_joints_frame_ref.detach().reshape(bz, -1, 3),
                                #                                 valid = self.frame_ref_mask)
                                c_mpjpe_pa_ours = compute_mpjpe_pa_first(self.pred_joints_frame_ref.detach().reshape(bz, self.opt.max_motion_length, -1, 3),
                                                            self.gt_joints_frame_ref.detach().reshape(bz, self.opt.max_motion_length, -1, 3),
                                                            valid = self.frame_ref_mask)
                                eval_g_mpjpe_pa_ours.append(c_mpjpe_pa_ours.item())

                            if self.opt.pred_cam:
                                # compute l1 error for cam_rot and cam_transl
                                l1_cam_rot = torch.abs(self.pred_cam_rot - self.gt_cam_rot)
                                l1_cam_rot = torch.nanmean(l1_cam_rot.reshape(-1, 6) * self.known_cam_mask.reshape(-1, 1))
                                # logs['l1_cam_rot_ref'] += l1_cam_rot.item()
                                eval_l1_cam_rot_ref.append(l1_cam_rot.item())
                                
                                if not self.opt.coord_sys == 'contact':
                                    l1_cam_transl = torch.linalg.norm(self.pred_cam_transl - self.gt_cam_transl, dim=-1)
                                    l1_cam_transl = torch.nanmean(l1_cam_transl.reshape(-1, 1) * self.known_cam_mask.reshape(-1, 1))
                                    # logs['l1_cam_transl_ref'] += l1_cam_transl.item()
                                    eval_l1_cam_transl_ref.append(l1_cam_transl.item())

                            if self.opt.coord_sys == 'contact':
                                curr_mpjpe_contact = compute_mpjpe(self.pred_motion_contact.reshape(-1, self.opt.joints_num, 3), 
                                                                   self.motions_contact.reshape(-1, self.opt.joints_num, 3),
                                                                   valid = self.known_mask.reshape(-1))
                                eval_mpjpe_contact.append(curr_mpjpe_contact.item())
                                curr_mpjpe_ra_contact = compute_mpjpe_ra(self.pred_motion_contact.reshape(-1, self.opt.joints_num, 3), 
                                                                         self.motions_contact.reshape(-1, self.opt.joints_num, 3),
                                                                         valid = self.known_mask.reshape(-1))
                                eval_mpjpe_ra_contact.append(curr_mpjpe_ra_contact.item())
                                curr_mpjpe_pa_contact = compute_mpjpe_pa(self.pred_motion_contact.reshape(-1, self.opt.joints_num, 3), 
                                                                         self.motions_contact.reshape(-1, self.opt.joints_num, 3),
                                                                         valid = self.known_mask.reshape(-1))
                                if not np.isnan(curr_mpjpe_pa_contact).any(): # check this
                                    eval_mpjpe_pa_contact.append(curr_mpjpe_pa_contact)

                                # compute l1 error for cam_t_contact
                                # l1_cam_t_contact = torch.linalg.norm(self.pred_cam_t_contact - self.gt_cam_t_contact, dim=-1)
                                # l1_cam_t_contact = torch.nanmean(l1_cam_t_contact.reshape(-1, 3) * self.known_mask.reshape(-1, 1))
                                # l1_cam_t_contact = torch.nanmean(l1_cam_t_contact.reshape(-1, 1) * self.frame_ref_mask.reshape(-1, 1))
                                
                                # check this for correct masking
                                l1_cam_t_contact = torch.linalg.norm(self.pred_cam_t_contact - self.gt_cam_t_contact, dim=-1).flatten()
                                l1_cam_t_contact[self.frame_ref_mask==0] = float('nan')
                                l1_cam_t_contact = torch.nanmean(l1_cam_t_contact)

                                eval_l1_cam_t_contact.append(l1_cam_t_contact.item())

                                # compute l1 error for cam_t_contact_ref
                                # l1_cam_t_contact_ref = torch.linalg.norm(self.pred_cam_t_contact_ref - self.gt_cam_t_contact_ref, dim=-1)
                                # l1_cam_t_contact_ref = torch.nanmean(l1_cam_t_contact_ref.reshape(-1, 3) * self.known_mask_ref.reshape(-1, 1))
                                # l1_cam_t_contact_ref = torch.nanmean(l1_cam_t_contact_ref.reshape(-1, 1) * self.frame_ref_mask.reshape(-1, 1))
                                
                                # repeat for cam_t_contact_ref
                                l1_cam_t_contact_ref = torch.linalg.norm(self.pred_cam_t_contact_ref - self.gt_cam_t_contact_ref, dim=-1).flatten()
                                l1_cam_t_contact_ref[self.frame_ref_mask==0] = float('nan')
                                l1_cam_t_contact_ref = torch.nanmean(l1_cam_t_contact_ref)

                                eval_l1_cam_t_contact_ref.append(l1_cam_t_contact_ref.item())

                            if self.opt.return_indices:
                                names = batch_data['name']
                                ranges = batch_data['range']
                                indices = more_loss['code_idx']
                                starts = batch_data['start']
                                ends = batch_data['end']
                                bz = indices.shape[0]
                                for b in range(bz):
                                    curr_dict = {'range': (ranges[0][b], ranges[1][b]), 'indices': indices[b].reshape(-1).detach().cpu().numpy()}
                                    if names[b] not in indices_out:
                                        indices_out[names[b]] = {}
                                    indices_out[names[b]][f'{starts[b]}_{ends[b]}'] = curr_dict

                        if self.opt.debug and ieval > 1:
                            break

                if self.opt.return_indices:
                    file_name = pjoin(self.opt.model_dir, 'indices_eval_E%04d.pkl' % (val_epoch))
                    # save as pickle
                    with open(file_name, 'wb') as f:
                        pickle.dump(indices_out, f)

                # eval_loss = eval_loss_rec / (len(eval_dataloader) + 1)
                # eval_loss = eval_loss / (len(eval_dataloader) + 1)
                # eval_loss_rec = eval_loss_rec / (len(eval_dataloader) + 1)
                # eval_loss_emb = eval_loss_emb / (len(eval_dataloader) + 1)
                self.logger.add_scalar('Eval/loss', np.nanmean(eval_loss), epoch)
                self.logger.add_scalar('Eval/loss_rec', np.nanmean(eval_loss_rec), epoch)
                self.logger.add_scalar('Eval/loss_vel', np.nanmean(eval_loss_vel), epoch)
                self.logger.add_scalar('Eval/loss_commit', np.nanmean(eval_loss_commit), epoch)
                self.logger.add_scalar('Eval/loss_perplexity', np.nanmean(eval_perpexity), epoch)

                if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
                    self.logger.add_scalar('Eval/mpjpe', np.nanmean(eval_mpjpe), epoch)
                    self.logger.add_scalar('Eval/mpjpe_ra', np.nanmean(eval_mpjpe_ra), epoch)
                    self.logger.add_scalar('Eval/mpjpe_pa', np.nanmean(eval_mpjpe_pa), epoch)
                    self.logger.add_scalar('Eval/mpjpe_pa_f', np.nanmean(eval_mpjpe_pa_f), epoch)
                    self.logger.add_scalar('Eval/mpjpe_pa_g', np.nanmean(eval_mpjpe_pa_g), epoch)

                    # self.logger.add_scalar('Eval/mpjpe_ref', sum(eval_mpjpe_ref) / len(eval_mpjpe_ref), epoch)
                    self.logger.add_scalar('Eval/mpjpe_ref', np.nanmean(eval_mpjpe_ref), epoch)
                    self.logger.add_scalar('Eval/l1_cam_t', np.nanmean(eval_l1_cam_t), epoch)

                    if self.opt.text2hoi:
                        self.logger.add_scalar('Eval/g_mpjpe_pa_t2h', np.nanmean(eval_g_mpjpe_pa_t2h), epoch)
                        self.logger.add_scalar('Eval/g_mpjpe_pa_ours', np.nanmean(eval_g_mpjpe_pa_ours), epoch)

                    if self.opt.pred_cam:
                        self.logger.add_scalar('Eval/l1_cam_rot_ref', np.nanmean(eval_l1_cam_rot_ref), epoch)
                        if not self.opt.coord_sys == 'contact':
                            self.logger.add_scalar('Eval/l1_cam_transl_ref', np.nanmean(eval_l1_cam_transl_ref), epoch)

                    if self.opt.joints_loss:
                        self.logger.add_scalar('Eval/loss_joints', np.nanmean(eval_loss_joints), epoch)
                        
                    if self.opt.contact_map:
                        self.logger.add_scalar('Eval/loss_contact_map', np.nanmean(eval_loss_contact_map), epoch)

                        all_logits = torch.cat(eval_contact_logits, dim=0)
                        all_labels = torch.cat(eval_contact_labels, dim=0)
                        all_masks = torch.cat(eval_contact_masks, dim=0)
                        # all_masks = all_masks.repeat(1, 1, all_logits.shape[-1])
                        precision, recall, f1 = binary_classification_metrics(all_logits, all_labels, all_masks)
                        self.logger.add_scalar('Eval/contact_precision', precision, epoch)
                        self.logger.add_scalar('Eval/contact_recall', recall, epoch)
                        self.logger.add_scalar('Eval/contact_f1', f1, epoch)

                    if self.opt.diffusion:
                        self.logger.add_scalar('Eval/loss_diff', np.nanmean(eval_loss_diff), epoch)

                    if self.opt.coord_sys == 'contact':
                        self.logger.add_scalar('Eval/mpjpe_contact', np.nanmean(eval_mpjpe_contact), epoch)
                        self.logger.add_scalar('Eval/mpjpe_ra_contact', np.nanmean(eval_mpjpe_ra_contact), epoch)
                        self.logger.add_scalar('Eval/mpjpe_pa_contact', np.nanmean(eval_mpjpe_pa_contact), epoch)

                        self.logger.add_scalar('Eval/l1_cam_t_contact', np.nanmean(eval_l1_cam_t_contact), epoch)
                        self.logger.add_scalar('Eval/l1_cam_t_contact_ref', np.nanmean(eval_l1_cam_t_contact_ref), epoch)


                if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
                    # print('Evaluation Loss: %.5f Reconstruction: %.5f, Velocity: %.5f, Commit: %.5f, MPJPE: %.5f' %
                    #     (sum(eval_loss)/len(eval_loss), sum(eval_loss_rec)/len(eval_loss), 
                    #     sum(eval_loss_vel)/len(eval_loss), sum(eval_loss_commit)/len(eval_loss), sum(eval_mpjpe)/len(eval_mpjpe)))
                    print('Evaluation Loss: %.5f Reconstruction: %.5f, Velocity: %.5f, Commit: %.5f, MPJPE: %.5f' %
                        (np.nanmean(eval_loss), np.nanmean(eval_loss_rec), 
                        np.nanmean(eval_loss_vel), np.nanmean(eval_loss_commit), np.nanmean(eval_mpjpe_ref)))
                else:
                    print('Evaluation Loss: %.5f Reconstruction: %.5f, Velocity: %.5f, Commit: %.5f' %
                        (sum(eval_loss)/len(eval_loss), sum(eval_loss_rec)/len(eval_loss), 
                        sum(eval_loss_vel)/len(eval_loss), sum(eval_loss_commit)/len(eval_loss)))

                # if eval_wrapper is not None: # validation is done on the test in the original code
                #     best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_vqvae(
                #         self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=best_fid,
                #         best_div=best_div, best_top1=best_top1,
                #         best_top2=best_top2, best_top3=best_top3, best_matching=best_matching, eval_wrapper=eval_wrapper)


                #     if val_epoch % self.opt.eval_every_e == 0:
                #         data = torch.cat([self.motions[:4], self.pred_motion[:4]], dim=0).detach().cpu().numpy()
                #         # np.save(pjoin(self.opt.eval_dir, 'E%04d.npy' % (epoch)), data)
                #         save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                #         os.makedirs(save_dir, exist_ok=True)
                #         plot_eval(data, save_dir)
                #         # if plot_eval is not None:
                #         #     save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                #         #     os.makedirs(save_dir, exist_ok=True)
                #         #     plot_eval(data, save_dir)

                #     # if epoch - min_val_epoch >= self.opt.early_stop_e:
                #     #     print('Early Stopping!~')

    
    def inference(self, val_loader, ckpt='latest'):
        self.vq_model.to(self.device)

        model_dir = pjoin(self.opt.model_dir, f'{ckpt}.tar')
        if self.opt.transfer_from is not None:
            splits = self.opt.transfer_from.split('/')
            transfer_dataset, transfer_model = splits[-2], splits[-1]
            model_dir = model_dir.replace(f'/{self.opt.dataset_name}/', f'/{transfer_dataset}/')
            model_dir = model_dir.replace(f'/{self.opt.name}/', f'/{transfer_model}/')
        assert os.path.exists(model_dir), f'Model {model_dir} does not exist!'
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        epoch, it = checkpoint['ep'], checkpoint['total_it']
        print (f'Loaded model from {model_dir} at epoch {epoch} and iteration {it}')

        if 'mano' in self.opt.motion_type:
            self.mano_mean = torch.from_numpy(val_loader.dataset.mean).to(self.device).float()
            self.mano_std = torch.from_numpy(val_loader.dataset.std).to(self.device).float()

        print('Running inference:')
        self.vq_model.eval()
        # val_loss_rec = []
        # val_loss_vel = []
        # val_loss_commit = []
        # val_loss = []
        # val_perpexity = []

        if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
            val_mpjpe = []
            val_mpjpe_ra = []
            val_mpjpe_pa = []

            val_mpjpe_ref = []
            val_l1_cam_t = []

            if self.opt.pred_cam:
                val_l1_cam_rot_ref = []
                val_l1_cam_transl_ref = []

            if self.opt.coord_sys == 'contact':
                val_mpjpe_contact = []
                val_mpjpe_ra_contact = []
                val_mpjpe_pa_contact = []

                val_l1_cam_t_contact = []
                val_l1_cam_t_contact_ref = []

            if self.opt.joints_loss:
                val_loss_joints = []
            
            if self.opt.contact_map:
                val_loss_contact_map = []
                val_contact_logits = []
                val_contact_labels = []
                val_contact_masks = []
        
        indices_out = {}
        with torch.no_grad():
            for ival, batch_data in enumerate(tqdm(val_loader)):
                loss, loss_rec, loss_vel, loss_commit, perplexity, more_loss = self.forward(batch_data, mode='val')
                
                # # compute all losses and metrics
                # # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                # # val_loss_emb += self.embedding_loss.item()
                # val_loss.append(loss.item())
                # val_loss_rec.append(loss_rec.item())
                # val_loss_vel.append(loss_vel.item())
                # val_loss_commit.append(loss_commit.item())
                # val_perpexity.append(perplexity.item())
                
                if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
                #     if self.opt.joints_loss:
                #         loss_joints = more_loss['joints_ref']
                #         # logs['loss_joints'] += loss_joints.item()
                #         val_loss_joints.append(loss_joints.item())

                #     if self.opt.contact_map:
                #         loss_contact_map = more_loss['contact_map']
                #         # logs['loss_contact_map'] += loss_contact_map.item()
                #         val_loss_contact_map.append(loss_contact_map.item())

                #         val_contact_logits.append(more_loss['logits'])
                #         val_contact_labels.append(more_loss['labels'])
                #         val_contact_masks.append(self.known_mask.reshape(-1, self.opt.max_motion_length, 1))
                    
                #     # ########## what is the right mask for metrics ##########
                #     # self.known_cam_mask = self.frame_ref_mask
                #     # ########################################################
                    
                #     curr_mpjpe = compute_mpjpe(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                #                         self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                #                         valid = self.known_cam_mask.reshape(-1))
                #     # logs['mpjpe'] += curr_mpjpe.item()
                #     val_mpjpe.append(curr_mpjpe.item())
                #     curr_mpjpe_ra = compute_mpjpe_ra(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                #                                     self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                #                                     valid = self.known_cam_mask.reshape(-1))
                #     # logs['mpjpe_ra'] += curr_mpjpe_ra.item()
                #     val_mpjpe_ra.append(curr_mpjpe_ra.item())
                #     curr_mpjpe_pa = compute_mpjpe_pa(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                #                                     self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                #                                     valid = self.known_cam_mask.reshape(-1))
                #     # logs['mpjpe_pa'] += curr_mpjpe_pa.item()
                #     val_mpjpe_pa.append(curr_mpjpe_pa)
                #     curr_mpjpe_ref = compute_mpjpe(self.pred_joints_frame_ref.detach().reshape(-1, self.opt.joints_num, 3), 
                #                             self.gt_joints_frame_ref.detach().reshape(-1, self.opt.joints_num, 3),
                #                             valid = self.frame_ref_mask.reshape(-1))
                #     # logs['mpjpe_ref'] += curr_mpjpe_ref.item()
                #     val_mpjpe_ref.append(curr_mpjpe_ref.item())
                    
                #     l1_cam_t = torch.linalg.norm(self.pred_cam_t - self.gt_cam_t, dim=-1)
                #     l1_cam_t = torch.nanmean(l1_cam_t.reshape(-1, 1) * self.known_cam_mask.reshape(-1, 1))
                #     # logs['l1_cam_t'] += l1_cam_t.item()
                #     val_l1_cam_t.append(l1_cam_t.item())

                #     if self.opt.pred_cam:
                #         # compute l1 error for cam_rot and cam_transl
                #         l1_cam_rot = torch.abs(self.pred_cam_rot - self.gt_cam_rot)
                #         l1_cam_rot = torch.nanmean(l1_cam_rot.reshape(-1, 6) * self.known_cam_mask.reshape(-1, 1))
                #         # logs['l1_cam_rot_ref'] += l1_cam_rot.item()
                #         val_l1_cam_rot_ref.append(l1_cam_rot.item())
                        
                #         if not self.opt.coord_sys == 'contact':
                #             l1_cam_transl = torch.linalg.norm(self.pred_cam_transl - self.gt_cam_transl, dim=-1)
                #             l1_cam_transl = torch.nanmean(l1_cam_transl.reshape(-1, 1) * self.known_cam_mask.reshape(-1, 1))
                #             # logs['l1_cam_transl_ref'] += l1_cam_transl.item()
                #             val_l1_cam_transl_ref.append(l1_cam_transl.item())

                #     if self.opt.coord_sys == 'contact':
                #         curr_mpjpe_contact = compute_mpjpe(self.pred_motion_contact.reshape(-1, self.opt.joints_num, 3), 
                #                                             self.motions_contact.reshape(-1, self.opt.joints_num, 3),
                #                                             valid = self.known_mask.reshape(-1))
                #         val_mpjpe_contact.append(curr_mpjpe_contact.item())
                #         curr_mpjpe_ra_contact = compute_mpjpe_ra(self.pred_motion_contact.reshape(-1, self.opt.joints_num, 3), 
                #                                                     self.motions_contact.reshape(-1, self.opt.joints_num, 3),
                #                                                     valid = self.known_mask.reshape(-1))
                #         val_mpjpe_ra_contact.append(curr_mpjpe_ra_contact.item())
                #         curr_mpjpe_pa_contact = compute_mpjpe_pa(self.pred_motion_contact.reshape(-1, self.opt.joints_num, 3), 
                #                                                     self.motions_contact.reshape(-1, self.opt.joints_num, 3),
                #                                                     valid = self.known_mask.reshape(-1))
                #         if not np.isnan(curr_mpjpe_pa_contact).any(): # check this
                #             val_mpjpe_pa_contact.append(curr_mpjpe_pa_contact)

                #         # compute l1 error for cam_t_contact
                #         # l1_cam_t_contact = torch.linalg.norm(self.pred_cam_t_contact - self.gt_cam_t_contact, dim=-1)
                #         # l1_cam_t_contact = torch.nanmean(l1_cam_t_contact.reshape(-1, 3) * self.known_mask.reshape(-1, 1))
                #         # l1_cam_t_contact = torch.nanmean(l1_cam_t_contact.reshape(-1, 1) * self.frame_ref_mask.reshape(-1, 1))
                        
                #         # check this for correct masking
                #         l1_cam_t_contact = torch.linalg.norm(self.pred_cam_t_contact - self.gt_cam_t_contact, dim=-1).flatten()
                #         l1_cam_t_contact[self.frame_ref_mask==0] = float('nan')
                #         l1_cam_t_contact = torch.nanmean(l1_cam_t_contact)

                #         val_l1_cam_t_contact.append(l1_cam_t_contact.item())

                #         # compute l1 error for cam_t_contact_ref
                #         # l1_cam_t_contact_ref = torch.linalg.norm(self.pred_cam_t_contact_ref - self.gt_cam_t_contact_ref, dim=-1)
                #         # l1_cam_t_contact_ref = torch.nanmean(l1_cam_t_contact_ref.reshape(-1, 3) * self.known_mask_ref.reshape(-1, 1))
                #         # l1_cam_t_contact_ref = torch.nanmean(l1_cam_t_contact_ref.reshape(-1, 1) * self.frame_ref_mask.reshape(-1, 1))
                        
                #         # repeat for cam_t_contact_ref
                #         l1_cam_t_contact_ref = torch.linalg.norm(self.pred_cam_t_contact_ref - self.gt_cam_t_contact_ref, dim=-1).flatten()
                #         l1_cam_t_contact_ref[self.frame_ref_mask==0] = float('nan')
                #         l1_cam_t_contact_ref = torch.nanmean(l1_cam_t_contact_ref)

                #         val_l1_cam_t_contact_ref.append(l1_cam_t_contact_ref.item())

                    if self.opt.return_indices:
                        names = batch_data['name']
                        ranges = batch_data['range']
                        indices = more_loss['code_idx']
                        starts = batch_data['start']
                        ends = batch_data['end']
                        bz = indices.shape[0]
                        for b in range(bz):
                            curr_dict = {'range': (ranges[0][b], ranges[1][b]), 'indices': indices[b].reshape(-1).detach().cpu().numpy()}
                            if names[b] not in indices_out:
                                indices_out[names[b]] = {}
                            indices_out[names[b]][f'{starts[b]}_{ends[b]}'] = curr_dict

                if self.opt.debug and ival > 0:
                    break

        if self.opt.return_indices:
            # file_name = pjoin(self.opt.model_dir, f'indices_E{epoch:04d}_{self.opt.setting}_{val_loader.dataset.split}.pkl')
            prefix = ''
            sett = '' if self.opt.setting is None else self.opt.setting
            if self.opt.transfer_from is not None:
                prefix = f'transfer_{transfer_dataset}_{transfer_model}_'
                # sett = ''
            file_name = pjoin(self.opt.model_dir, f'{prefix}indices_finest_{sett}_{val_loader.dataset.split}.pkl')
            # save as pickle
            with open(file_name, 'wb') as f:
                pickle.dump(indices_out, f)


class LengthEstTrainer(object):

    def __init__(self, args, estimator, text_encoder, encode_fnc):
        self.opt = args
        self.estimator = estimator
        self.text_encoder = text_encoder
        self.encode_fnc = encode_fnc
        self.device = args.device

        if args.is_train:
            # self.motion_dis
            self.logger = SummaryWriter(args.log_dir)
            self.mul_cls_criterion = torch.nn.CrossEntropyLoss()

    def resume(self, model_dir):
        checkpoints = torch.load(model_dir, map_location=self.device)
        self.estimator.load_state_dict(checkpoints['estimator'])
        # self.opt_estimator.load_state_dict(checkpoints['opt_estimator'])
        return checkpoints['epoch'], checkpoints['iter']

    def save(self, model_dir, epoch, niter):
        state = {
            'estimator': self.estimator.state_dict(),
            # 'opt_estimator': self.opt_estimator.state_dict(),
            'epoch': epoch,
            'niter': niter,
        }
        torch.save(state, model_dir)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def train(self, train_dataloader, val_dataloader):
        self.estimator.to(self.device)
        self.text_encoder.to(self.device)

        self.opt_estimator = optim.Adam(self.estimator.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            if not os.path.exists(model_dir):
                print ('No model found, training from scratch')
            else:
                epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        min_val_loss = np.inf
        logs = defaultdict(float)
        while epoch < self.opt.max_epoch:
            # time0 = time.time()
            for i, batch_data in enumerate(train_dataloader):
                self.estimator.train()

                conds, _, m_lens = batch_data
                # word_emb = word_emb.detach().to(self.device).float()
                # pos_ohot = pos_ohot.detach().to(self.device).float()
                # m_lens = m_lens.to(self.device).long()
                text_embs = self.encode_fnc(self.text_encoder, conds, self.opt.device).detach()
                # print(text_embs.shape, text_embs.device)

                pred_dis = self.estimator(text_embs)

                self.zero_grad([self.opt_estimator])

                gt_labels = m_lens // self.opt.unit_length
                gt_labels = gt_labels.long().to(self.device)
                # print(gt_labels.shape, pred_dis.shape)
                # print(gt_labels.max(), gt_labels.min())
                # print(pred_dis)
                acc = (gt_labels == pred_dis.argmax(dim=-1)).sum() / len(gt_labels)
                loss = self.mul_cls_criterion(pred_dis, gt_labels)

                loss.backward()

                self.clip_norm([self.estimator])
                self.step([self.opt_estimator])

                logs['loss'] += loss.item()
                logs['acc'] += acc.item()

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    # self.logger.add_scalar('Val/loss', val_loss, it)

                    for tag, value in logs.items():
                        self.logger.add_scalar("Train/%s"%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(float)
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                    if it % self.opt.save_latest == 0:
                        self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1

            print('Validation time:')

            val_loss = 0
            val_acc = 0
            # self.estimator.eval()
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.estimator.eval()

                    conds, _, m_lens = batch_data
                    # word_emb = word_emb.detach().to(self.device).float()
                    # pos_ohot = pos_ohot.detach().to(self.device).float()
                    # m_lens = m_lens.to(self.device).long()
                    text_embs = self.encode_fnc(self.text_encoder, conds, self.opt.device)
                    pred_dis = self.estimator(text_embs)

                    gt_labels = m_lens // self.opt.unit_length
                    gt_labels = gt_labels.long().to(self.device)
                    loss = self.mul_cls_criterion(pred_dis, gt_labels)
                    acc = (gt_labels == pred_dis.argmax(dim=-1)).sum() / len(gt_labels)

                    val_loss += loss.item()
                    val_acc += acc.item()


            val_loss = val_loss / len(val_dataloader)
            val_acc = val_acc / len(val_dataloader)
            print('Validation Loss: %.5f Validation Acc: %.5f' % (val_loss, val_acc))

            if val_loss < min_val_loss:
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                min_val_loss = val_loss
