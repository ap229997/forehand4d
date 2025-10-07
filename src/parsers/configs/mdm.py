from src.parsers.configs.generic import DEFAULT_ARGS_EGO

DEFAULT_ARGS_EGO["batch_size"] = 16
DEFAULT_ARGS_EGO["test_batch_size"] = 16
DEFAULT_ARGS_EGO["num_workers"] = 8
DEFAULT_ARGS_EGO["img_res"] = 224 # image resolution input to the model
DEFAULT_ARGS_EGO["logger"] = 'tensorboard'
DEFAULT_ARGS_EGO['vis_every'] = 1000
DEFAULT_ARGS_EGO['log_every'] = 1
DEFAULT_ARGS_EGO['flip_prob'] = 0.0
DEFAULT_ARGS_EGO["lr"] = 1e-5
DEFAULT_ARGS_EGO["lr_dec_factor"] = 0
DEFAULT_ARGS_EGO["lr_decay"] = 0

# motion dataset params
DEFAULT_ARGS_EGO['dataset'] = 'arctic_ego' # multidataset supported, e.g. arctic_ego+h2o+...
DEFAULT_ARGS_EGO['val_dataset'] = 'arctic_ego'
# DEFAULT_ARGS_EGO['eval_every_epoch'] = 1 # use 1 when using assembly + holo for finetuning
# DEFAULT_ARGS_EGO['finetune_2d'] = 1 # use for finetuning MANO labels using 2D repojection loss
# DEFAULT_ARGS_EGO['finetune_3d'] = 1 # use for finetuning MANO labels using 3D keypoint loss
# DEFAULT_ARGS_EGO['grad_clip'] = 1.0 # 150.0 is default, reduce this during finetuning to reduce effect of noisy labels
DEFAULT_ARGS_EGO['trainsplit'] = 'train'
DEFAULT_ARGS_EGO['valsplit'] = 'val'
DEFAULT_ARGS_EGO['rot_hot3d_cam'] = True # hot3d camera's is tilted 90 degree counter-clockwise, so we rotate it to align with the camera in other datasets
# DEFAULT_ARGS_EGO['rot_factor'] = 30.0 # rotation augmentation during training, use this only for 2D traj -> MANO params model
# DEFAULT_ARGS_EGO['noise_factor'] = 0.4 # pixel level noise
# DEFAULT_ARGS_EGO['scale_factor'] = 0.25 # scale augmentation during training
DEFAULT_ARGS_EGO["window_size"] = 1 # history frames, multi-timestep suuported for some datasets, TODO: check for all datasets
DEFAULT_ARGS_EGO["rot_rep"] = 'rot6d' # rot6d, axis_angle, rotmat
# DEFAULT_ARGS_EGO["use_fixed_length"] = True # use fixed length for training
DEFAULT_ARGS_EGO["max_motion_length"] = 256 # prediction horizon
DEFAULT_ARGS_EGO["relative_kp3d"] = True # use root-relative 3D keypoint loss, root position is captured by translation loss
DEFAULT_ARGS_EGO["frame_of_ref"] = 'view' # 'view', 'mano', 'residual'
DEFAULT_ARGS_EGO["normalize_transl"] = True # normalize translation component, useful during training
# DEFAULT_ARGS_EGO['interpolate'] = True # goal image setting

# diffusion params
DEFAULT_ARGS_EGO['noise_schedule'] = 'cosine'
DEFAULT_ARGS_EGO['diffusion_steps'] = 1000
DEFAULT_ARGS_EGO['sigma_small'] = True
DEFAULT_ARGS_EGO['arch'] = 'trans_enc' # 'trans_enc', 'trans_dec', 'gru', 'trans_enc_local', 'trans_dec_local'
DEFAULT_ARGS_EGO['emb_trans_dec'] = False
DEFAULT_ARGS_EGO['tf_dropout'] = 0.1
DEFAULT_ARGS_EGO['layers'] = 16
DEFAULT_ARGS_EGO["latent_dim"] = 1024
DEFAULT_ARGS_EGO["cond_mode"] = 'spatial' # 'no_cond', 'pose', 'img', 'spatial', 'pose+img', this is for forecasting model
# DEFAULT_ARGS_EGO["cond_mode"] = 'future_j2d+future_cam+future_plucker+future_kpe' # 'future_j2d', 'future_cam', 'future_plucker', 'future_kpe' or combinations with + (e.g. 'future_j2d+future_cam+future_plucker+future_kpe'), this is for lifting model
DEFAULT_ARGS_EGO["kpe_freq"] = 4 # only used with future_kpe camera conditioning
DEFAULT_ARGS_EGO['img_feat'] = 'vit'
DEFAULT_ARGS_EGO['use_pretrained_feats'] = True # this helps
DEFAULT_ARGS_EGO['freeze_pretrained_feats'] = False
DEFAULT_ARGS_EGO['cond_mask_prob'] = 0.1
DEFAULT_ARGS_EGO["lambda_rcxyz"] = 0.0 # taken from MDM
DEFAULT_ARGS_EGO['lambda_vel'] = 0.0 # taken from MDM
DEFAULT_ARGS_EGO['lambda_fc'] = 0.0 # taken from MDM
# DEFAULT_ARGS_EGO['unconstrained'] = True # taken from MDM

# inference params
DEFAULT_ARGS_EGO['inference'] = True
DEFAULT_ARGS_EGO['augment_length'] = False # False during inference for consistency across runs, default is True for training
# DEFAULT_ARGS_EGO['start_num'] = 0.50 # start index for running inference on a subset
# DEFAULT_ARGS_EGO['end_num'] = 1.0 # end index for running inference on a subset
# DEFAULT_ARGS_EGO['infer_split'] = 'val' # 'train' or 'val'
# DEFAULT_ARGS_EGO['refine_iters'] = 1000 # for refining lifted MANO labels using 2D/3D keypoints
# DEFAULT_ARGS_EGO['loss_thres'] = 1e-4 # stopping criteria for refinement stage
# DEFAULT_ARGS_EGO['lr_mano'] = 1e-2 # learning rate for refinement stage
DEFAULT_ARGS_EGO['return_metrics'] = True # return metrics as well during eval/inference
# DEFAULT_ARGS_EGO['save_pred'] = 1 # index of iterative optimization, 0 is base dataset, i.e., no optimization, starts from 1
