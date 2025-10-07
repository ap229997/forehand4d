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

# inference params
DEFAULT_ARGS_EGO['inference'] = True
DEFAULT_ARGS_EGO['augment_length'] = False # False during inference for consistency across runs, default is True for training
# DEFAULT_ARGS_EGO['start_num'] = 0.50 # start index for running inference on a subset
# DEFAULT_ARGS_EGO['end_num'] = 1.0 # end index for running inference on a subset
# DEFAULT_ARGS_EGO['infer_split'] = 'val' # 'train' or 'val'
# DEFAULT_ARGS_EGO['refine_iters'] = 1000 # for refining lifted MANO labels using 2D/3D keypoints
# DEFAULT_ARGS_EGO['loss_thres'] = 1e-4 # stopping criteria for refinement stage
# DEFAULT_ARGS_EGO['lr_mano'] = 1e-2 # learning rate for refinement stage
DEFAULT_ARGS_EGO['return_metrics'] = True # return metrics as well during inference
# DEFAULT_ARGS_EGO['save_pred'] = 1 # index of iterative optimization, 0 is base dataset, i.e., no optimization, starts from 1

# LatentAct args
# vqvae params
DEFAULT_ARGS_EGO["code_dim"] = 512
DEFAULT_ARGS_EGO["nb_code"] = 512
DEFAULT_ARGS_EGO["mu"] = 0.99
DEFAULT_ARGS_EGO["down_t"] = 2
DEFAULT_ARGS_EGO["stride_t"] = 2
DEFAULT_ARGS_EGO["width"] = 512
DEFAULT_ARGS_EGO["depth"] = 3
DEFAULT_ARGS_EGO["dilation_growth_rate"] = 3
DEFAULT_ARGS_EGO["output_emb_width"] = 512
DEFAULT_ARGS_EGO["vq_act"] = "relu"
DEFAULT_ARGS_EGO["vq_norm"] = None
DEFAULT_ARGS_EGO["num_quantizers"] = 6
DEFAULT_ARGS_EGO["shared_codebook"] = False
DEFAULT_ARGS_EGO["quantize_dropout_prob"] = 0.2
DEFAULT_ARGS_EGO["sample_codebook_temp"] = 0.5
DEFAULT_ARGS_EGO["ext"] = 'default'
DEFAULT_ARGS_EGO["vqvae_tfenc_nhead"] = 4 # og: 1
DEFAULT_ARGS_EGO["vqvae_tfenc_nlayer"] = 4 # og: 1
DEFAULT_ARGS_EGO["vqvae_tfenc_dp"] = 0.0 # og: 0.0
DEFAULT_ARGS_EGO["vqvae_tfdec_nhead"] = 4 # og: 1
DEFAULT_ARGS_EGO["vqvae_tfdec_nlayer"] = 4 # og: 1
DEFAULT_ARGS_EGO["vqvae_tfdec_dp"] = 0.0 # og: 0.2
# model args
DEFAULT_ARGS_EGO["motion_type"] = "mano"
DEFAULT_ARGS_EGO["model_type"] = "tf"
DEFAULT_ARGS_EGO["n_freq"] = 4
DEFAULT_ARGS_EGO["video_feats"] = None
DEFAULT_ARGS_EGO["text_feats"] = False
DEFAULT_ARGS_EGO["contact_grid"] = None
DEFAULT_ARGS_EGO["contact_dim"] = None # og: 16, contacts not available
DEFAULT_ARGS_EGO["coord_sys"] = None
DEFAULT_ARGS_EGO["pred_cam"] = False
DEFAULT_ARGS_EGO["joints_loss"] = False
DEFAULT_ARGS_EGO["contact_map"] = False
DEFAULT_ARGS_EGO["residual_transf"] = False
DEFAULT_ARGS_EGO["return_indices"] = False
DEFAULT_ARGS_EGO["only_first"] = False
DEFAULT_ARGS_EGO["decoder_only"] = True
DEFAULT_ARGS_EGO["use_inpaint"] = False
DEFAULT_ARGS_EGO["interpolate"] = False
DEFAULT_ARGS_EGO["traj_only"] = False
DEFAULT_ARGS_EGO["stochastic"] = False
# diffusion args in LatentAct
DEFAULT_ARGS_EGO["diffusion"] = False
DEFAULT_ARGS_EGO["joint_vqvae"] = False
DEFAULT_ARGS_EGO["latent"] = False
DEFAULT_ARGS_EGO["noise_schedule"] = "cosine"
DEFAULT_ARGS_EGO["diffusion_steps"] = 1000
DEFAULT_ARGS_EGO["sigma_small"] = True
DEFAULT_ARGS_EGO["arch"] = "trans_enc"
DEFAULT_ARGS_EGO["emb_trans_dec"] = False
DEFAULT_ARGS_EGO["layers"] = 8
DEFAULT_ARGS_EGO["latent_dim"] = 512
DEFAULT_ARGS_EGO["cond_mask_prob"] = 0.1
DEFAULT_ARGS_EGO["lambda_rcxyz"] = 0.0
DEFAULT_ARGS_EGO["lambda_vel"] = 0.0
DEFAULT_ARGS_EGO["lambda_fc"] = 0.0
DEFAULT_ARGS_EGO["unconstrained"] = False
# feedforward params
DEFAULT_ARGS_EGO["feedforward"] = False
DEFAULT_ARGS_EGO["use_vit"] = False
# cross-dataset transfer params
DEFAULT_ARGS_EGO["transfer_from"] = None
DEFAULT_ARGS_EGO["eval_model"] = None
