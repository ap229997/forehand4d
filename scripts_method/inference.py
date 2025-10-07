import os
import sys
sys.path.append(".")

import numpy as np
from tqdm import tqdm
import torch
from loguru import logger

import src.factory as factory
from common.torch_utils import reset_all_seeds
from src.utils.const import args


# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
torch.set_float32_matmul_precision('medium')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.autograd.set_detect_anomaly(True)

def to_device(data, device):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device)
    return data

def main(args):
    reset_all_seeds(args.seed)
    torch.set_num_threads(args.num_threads)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = factory.fetch_model(args).to(device)
    
    if args.load_ckpt != "":
        ckpt = torch.load(args.load_ckpt)
        wrapper.load_state_dict(ckpt["state_dict"])
        logger.info(f"Loaded weights from {args.load_ckpt}")
    wrapper.eval()

    infer_split = args.get('infer_split', 'val')
    is_train = infer_split == 'train'
    dataset = factory.fetch_dataset(args, is_train=is_train)
    data_loader = factory.DataLoader(
            dataset=dataset,
            batch_size=args.test_batch_size,
            shuffle=False, # keep this False during inference
            num_workers=args.num_workers,
            collate_fn=factory.collate_stack_fn,
        )
    
    motion_data = data_loader.dataset.datasets[0].dataset.motion_data
    if 'subsampled_indices' in motion_data:
        video_info = data_loader.dataset.datasets[0].dataset.video_info
    
    if infer_split == 'train':
        dataset_name = args.get('dataset', 'assembly')
    else:
        dataset_name = args.get('val_dataset', 'assembly')

    save_pred = args.get('save_pred', None)
    if save_pred is not None:
        save_dir = f'{os.environ["DOWNLOADS_DIR"]}/lifted_labels/{dataset_name}_preds_iter_{save_pred:02d}'
        os.makedirs(save_dir, exist_ok=True)

    preds_labels = {}
    return_metrics = args.get('return_metrics', False)
    dataset_means = {}
    relevant_keys = ['mpjpe/f/h', 'mpjpe/f/pag/h', 'mpjpe/f/paf/h', 'mpjpe/f/ra/h', 'mrrpe/f/r/l']
    for k in relevant_keys:
        dataset_means[k] = []
    for idx, batch in enumerate(tqdm(data_loader)):
        inputs, targets, meta_info = batch
        inputs = to_device(inputs, device)
        meta_info = to_device(meta_info, device)
        targets = to_device(targets, device)

        out = wrapper.refine_model_predictions(inputs, targets, meta_info, return_metrics=return_metrics)
        
        if return_metrics:
            pose_r, transl_r, pose_l, transl_l, metrics = out
        else:
            pose_r, transl_r, pose_l, transl_l = out

        if pose_r is None or transl_r is None or pose_l is None or transl_l is None:
            logger.warning("Skipping batch due to NaN values in output")
            continue

        inputs = to_device(inputs, "cpu")
        targets = to_device(targets, "cpu")
        meta_info = to_device(meta_info, "cpu")

        # poses are (B, T, 48), transl are (B, T, 3)
        # use mask_timesteps to get the relevant future timesteps
        # transfer to cpu
        pose_r = pose_r.cpu().numpy()
        transl_r = transl_r.cpu().numpy()
        pose_l = pose_l.cpu().numpy()
        transl_l = transl_l.cpu().numpy()
        mask_timesteps = meta_info['mask_timesteps'].cpu().numpy()
        future_indices = meta_info['future_ids'].cpu().numpy()

        bz = pose_r.shape[0]
        for b in range(bz):
            curr_pose_r = pose_r[b]
            curr_transl_r = transl_r[b]
            curr_pose_l = pose_l[b]
            curr_transl_l = transl_l[b]

            curr_mask = mask_timesteps[b]
            mask_indices = np.where(curr_mask == 1)[0]
            # get the relevant pose and transl
            curr_pose_r = curr_pose_r[mask_indices]
            curr_pose_l = curr_pose_l[mask_indices]
            curr_transl_l = curr_transl_l[mask_indices]
            curr_transl_r = curr_transl_r[mask_indices]

            if return_metrics:
                for k in relevant_keys:
                    dataset_means[k].append(np.nanmean(metrics[k][b]))

            if save_pred is not None:
                # get image name
                seqname = meta_info['video_name'][b]
                future_ind = future_indices[b]
                future_ind = future_ind[mask_indices]
                if 'subsampled_indices' in motion_data:
                    if 'assembly' in dataset_name:
                        imgname = meta_info['imgname'][b][-1]
                        img_idx = imgname.split('/')[-1].split('.')[0]
                        motion_idx = video_info[seqname][img_idx]
                        range_indices = motion_data['subsampled_indices'][motion_idx]
                        future_ind = [range_indices[i] for i in future_ind]
                    else:
                        future_ind = [video_info[seqname]['indices'][i] for i in future_ind]
                
                for j, idx in enumerate(future_ind):
                    rel_idx = idx
                    if 'assembly' in dataset_name:
                        rel_idx = str(rel_idx).zfill(6)
                    elif 'holo' in dataset_name:
                        rel_idx = str(rel_idx).zfill(6)
                    elif 'epic' in dataset_name:
                        rel_idx = f'frame_{str(rel_idx).zfill(10)}'
                    if seqname not in preds_labels:
                        preds_labels[seqname] = {}
                    preds_labels[seqname][rel_idx] = curr_pose_r[j], curr_transl_r[j], curr_pose_l[j], curr_transl_l[j]

                    label_dir = os.path.join(save_dir, seqname)
                    os.makedirs(label_dir, exist_ok=True)
                    label_file = os.path.join(label_dir, f"{rel_idx}.npz")
                    preds = {
                        'pose_r': curr_pose_r[j],
                        'transl_r': curr_transl_r[j],
                        'pose_l': curr_pose_l[j],
                        'transl_l': curr_transl_l[j]
                    }
                    np.savez_compressed(label_file, **preds)
    
    if return_metrics:
        for k, v in dataset_means.items():
            dataset_means[k] = np.nanmean(v, axis=0)
            print (f'{k}: {dataset_means[k]}')


if __name__ == "__main__":
    main(args)
