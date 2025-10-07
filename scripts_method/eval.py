import os
import math
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

    if '1t' in args.load_ckpt:
        args.max_motion_length = 256
        args.use_fixed_length = False

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
    
    if 'mdm_ff' in args.load_ckpt or 'latentact' in args.method:
        num_samples = 1
    else:
        num_samples = 5 # best of 5 generated motions for probabilistic methods

    relevant_keys = ['mpjpe/f/h', 'mpjpe/f/pag/h', 'mpjpe/f/paf/h', 'mpjpe/f/ra/h', 'mrrpe/f/r/l']
    dataset_means = {}
    for k in relevant_keys:
        dataset_means[k] = []
    all_metrics, full_metrics = {}, {}
    
    for idx, batch in enumerate(tqdm(data_loader)):
        inputs, targets, meta_info = batch
        inputs = to_device(inputs, device)
        meta_info = to_device(meta_info, device)
        targets = to_device(targets, device)

        out = wrapper.run_eval(inputs, targets, meta_info, num_samples=num_samples)

        bz = meta_info['mask_timesteps'].shape[0]
        
        for b in range(bz):

            relevant_metrics = {}
            for k in relevant_keys:
                relevant_metrics[k] = []
            mean_metrics = {}
            for k in relevant_keys:
                mean_metrics[k] = []
            
            for n in range(num_samples):
                for k in relevant_keys:
                    # out[k] is a list of batch of list of individual values for each timestep
                    # convert to batch of list of list of values for each timestep
                    relevant_metrics[k].append(out[k][n][b])
            
            for k in relevant_keys:
                # each value is list of list of values for each timestep
                curr_values = []
                for n in range(num_samples):
                    curr_ts = relevant_metrics[k][n]
                    curr_mean = np.nanmean(curr_ts, axis=0)
                    curr_values.append(curr_mean)
                mean_metrics[k] = curr_values

            curr_min = 1e9
            for n in range(num_samples):
                curr_sum = 0
                for k, v in mean_metrics.items():
                    if math.isnan(v[n]):
                        v[n] = 1e7
                    curr_sum += v[n]
                if curr_sum < curr_min:
                    curr_min = curr_sum
                    best_sample = n

            # get the best_sample for this batch
            best_metrics = {}
            for k in relevant_keys:
                best_metrics[k] = relevant_metrics[k][best_sample]
                dataset_means[k].append(np.nanmean(best_metrics[k], axis=0))

            curr_imgname = meta_info['imgname'][b][-1]
            all_metrics[curr_imgname] = best_metrics

            full_metrics[curr_imgname] = {}
            for k in relevant_keys:
                curr_values = []
                for n in range(num_samples):
                    curr_ts = relevant_metrics[k][n]
                    curr_values.append(curr_ts)
                curr_values = np.array(curr_values) # (N, T)
                full_metrics[curr_imgname][k] = curr_values

        if args.debug and idx > 1:
            break

    for k, v in dataset_means.items():
        dataset_means[k] = np.nanmean(v, axis=0)
        print (f'{k}: {dataset_means[k]}')


if __name__ == "__main__":
    main(args)