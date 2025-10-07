import torch
from torch.utils.data import DataLoader

from common.tb_utils import push_images
import src.models as models
import src.datasets as datasets
from src.registry import ModelRegistry
from src.dataset_registry import DatasetRegistry


def fetch_dataset(args, is_train):
    split = args.trainsplit if is_train else args.valsplit
    relevant_dataset = args.get('dataset', None) if is_train else args.get('val_dataset', None)
    method = args.method.lower()
    # Map method to dataset registry key
    method_to_dataset = {
        "mdm2d": "variable_length_motion_2d",
        "mdm2d_light": "variable_length_motion_2d",
    } # everything else is variable_length_motion wrapper
    DATASET_KEY = method_to_dataset.get(method, "variable_length_motion")
    DATASET = DatasetRegistry.get_dataset(DATASET_KEY)

    if relevant_dataset is None:
        ds = DATASET(args=args, split=split)
    else:
        sets = relevant_dataset.split('+')
        ds_list = []
        for s in sets:
            s_key = s.lower()
            # Map dataset string to registry key
            dataset_map = {
                "arctic_ego": "arctic_light_dataset",
                "arctic_exo": "arctic_light_dataset",
                "h2o": "h2o_dataset",
                "h2o3d": "h2o3d_dataset",
                "dexycb": "dexycb_dataset",
                "hot3d": "hot3d_dataset",
                "assembly": "assembly_dataset",
                "epic": "epic_dataset",
                "holo": "holo_dataset",
                "egoexo": "ego_exo_dataset",
            }
            ds_cls = DatasetRegistry.get_dataset(dataset_map.get(s_key, DATASET_KEY))
            if s_key in ["arctic_ego"]:
                curr_ds = ds_cls(args, split, setup='p2')
            elif s_key in ["arctic_exo"]:
                curr_ds = ds_cls(args, split, setup='p1')
            else:
                curr_ds = ds_cls(args, split)
            curr_ds = DATASET(args, curr_ds, split) # motion dataset wrapper
            ds_list.append(curr_ds)
        ds = torch.utils.data.ConcatDataset(ds_list)
    return ds


def collate_custom_fn(data_list):
    data = data_list[0]
    _inputs, _targets, _meta_info = data
    out_inputs = {}
    out_targets = {}
    out_meta_info = {}

    for key in _inputs.keys():
        out_inputs[key] = []

    for key in _targets.keys():
        out_targets[key] = []

    for key in _meta_info.keys():
        out_meta_info[key] = []

    for data in data_list:
        inputs, targets, meta_info = data
        for key, val in inputs.items():
            out_inputs[key].append(val)

        for key, val in targets.items():
            out_targets[key].append(val)

        for key, val in meta_info.items():
            out_meta_info[key].append(val)

    for key in _inputs.keys():
        out_inputs[key] = torch.cat(out_inputs[key], dim=0)

    for key in _targets.keys():
        out_targets[key] = torch.cat(out_targets[key], dim=0)

    for key in _meta_info.keys():
        if key not in ["imgname", "query_names"]:
            out_meta_info[key] = torch.cat(out_meta_info[key], dim=0)
        else:
            out_meta_info[key] = sum(out_meta_info[key], [])

    return out_inputs, out_targets, out_meta_info


def collate_stack_fn(data_list):
    data = data_list[0]
    _inputs, _targets, _meta_info = data
    out_inputs = {}
    out_targets = {}
    out_meta_info = {}

    for key in _inputs.keys():
        out_inputs[key] = []

    for key in _targets.keys():
        out_targets[key] = []

    for key in _meta_info.keys():
        out_meta_info[key] = []

    for data in data_list:
        inputs, targets, meta_info = data
        for key, val in inputs.items():
            out_inputs[key].append(val)

        for key, val in targets.items():
            out_targets[key].append(val)

        for key, val in meta_info.items():
            out_meta_info[key].append(val)

    for key in _inputs.keys():
        out_inputs[key] = torch.stack(out_inputs[key], dim=0)

    for key in _targets.keys():
        out_targets[key] = torch.stack(out_targets[key], dim=0)

    for key in _meta_info.keys():
        if key not in ["imgname", "query_names", "dataset", "video_name"]:
            out_meta_info[key] = torch.stack(out_meta_info[key], dim=0)
        # else:
        #     out_meta_info[key] = sum(out_meta_info[key], [])

    return out_inputs, out_targets, out_meta_info


def fetch_motion_dataloader(args, mode):
    is_train = "train" in mode
    dataset = fetch_dataset(args, is_train=is_train)
    batch_size = args.batch_size if is_train else args.test_batch_size
    shuffle = args.shuffle_train if is_train else False
    pin_memory = getattr(args, 'pin_memory', False)
    num_workers = getattr(args, 'num_workers', 0)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        collate_fn=collate_stack_fn,
    )


def fetch_model(args):
    try:
        Wrapper = ModelRegistry.get_model(args.method)
    except KeyError:
        raise ValueError(f"Invalid method ({args.method})")
    
    if args.logger == "comet":
        model = Wrapper(args)
    elif args.logger == "tensorboard":
        model = Wrapper(args, push_images_fn=push_images)
    return model
