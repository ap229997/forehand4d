import importlib
import argparse
from easydict import EasyDict

from common.args_utils import set_default_params, set_extra_params
from src.parsers.generic_parser import add_generic_args


def get_config_for_method(method):
    """
    Returns the config module for a given method name using a registry.
    Raises ValueError if method is not found.
    """
    registry = {
        "latentact": "src.parsers.configs.latentact",
        "latentact_light": "src.parsers.configs.latentact",
        "mdm": "src.parsers.configs.mdm",
        "mdm_light": "src.parsers.configs.mdm",
        "mdm_ff": "src.parsers.configs.mdm_ff",
        "mdm_ff_light": "src.parsers.configs.mdm_ff",
        "mdm_hybrid": "src.parsers.configs.mdm_hybrid",
        "mdm_hybrid_light": "src.parsers.configs.mdm_hybrid",
    }
    if method not in registry:
        raise ValueError(f"Method '{method}' not found in registry.")
    return importlib.import_module(registry[method])


def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=[None, "latentact_light", "mdm_light", "mdm_ff_light", "mdm_hybrid_light"],
    )
    parser.add_argument("--root_dir", type=str, default='./logs')
    parser.add_argument("--exp_key", type=str, default=None)
    parser.add_argument("--extraction_mode", type=str, default=None)
    parser.add_argument("--img_feat_version", type=str, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser = add_generic_args(parser)
    args = EasyDict(vars(parser.parse_args()))

    config = get_config_for_method(args.method)

    default_args = config.DEFAULT_ARGS_EGO
    args = set_default_params(args, default_args) # only preserves keys from args
    args = set_extra_params(args, default_args) # also preserves new keys from default_args which are not present in args

    args.focal_length = 1000.0
    args.rot_factor = args.get('rot_factor', 0.0)
    args.noise_factor = args.get('noise_factor', 0.0)
    args.scale_factor = args.get('scale_factor', 0.0)

    args.img_norm_mean = [0.485, 0.456, 0.406]
    args.img_norm_std = [0.229, 0.224, 0.225]
    args.pin_memory = True
    args.shuffle_train = True
    args.seed = 1

    # for compatibility with arctic code
    args.use_gt_k = False
    args.speedup = True  # load cropped images for faster training, only for arctic
    args.max_dist = 0.10  # distance range the model predicts on
    args.ego_image_scale = 0.3
    args.project = "arctic"
    args.interface_p = None

    if args.debug:
        args.eval_every_epoch = 50000000000

    return args
