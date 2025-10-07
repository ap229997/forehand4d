import torch

import common.torch_utils as torch_utils
import common.data_utils as data_utils
import common.ld_utils as ld_utils
from common.xdict import xdict
from src.callbacks.loss.loss_function import compute_loss_motion, compute_loss_2d
from src.callbacks.process.process_arctic import process_data_light
from src.callbacks.vis.visualize_arctic import visualize_motion
from src.models.mdm.model.mano_optimizer import optimize_mano_params
from src.models.generic.wrapper import GenericWrapper, mul_loss_dict
from src.utils.eval_modules import eval_fn_dict
from src.models.mdm_ff.model import MotionFeedForward

from src.callbacks.process.process_arctic import process_future_data
from src.utils.eval_modules import eval_fn_dict


class MotionFeedForwardWrapper(GenericWrapper):
    def __init__(self, args, push_images_fn):
        super().__init__(args, push_images_fn)

        self.args = args
        self.model = MotionFeedForward(
            focal_length=args.focal_length,
            img_res=args.img_res,
            args=args,
        )

        self.process_fn = process_data_light
        self.loss_fn = compute_loss_motion
        self.metric_dict = [
            "mpjpe.future",
            "mrrpe.future",
        ]

        if 'assembly' in args.dataset or 'holo' in args.dataset or 'epic' in args.dataset:
            self.loss_fn = [compute_loss_motion, compute_loss_2d]

        self.motion_data = True
        self.vis_fns = [visualize_motion]
        self.num_vis_train = 1
        self.num_vis_val = 1

    def inference_diversity(self, inputs, targets, meta_info, mode='vis', num_samples=1):
        models = {
            "mano_r": self.mano_r,
            "mano_l": self.mano_l
        }

        self.set_flags(mode)
        inputs = xdict(inputs)
        targets = xdict(targets)
        meta_info = xdict(meta_info)

        with torch.no_grad():
            inputs, targets, meta_info = self.process_fn(
                models, inputs, targets, meta_info, mode, self.args
            )

            if hasattr(self, "motion_data") and self.motion_data:
                inputs, targets, meta_info = process_future_data(
                    models, inputs, targets, meta_info, mode, self.args
                )

        move_keys = ["object.v_len"]
        for key in move_keys:
            if key in targets:
                meta_info[key] = targets[key]
        meta_info["mano.faces.r"] = self.mano_r.faces
        meta_info["mano.faces.l"] = self.mano_l.faces

        all_vis_dict = []
        self.model.eval()
        for i in range(num_samples):
            with torch.no_grad():
                motion_out = self.model.sample(inputs, meta_info, targets)
                pose_r, transl_r, pose_l, transl_l = self.model.process_motion_output(motion_out)
                bz, ts = pose_r.shape[:2]
                K = meta_info['intrinsics'][:, -1:].repeat(1, ts, 1, 1)
                mano_output_r, mano_output_l = self.model.run_mano_on_pose_predictions(pose_r, transl_r, pose_l, transl_l, targets, K)

                mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
                mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
                
                mano_output_r['future.j3d.cam.r'] = mano_output_r.pop('mano.j3d.cam.r')
                mano_output_r['future.v3d.cam.r'] = mano_output_r.pop('mano.v3d.cam.r')
                mano_output_l['future.j3d.cam.l'] = mano_output_l.pop('mano.j3d.cam.l')
                mano_output_l['future.v3d.cam.l'] = mano_output_l.pop('mano.v3d.cam.l')

                # also pop j2d.norm keys
                mano_output_r['future.j2d.norm.r'] = mano_output_r.pop('mano.j2d.norm.r')
                mano_output_l['future.j2d.norm.l'] = mano_output_l.pop('mano.j2d.norm.l')

                output = xdict()
                output.merge(mano_output_r)
                output.merge(mano_output_l)

                pred = torch_utils.expand_dict_dims(output, curr_dim=0, dims=(bz, -1))
                motion_targets = self.model.get_target_motion(targets, meta_info)
                
            vis_dict = xdict()
            vis_dict.merge(inputs.prefix("inputs."))
            vis_dict.merge(pred.prefix("pred."))
            vis_dict.merge(targets.prefix("targets."))
            vis_dict.merge(meta_info.prefix("meta_info."))
            vis_dict['pred.motion'] = motion_out
            vis_dict['targets.motion'] = motion_targets
            vis_dict = vis_dict.detach()
            all_vis_dict.append(vis_dict)

        return all_vis_dict
    
    def forward(self, inputs, targets, meta_info, mode):
        models = {
            "mano_r": self.mano_r,
            "mano_l": self.mano_l
        }

        self.set_flags(mode)
        inputs = xdict(inputs)
        targets = xdict(targets)
        meta_info = xdict(meta_info)
        with torch.no_grad():
            inputs, targets, meta_info = self.process_fn(
                models, inputs, targets, meta_info, mode, self.args
            )

            if hasattr(self, "motion_data") and self.motion_data:
                inputs, targets, meta_info = process_future_data(
                    models, inputs, targets, meta_info, mode, self.args
                )

        move_keys = ["object.v_len"]
        for key in move_keys:
            if key in targets:
                meta_info[key] = targets[key]
        meta_info["mano.faces.r"] = self.mano_r.faces
        meta_info["mano.faces.l"] = self.mano_l.faces

        more_args = {}
        if hasattr(self, "motion_data") and self.motion_data:
            pred = self.model(inputs, meta_info, targets, **more_args)
        else:
            pred = self.model(inputs, meta_info)
        
        num_dims = len(meta_info['intrinsics'].shape)
        if num_dims == 4: # B x T x 3 x 3
            import common.torch_utils as torch_utils
            bz = meta_info['intrinsics'].shape[0]
            targets = torch_utils.reduce_dict_dims(targets, dims=[0, 1])
            pred = torch_utils.reduce_dict_dims(pred, dims=[0, 1])
        
        if isinstance(self.loss_fn, list):
            loss_dict = {}
            for loss_fn in self.loss_fn:
                curr_loss = loss_fn(
                    pred=pred, gt=targets, meta_info=meta_info, args=self.args, targets=targets,
                )
                loss_dict.update(curr_loss)
        else:
            loss_dict = self.loss_fn(
                pred=pred, gt=targets, meta_info=meta_info, args=self.args,
            )
        loss_dict = {k: (loss_dict[k][0].mean(), loss_dict[k][1]) for k in loss_dict}
        
        loss_dict_unweighted = loss_dict.copy()
        loss_dict = mul_loss_dict(loss_dict)
        loss_dict["loss"] = sum(loss_dict[k] for k in loss_dict)

        # conversion for vis and eval
        keys = list(pred.keys())
        for key in keys:
            # denormalize 2d keypoints
            if "2d.norm" in key and 'object' not in key:
                denorm_key = key.replace(".norm", "")
                assert key in targets.keys(), f"Do not have key {key}"

                val_pred = pred[key]
                val_gt = targets[key]

                val_denorm_pred = data_utils.unormalize_kp2d(
                    val_pred, self.args.img_res
                )
                val_denorm_gt = data_utils.unormalize_kp2d(val_gt, self.args.img_res)

                pred[denorm_key] = val_denorm_pred
                targets[denorm_key] = val_denorm_gt

        if num_dims == 4:
            pred = torch_utils.expand_dict_dims(pred, curr_dim=0, dims=(bz,-1))
            targets = torch_utils.expand_dict_dims(targets, curr_dim=0, dims=(bz,-1))
        
        if mode == "train":
            return {"out_dict": (inputs, targets, meta_info, pred), "loss": loss_dict, "loss_unweighted": loss_dict_unweighted}

        if mode == "vis":
            vis_dict = xdict()
            vis_dict.merge(inputs.prefix("inputs."))
            vis_dict.merge(pred.prefix("pred."))
            vis_dict.merge(targets.prefix("targets."))
            vis_dict.merge(meta_info.prefix("meta_info."))
            vis_dict = vis_dict.detach()
            return vis_dict

        # evaluate metrics
        metrics_all = self.evaluate_metrics(
            pred, targets, meta_info, self.metric_dict
        ).to_torch()
        out_dict = xdict()
        out_dict["imgname"] = meta_info["imgname"]
        out_dict.merge(ld_utils.prefix_dict(metrics_all, "metric."))

        if mode == "extract":
            mydict = xdict()
            mydict.merge(inputs.prefix("inputs."))
            mydict.merge(pred.prefix("pred."))
            mydict.merge(targets.prefix("targets."))
            mydict.merge(meta_info.prefix("meta_info."))
            mydict.merge(out_dict) # for analyzing metrics
            mydict = mydict.detach()
            return mydict
        return out_dict, loss_dict

    def evaluate_metrics(self, pred, targets, meta_info, specs):
        metric_dict = xdict()
        for key in specs:
            metrics = eval_fn_dict[key](pred, targets, meta_info)
            metric_dict.merge(metrics)
            if 'mpjpe' in key:
                # compute root_align, procrustus, procrustus_align, procrustus_first variants as well
                metrics = eval_fn_dict[key](pred, targets, meta_info, root_align=True)
                metric_dict.merge(metrics)
                metrics = eval_fn_dict[key](pred, targets, meta_info, procrustus=True)
                metric_dict.merge(metrics)
                metrics = eval_fn_dict[key](pred, targets, meta_info, procrustus_align=True)
                metric_dict.merge(metrics)
                metrics = eval_fn_dict[key](pred, targets, meta_info, procrustus_first=True)
                metric_dict.merge(metrics)

        return metric_dict
    
    def run_eval(self, inputs, targets, meta_info, mode='val', num_samples=1):
        models = {
            "mano_r": self.mano_r,
            "mano_l": self.mano_l
        }

        self.set_flags(mode)
        inputs = xdict(inputs)
        targets = xdict(targets)
        meta_info = xdict(meta_info)
        with torch.no_grad():
            inputs, targets, meta_info = self.process_fn(
                models, inputs, targets, meta_info, mode, self.args
            )

            if hasattr(self, "motion_data") and self.motion_data:
                inputs, targets, meta_info = process_future_data(
                    models, inputs, targets, meta_info, mode, self.args
                )

        move_keys = ["object.v_len"]
        for key in move_keys:
            if key in targets:
                meta_info[key] = targets[key]
        meta_info["mano.faces.r"] = self.mano_r.faces
        meta_info["mano.faces.l"] = self.mano_l.faces

        meta_info['stack'] = True # this returns individual timestep metrics instead of concatenated metrics
        metrics_all = xdict()
        self.model.eval()
        for i in range(num_samples):
            with torch.no_grad():
                motion_out = self.model.sample(inputs, meta_info, targets)
                pose_r, transl_r, pose_l, transl_l = self.model.process_motion_output(motion_out)
                bz, ts = pose_r.shape[:2]
                K = meta_info['intrinsics'][:, -1:].repeat(1, ts, 1, 1)
                mano_output_r, mano_output_l = self.model.run_mano_on_pose_predictions(pose_r, transl_r, pose_l, transl_l, targets, K)

                mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
                mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
                
                mano_output_r['future.j3d.cam.r'] = mano_output_r.pop('mano.j3d.cam.r')
                mano_output_r['future.v3d.cam.r'] = mano_output_r.pop('mano.v3d.cam.r')
                mano_output_l['future.j3d.cam.l'] = mano_output_l.pop('mano.j3d.cam.l')
                mano_output_l['future.v3d.cam.l'] = mano_output_l.pop('mano.v3d.cam.l')

                # also pop j2d.norm keys
                mano_output_r['future.j2d.norm.r'] = mano_output_r.pop('mano.j2d.norm.r')
                mano_output_l['future.j2d.norm.l'] = mano_output_l.pop('mano.j2d.norm.l')

                output = xdict()
                output.merge(mano_output_r)
                output.merge(mano_output_l)

                pred = torch_utils.expand_dict_dims(output, curr_dim=0, dims=(bz, -1))

            # evaluate metrics
            curr_metrics = self.evaluate_metrics(
                pred, targets, meta_info, self.metric_dict
            )
            # update metrics_all with curr_metrics
            for k, v in curr_metrics.items():
                if k not in metrics_all:
                    metrics_all[k] = []
                metrics_all[k].append(v)

        return metrics_all
    
    def run_viz(self, inputs, targets, meta_info, mode='vis', num_samples=1):
        models = {
            "mano_r": self.mano_r,
            "mano_l": self.mano_l
        }

        self.set_flags(mode)
        inputs = xdict(inputs)
        targets = xdict(targets)
        meta_info = xdict(meta_info)
        with torch.no_grad():
            inputs, targets, meta_info = self.process_fn(
                models, inputs, targets, meta_info, mode, self.args
            )

            if hasattr(self, "motion_data") and self.motion_data:
                inputs, targets, meta_info = process_future_data(
                    models, inputs, targets, meta_info, mode, self.args
                )

        move_keys = ["object.v_len"]
        for key in move_keys:
            if key in targets:
                meta_info[key] = targets[key]
        meta_info["mano.faces.r"] = self.mano_r.faces
        meta_info["mano.faces.l"] = self.mano_l.faces

        all_vis_dict = []
        self.model.eval()
        for i in range(num_samples):
            with torch.no_grad():
                motion_out = self.model.sample(inputs, meta_info, targets)
                pose_r, transl_r, pose_l, transl_l = self.model.process_motion_output(motion_out)
                bz, ts = pose_r.shape[:2]
                K = meta_info['intrinsics'][:, -1:].repeat(1, ts, 1, 1)
                mano_output_r, mano_output_l = self.model.run_mano_on_pose_predictions(pose_r, transl_r, pose_l, transl_l, targets, K)

                mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
                mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
                
                mano_output_r['future.j3d.cam.r'] = mano_output_r.pop('mano.j3d.cam.r')
                mano_output_r['future.v3d.cam.r'] = mano_output_r.pop('mano.v3d.cam.r')
                mano_output_l['future.j3d.cam.l'] = mano_output_l.pop('mano.j3d.cam.l')
                mano_output_l['future.v3d.cam.l'] = mano_output_l.pop('mano.v3d.cam.l')

                # also pop j2d.norm keys
                mano_output_r['future.j2d.norm.r'] = mano_output_r.pop('mano.j2d.norm.r')
                mano_output_l['future.j2d.norm.l'] = mano_output_l.pop('mano.j2d.norm.l')

                output = xdict()
                output.merge(mano_output_r)
                output.merge(mano_output_l)

                pred = torch_utils.expand_dict_dims(output, curr_dim=0, dims=(bz, -1))

            if mode == "vis":
                vis_dict = xdict()
                vis_dict.merge(inputs.prefix("inputs."))
                vis_dict.merge(pred.prefix("pred."))
                vis_dict.merge(targets.prefix("targets."))
                vis_dict.merge(meta_info.prefix("meta_info."))
                vis_dict = vis_dict.detach()
                all_vis_dict.append(vis_dict)

        return all_vis_dict