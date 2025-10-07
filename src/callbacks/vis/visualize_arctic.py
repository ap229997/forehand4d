import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import common.thing as thing
import common.transforms as tf
import common.vis_utils as vis_utils
from common.data_utils import denormalize_images
from common.mesh import Mesh
from common.rend_utils import color2material
from common.torch_utils import unpad_vtensor

mesh_color_dict = {
    "right": [100,100,254], # blue
    "left": [183,100,254], # purple
    "object": [144, 250, 100],
    "top": [144, 250, 100],
    "bottom": [129, 159, 214],
}


def visualize_one_example(
    images_i,
    kp2d_proj_b_i,
    kp2d_proj_t_i,
    joints2d_r_i,
    joints2d_l_i,
    kp2d_b_i,
    kp2d_t_i,
    bbox2d_b_i,
    bbox2d_t_i,
    joints2d_proj_r_i,
    joints2d_proj_l_i,
    bbox2d_proj_b_i,
    bbox2d_proj_t_i,
    joints_valid_r,
    joints_valid_l,
    flag,
    only_hands
):
    # whether the hand is cleary visible
    valid_idx_r = (joints_valid_r.long() == 1).nonzero().view(-1).numpy()
    valid_idx_l = (joints_valid_l.long() == 1).nonzero().view(-1).numpy()

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax = ax.reshape(-1)

    # GT 2d keypoints (good overlap as it is from perspective camera)
    ax[0].imshow(images_i)
    if not only_hands:
        ax[0].scatter(
            kp2d_b_i[:, 0], kp2d_b_i[:, 1], color="r"
        )  # keypoints from bottom part of object
        ax[0].scatter(kp2d_t_i[:, 0], kp2d_t_i[:, 1], color="b")  # keypoints from top part

    # right hand keypoints
    ax[0].scatter(
        joints2d_r_i[valid_idx_r, 0],
        joints2d_r_i[valid_idx_r, 1],
        color="r",
        marker="x",
    )
    ax[0].scatter(
        joints2d_l_i[valid_idx_l, 0],
        joints2d_l_i[valid_idx_l, 1],
        color="b",
        marker="x",
    )
    ax[0].set_title(f"{flag} 2D keypoints")

    # GT 2d keypoints (good overlap as it is from perspective camera)
    ax[1].imshow(images_i)
    if not only_hands:
        vis_utils.plot_2d_bbox(bbox2d_b_i, None, "r", ax[1])
        vis_utils.plot_2d_bbox(bbox2d_t_i, None, "b", ax[1])
    ax[1].set_title(f"{flag} 2D bbox")

    # GT 3D keypoints projected to 2D using weak perspective projection
    # (sometimes not completely overlap because of a weak perspective camera)
    ax[2].imshow(images_i)
    if not only_hands:
        ax[2].scatter(kp2d_proj_b_i[:, 0], kp2d_proj_b_i[:, 1], color="r")
        ax[2].scatter(kp2d_proj_t_i[:, 0], kp2d_proj_t_i[:, 1], color="b")
    ax[2].scatter(
        joints2d_proj_r_i[valid_idx_r, 0],
        joints2d_proj_r_i[valid_idx_r, 1],
        color="r",
        marker="x",
    )
    ax[2].scatter(
        joints2d_proj_l_i[valid_idx_l, 0],
        joints2d_proj_l_i[valid_idx_l, 1],
        color="b",
        marker="x",
    )
    ax[2].set_title(f"{flag} 3D keypoints reprojection from cam")

    # GT 3D bbox projected to 2D using weak perspective projection
    # (sometimes not completely overlap because of a weak perspective camera)
    ax[3].imshow(images_i)
    if not only_hands:
        vis_utils.plot_2d_bbox(bbox2d_proj_b_i, None, "r", ax[3])
        vis_utils.plot_2d_bbox(bbox2d_proj_t_i, None, "b", ax[3])
    ax[3].set_title(f"{flag} 3D keypoints reprojection from cam")

    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    fig.tight_layout()
    plt.close()

    im = vis_utils.fig2img(fig)
    return im


def visualize_kps(vis_dict, flag, max_examples, only_hands):
    # visualize keypoints for predition or GT
    images = (vis_dict["vis.images"].permute(0, 2, 3, 1) * 255).numpy().astype(np.uint8)
    K = vis_dict["meta_info.intrinsics"]
    if not only_hands:
        kp2d_b = vis_dict[f"{flag}.object.kp2d.b"].numpy()
        kp2d_t = vis_dict[f"{flag}.object.kp2d.t"].numpy()
        bbox2d_b = vis_dict[f"{flag}.object.bbox2d.b"].numpy()
        bbox2d_t = vis_dict[f"{flag}.object.bbox2d.t"].numpy()

    joints2d_r = vis_dict[f"{flag}.mano.j2d.r"].numpy()
    joints2d_l = vis_dict[f"{flag}.mano.j2d.l"].numpy()

    if not only_hands:
        kp3d_o = vis_dict[f"{flag}.object.kp3d.cam"]
        bbox3d_o = vis_dict[f"{flag}.object.bbox3d.cam"]
        kp2d_proj = tf.project2d_batch(K, kp3d_o)
        kp2d_proj_t, kp2d_proj_b = torch.split(kp2d_proj, [16, 16], dim=1)
        kp2d_proj_t = kp2d_proj_t.numpy()
        kp2d_proj_b = kp2d_proj_b.numpy()

        bbox2d_proj = tf.project2d_batch(K, bbox3d_o)
        bbox2d_proj_t, bbox2d_proj_b = torch.split(bbox2d_proj, [8, 8], dim=1)
        bbox2d_proj_t = bbox2d_proj_t.numpy()
        bbox2d_proj_b = bbox2d_proj_b.numpy()

    # project 3D to 2D using weak perspective camera (not completely overlap)
    joints3d_r = vis_dict[f"{flag}.mano.j3d.cam.r"]
    joints2d_proj_r = tf.project2d_batch(K, joints3d_r).numpy()
    joints3d_l = vis_dict[f"{flag}.mano.j3d.cam.l"]
    joints2d_proj_l = tf.project2d_batch(K, joints3d_l).numpy()

    joints_valid_r = vis_dict["targets.joints_valid_r"]
    joints_valid_l = vis_dict["targets.joints_valid_l"]

    im_list = []
    for idx in range(min(images.shape[0], max_examples)):
        image_id = vis_dict["vis.image_ids"][idx]
        if not only_hands:
            im = visualize_one_example(
                images[idx],
                kp2d_proj_b[idx],
                kp2d_proj_t[idx],
                joints2d_r[idx],
                joints2d_l[idx],
                kp2d_b[idx],
                kp2d_t[idx],
                bbox2d_b[idx],
                bbox2d_t[idx],
                joints2d_proj_r[idx],
                joints2d_proj_l[idx],
                bbox2d_proj_b[idx],
                bbox2d_proj_t[idx],
                joints_valid_r[idx],
                joints_valid_l[idx],
                flag,
                only_hands,
            )
        else:
            im = visualize_one_example(
                images[idx],
                None,
                None,
                joints2d_r[idx],
                joints2d_l[idx],
                None,
                None,
                None,
                None,
                joints2d_proj_r[idx],
                joints2d_proj_l[idx],
                None,
                None,
                joints_valid_r[idx],
                joints_valid_l[idx],
                flag,
                only_hands,
            )
        im_list.append({"fig_name": f"{image_id}__kps", "im": im})
    return im_list


def visualize_rend(
    renderer,
    vertices_r,
    vertices_l,
    mano_faces_r,
    mano_faces_l,
    vertices_o,
    faces_o,
    r_valid,
    l_valid,
    K,
    img,
    only_hands,
    only_obj=False,
):
    # render 3d meshes
    mesh_r = Mesh(v=vertices_r, f=mano_faces_r)
    mesh_l = Mesh(v=vertices_l, f=mano_faces_l)
    if not only_hands:
        mesh_o = Mesh(v=thing.thing2np(vertices_o), f=thing.thing2np(faces_o))

    # render only valid meshes
    meshes = []
    mesh_names = []
    if r_valid:
        meshes.append(mesh_r)
        mesh_names.append("right")

    if l_valid:
        meshes.append(mesh_l)
        mesh_names.append("left")
    if not only_hands:
        meshes = meshes + [mesh_o]
        mesh_names = mesh_names + ["object"]

    if only_obj:
        meshes = [mesh_o]
        mesh_names = ["object"]

    materials = [color2material(mesh_color_dict[name]) for name in mesh_names]

    if len(meshes) > 0:
        # render in image space
        render_img_img = renderer.render_meshes_pose(
            cam_transl=None,
            meshes=meshes,
            image=img,
            materials=materials,
            sideview_angle=None,
            K=K,
        )
        render_img_list = [render_img_img]

        # render rotated meshes
        for angle in list(np.linspace(45, 300, 3)):
            render_img_angle = renderer.render_meshes_pose(
                cam_transl=None,
                meshes=meshes,
                image=None,
                materials=materials,
                sideview_angle=angle,
                K=K,
            )
            render_img_list.append(render_img_angle)

        # cat all images
        render_img = np.concatenate(render_img_list, axis=0)
        return render_img

    else:
        # dummy image
        render_img = np.concatenate([img] * 4, axis=0)
        return render_img


def visualize_rends(renderer, vis_dict, max_examples, only_hands, return_pil=False):
    # render meshes

    # unpack data
    image_ids = vis_dict["vis.image_ids"]
    right_valid = vis_dict["targets.right_valid"].bool()
    left_valid = vis_dict["targets.left_valid"].bool()
    images = vis_dict["vis.images"].permute(0, 2, 3, 1).numpy()
    gt_vertices_r_cam = vis_dict["targets.mano.v3d.cam.r"]
    gt_vertices_l_cam = vis_dict["targets.mano.v3d.cam.l"]
    mano_faces_r = vis_dict["meta_info.mano.faces.r"]
    mano_faces_l = vis_dict["meta_info.mano.faces.l"]
    pred_vertices_r_cam = vis_dict["pred.mano.v3d.cam.r"]
    pred_vertices_l_cam = vis_dict["pred.mano.v3d.cam.l"]

    if not only_hands:
        # object
        gt_obj_v_cam = unpad_vtensor(
            vis_dict["targets.object.v.cam"], vis_dict["targets.object.v_len"]
        )  # meter
        pred_obj_v_cam = unpad_vtensor(
            vis_dict["pred.object.v.cam"], vis_dict["pred.object.v_len"]
        )
        pred_obj_f = unpad_vtensor(vis_dict["pred.object.f"], vis_dict["pred.object.f_len"])
    K = vis_dict["meta_info.intrinsics"]

    # rendering
    im_list = []
    for idx in range(min(len(image_ids), max_examples)):
        r_valid = right_valid[idx]
        l_valid = left_valid[idx]
        K_i = K[idx]
        image_id = image_ids[idx]

        # render gt
        image_list = []
        image_list.append(images[idx])
        if not only_hands:
            image_gt = visualize_rend(
                renderer,
                gt_vertices_r_cam[idx],
                gt_vertices_l_cam[idx],
                mano_faces_r,
                mano_faces_l,
                gt_obj_v_cam[idx],
                pred_obj_f[idx],
                r_valid,
                l_valid,
                K_i,
                images[idx],
                only_hands,
            )
        else:
            image_gt = visualize_rend(
                renderer,
                gt_vertices_r_cam[idx],
                gt_vertices_l_cam[idx],
                mano_faces_r,
                mano_faces_l,
                None,
                None,
                r_valid,
                l_valid,
                K_i,
                images[idx],
                only_hands,
            )
        image_list.append(image_gt)

        # render pred
        if not only_hands:
            image_pred = visualize_rend(
                renderer,
                pred_vertices_r_cam[idx],
                pred_vertices_l_cam[idx],
                mano_faces_r,
                mano_faces_l,
                pred_obj_v_cam[idx],
                pred_obj_f[idx],
                r_valid,
                l_valid,
                K_i,
                images[idx],
                only_hands,
            )
        else:
            image_pred = visualize_rend(
                renderer,
                pred_vertices_r_cam[idx],
                pred_vertices_l_cam[idx],
                mano_faces_r,
                mano_faces_l,
                None,
                None,
                r_valid,
                l_valid,
                K_i,
                images[idx],
                only_hands,
            )
        image_list.append(image_pred)

        if not return_pil:
            # stack images into one
            image_pred = vis_utils.im_list_to_plt(
                image_list,
                figsize=(15, 8),
                title_list=["input image", "GT", "pred w/ pred_cam_t"],
            )
        im_list.append(
            {
                "fig_name": f"{image_id}__rend_rvalid={r_valid}, lvalid={l_valid} ",
                "im": image_pred,
            }
        )
    return im_list


def visualize_single(
    renderer,
    vertices_r, # V x 3
    vertices_l, # V x 3
    mano_faces_r,
    mano_faces_l,
    vertices_o,
    faces_o,
    r_valid, # T booleans
    l_valid, # T booleans
    K,
    img,
    only_hands,
    only_obj=False,
    fading_factor=0.9,
):
    
    meshes = []
    mesh_names = []
    materials = []

    mesh_r = Mesh(v=vertices_r, f=mano_faces_r)
    mesh_l = Mesh(v=vertices_l, f=mano_faces_l)
    if r_valid:
        meshes.append(mesh_r)
        mesh_names.append("right")
        materials.append(color2material(mesh_color_dict["right"]))
    if l_valid:
        meshes.append(mesh_l)
        mesh_names.append("left")
        materials.append(color2material(mesh_color_dict["left"]))

    # render 3d meshes
    if not only_hands:
        mesh_o = Mesh(v=thing.thing2np(vertices_o), f=thing.thing2np(faces_o))

    if not only_hands:
        meshes = meshes + [mesh_o]
        mesh_names = mesh_names + ["object"]
        color_o = mesh_color_dict['object']
        materials.append(color2material(color_o))

    if only_obj:
        meshes = [mesh_o]
        mesh_names = ["object"]
        color_o = mesh_color_dict['object']
        materials = [color2material(color_o)]

    if len(meshes) > 0:
        # render in image space
        render_img_img = renderer.render_meshes_pose(
            cam_transl=None,
            meshes=meshes,
            image=img,
            materials=materials,
            sideview_angle=None,
            K=K,
        )
        return render_img_img
    else:
        # dummy image
        render_img = np.concatenate([img] * 4, axis=0)
        return render_img


def visualize_single_views(
    renderer,
    vertices_r, # V x 3
    vertices_l, # V x 3
    mano_faces_r,
    mano_faces_l,
    vertices_o,
    faces_o,
    r_valid, # T booleans
    l_valid, # T booleans
    K,
    img,
    only_hands,
    only_obj=False,
    fading_factor=0.9,
):
    
    meshes = []
    mesh_names = []
    materials = []

    mesh_r = Mesh(v=vertices_r, f=mano_faces_r)
    mesh_l = Mesh(v=vertices_l, f=mano_faces_l)
    if r_valid:
        meshes.append(mesh_r)
        mesh_names.append("right")
        materials.append(color2material(mesh_color_dict["right"]))
    if l_valid:
        meshes.append(mesh_l)
        mesh_names.append("left")
        materials.append(color2material(mesh_color_dict["left"]))

    # render 3d meshes
    if not only_hands:
        mesh_o = Mesh(v=thing.thing2np(vertices_o), f=thing.thing2np(faces_o))

    if not only_hands:
        meshes = meshes + [mesh_o]
        mesh_names = mesh_names + ["object"]
        color_o = mesh_color_dict['object']
        materials.append(color2material(color_o))

    if only_obj:
        meshes = [mesh_o]
        mesh_names = ["object"]
        color_o = mesh_color_dict['object']
        materials = [color2material(color_o)]

    if len(meshes) > 0:
        # render in image space
        render_img_img = renderer.render_meshes_pose(
            cam_transl=None,
            meshes=meshes,
            image=img,
            materials=materials,
            sideview_angle=None,
            K=K,
        )
        render_img_list = [render_img_img]

        # render rotated meshes
        for angle in list(np.linspace(45, 300, 3)):
            render_img_angle = renderer.render_meshes_pose(
                cam_transl=None,
                meshes=meshes,
                image=None,
                materials=materials,
                sideview_angle=angle,
                # topdown_angle=angle,
                K=K,
            )
            render_img_list.append(render_img_angle)

        # cat all images
        render_img = np.concatenate(render_img_list, axis=0)
        return render_img

    else:
        # dummy image
        render_img = np.concatenate([img] * 4, axis=0)
        return render_img


def visualize_multiple(
    renderer,
    vertices_r, # T x V x 3
    vertices_l, # T x V x 3
    mano_faces_r,
    mano_faces_l,
    vertices_o,
    faces_o,
    r_valid, # T booleans
    l_valid, # T booleans
    K,
    img,
    only_hands,
    only_obj=False,
    fading_factor=0.9,
):
    
    meshes = []
    mesh_names = []
    materials = []

    # render 3d meshes
    ts = vertices_r.shape[0]
    skip_ts = 5
    max_ts = (256 / skip_ts) # hardocded for now TODO: either pass it as arg or get from renderer
    target_color = [255, 255, 255] # fade to white at the end of motion
    for t in range(0, ts, skip_ts):
        vertices_r_t = vertices_r[t]
        vertices_l_t = vertices_l[t]
        mesh_r_t = Mesh(v=vertices_r_t, f=mano_faces_r)
        mesh_l_t = Mesh(v=vertices_l_t, f=mano_faces_l)
        if r_valid[t]: # this is defined for the current timestep
            meshes.append(mesh_r_t)
            mesh_names.append("right")
            # color_r = [c * (fading_factor ** t) for c in mesh_color_dict['right']]
            color_r = [
                int(c + (target_color[i] - c) * (t / max_ts))
                for i, c in enumerate(mesh_color_dict['right'])
            ]
            materials.append(color2material(color_r))
        if l_valid[t]:
            meshes.append(mesh_l_t)
            mesh_names.append("left")
            # color_l = [c * (fading_factor ** t) for c in mesh_color_dict['left']]
            color_l = [
                int(c + (target_color[i] - c) * (t / max_ts))
                for i, c in enumerate(mesh_color_dict['left'])
            ]
            materials.append(color2material(color_l))

    if not only_hands:
        mesh_o = Mesh(v=thing.thing2np(vertices_o), f=thing.thing2np(faces_o))

    if not only_hands:
        meshes = meshes + [mesh_o]
        mesh_names = mesh_names + ["object"]
        color_o = mesh_color_dict['object']
        materials.append(color2material(color_o))

    if only_obj:
        meshes = [mesh_o]
        mesh_names = ["object"]
        color_o = mesh_color_dict['object']
        materials = [color2material(color_o)]

    if len(meshes) > 0:
        # render in image space
        render_img_img = renderer.render_meshes_pose(
            cam_transl=None,
            meshes=meshes,
            image=img,
            materials=materials,
            sideview_angle=None,
            K=K,
        )
        render_img_list = [render_img_img]

        # render rotated meshes
        for angle in list(np.linspace(45, 300, 3)):
            render_img_angle = renderer.render_meshes_pose(
                cam_transl=None,
                meshes=meshes,
                image=None,
                materials=materials,
                sideview_angle=angle,
                K=K,
            )
            render_img_list.append(render_img_angle)

        # cat all images
        render_img = np.concatenate(render_img_list, axis=0)
        return render_img

    else:
        # dummy image
        render_img = np.concatenate([img] * 4, axis=0)
        return render_img


def visualize_multiple_kps(
    vertices_r, # T x V x 2
    vertices_l, # T x V x 2
    vertices_o,
    faces_o,
    r_valid, # T booleans
    l_valid, # T booleans
    K,
    img,
    only_hands,
    flag,
    only_obj=False,
    fading_factor=0.9,
):
    
    kps_r, kps_l = [], []
    colors_r, colors_l = [], []

    # render 3d meshes
    ts = vertices_r.shape[0]
    skip_ts = 5
    max_ts = (256 / skip_ts) # hardocded for now TODO: either pass it as arg or get from renderer
    target_color = [255, 255, 255] # fade to white at the end of motion
    for t in range(0, ts, skip_ts):
        vertices_r_t = vertices_r[t]
        vertices_l_t = vertices_l[t]
        if r_valid[t]: # this is defined for the current timestep
            kps_r.append(vertices_r_t)
            # color_r = [c * (fading_factor ** t) for c in mesh_color_dict['right']]
            color_r = [
                int(c + (target_color[i] - c) * (t / max_ts)) / 255
                for i, c in enumerate(mesh_color_dict['right'])
            ]
            colors_r.append(color_r)
        if l_valid[t]:
            kps_l.append(vertices_l_t)
            # color_l = [c * (fading_factor ** t) for c in mesh_color_dict['left']]
            color_l = [
                int(c + (target_color[i] - c) * (t / max_ts)) / 255
                for i, c in enumerate(mesh_color_dict['left'])
            ]
            colors_l.append(color_l)

    # visualize keypoints for predition or GT
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)
    for i, kps in enumerate(kps_r):
        ax.scatter(kps[:, 0], kps[:, 1], color=colors_r[i], marker="x")
    for i, kps in enumerate(kps_l):
        ax.scatter(kps[:, 0], kps[:, 1], color=colors_l[i], marker="x")
    ax.set_title(f"{flag} 2D keypoints")
    ax.axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    fig.tight_layout()
    plt.close()

    im = vis_utils.fig2img(fig)
    return im


def future_motion_rends(renderer, vis_dict, max_examples, only_hands, return_pil=False):
    # render meshes

    # unpack data
    image_ids = vis_dict["vis.image_ids"]
    right_valid = vis_dict["targets.future_valid_r"].bool()
    left_valid = vis_dict["targets.future_valid_l"].bool()
    if len(right_valid.shape) == 3: # valid bools for each joint is available
        right_valid = (torch.sum(right_valid, dim=-1) >= 3)
        left_valid = (torch.sum(left_valid, dim=-1) >= 3)
    images = vis_dict["vis.images"].permute(0, 2, 3, 1).numpy()
    if 'vis.goal_images' in vis_dict:
        goal_images = vis_dict["vis.goal_images"].permute(0, 2, 3, 1).numpy()
    gt_vertices_r_cam = vis_dict["targets.future.v3d.cam.r"]
    gt_vertices_l_cam = vis_dict["targets.future.v3d.cam.l"]
    mano_faces_r = vis_dict["meta_info.mano.faces.r"]
    mano_faces_l = vis_dict["meta_info.mano.faces.l"]
    pred_vertices_r_cam = vis_dict["pred.future.v3d.cam.r"]
    pred_vertices_l_cam = vis_dict["pred.future.v3d.cam.l"]
    
    if 'meta_info.mask_timesteps' in vis_dict:
        mask_gt_vertices_r_cam = []
        mask_gt_vertices_l_cam = []
        mask_right_valid = []
        mask_left_valid = []
        mask_pred_vertices_r_cam = []
        mask_pred_vertices_l_cam = []
        bz, ts, num_verts = gt_vertices_r_cam.shape[:3]
        for b in range(bz):
            curr_mask = vis_dict['meta_info.mask_timesteps'][b]
            mask_gt_vertices_r_cam.append(gt_vertices_r_cam[b][curr_mask])
            mask_gt_vertices_l_cam.append(gt_vertices_l_cam[b][curr_mask])
            mask_right_valid.append(right_valid[b][curr_mask])
            mask_left_valid.append(left_valid[b][curr_mask])
            mask_pred_vertices_r_cam.append(pred_vertices_r_cam[b][curr_mask])
            mask_pred_vertices_l_cam.append(pred_vertices_l_cam[b][curr_mask])
        gt_vertices_r_cam = mask_gt_vertices_r_cam
        gt_vertices_l_cam = mask_gt_vertices_l_cam
        right_valid = mask_right_valid
        left_valid = mask_left_valid
        pred_vertices_r_cam = mask_pred_vertices_r_cam
        pred_vertices_l_cam = mask_pred_vertices_l_cam


    if not only_hands:
        # object
        gt_obj_v_cam = unpad_vtensor(
            vis_dict["targets.object.v.cam"], vis_dict["targets.object.v_len"]
        )  # meter
        pred_obj_v_cam = unpad_vtensor(
            vis_dict["pred.object.v.cam"], vis_dict["pred.object.v_len"]
        )
        pred_obj_f = unpad_vtensor(vis_dict["pred.object.f"], vis_dict["pred.object.f_len"])
    
    K = vis_dict["meta_info.intrinsics"]
    if len(K.shape) == 4:
        K = K[:, 0] # B x T x 3 x 3, same for all timesteps

    # rendering
    im_list = []
    for idx in range(min(len(image_ids), max_examples)):
        r_valid = right_valid[idx]
        l_valid = left_valid[idx]
        K_i = K[idx]
        image_id = image_ids[idx]

        # render gt
        image_list = []
        if 'vis.goal_images' in vis_dict:
            image_list.append(np.vstack([images[idx], goal_images[idx]]))
        else:
            image_list.append(images[idx])
        image_gt = images[idx].copy()
        image_pred = images[idx].copy()
        
        if not only_hands:
            image_gt = visualize_multiple(
                renderer,
                gt_vertices_r_cam[idx],
                gt_vertices_l_cam[idx],
                mano_faces_r,
                mano_faces_l,
                gt_obj_v_cam[idx],
                pred_obj_f[idx],
                r_valid,
                l_valid,
                K_i,
                image_gt,
                only_hands,
            )
        else:
            image_gt = visualize_multiple(
                renderer,
                gt_vertices_r_cam[idx],
                gt_vertices_l_cam[idx],
                mano_faces_r,
                mano_faces_l,
                None,
                None,
                r_valid,
                l_valid,
                K_i,
                image_gt,
                only_hands,
            )

        # render pred
        if not only_hands:
            image_pred = visualize_multiple(
                renderer,
                pred_vertices_r_cam[idx],
                pred_vertices_l_cam[idx],
                mano_faces_r,
                mano_faces_l,
                pred_obj_v_cam[idx],
                pred_obj_f[idx],
                r_valid,
                l_valid,
                K_i,
                image_pred,
                only_hands,
            )
        else:
            image_pred = visualize_multiple(
                renderer,
                pred_vertices_r_cam[idx],
                pred_vertices_l_cam[idx],
                mano_faces_r,
                mano_faces_l,
                None,
                None,
                r_valid,
                l_valid,
                K_i,
                image_pred,
                only_hands,
            )
        
        image_list.append(image_gt)
        image_list.append(image_pred)

        if not return_pil:
            # stack images into one
            image_pred = vis_utils.im_list_to_plt(
                image_list,
                figsize=(15, 8),
                title_list=["input image", "GT", "pred w/ pred_cam_t"],
            )
        im_list.append(
            {
                # "fig_name": f"{image_id}__rend_rvalid={r_valid}, lvalid={l_valid} ",
                "fig_name": f"{image_id}",
                "im": image_pred,
            }
        )
    return im_list


def future_motion_rends_viz(renderer, vis_dict, max_examples, only_hands, return_pil=True):
    # render meshes

    # unpack data
    image_ids = vis_dict["vis.image_ids"]
    right_valid = vis_dict["targets.future_valid_r"].bool()
    left_valid = vis_dict["targets.future_valid_l"].bool()
    if len(right_valid.shape) == 3: # valid bools for each joint is available
        right_valid = (torch.sum(right_valid, dim=-1) >= 3)
        left_valid = (torch.sum(left_valid, dim=-1) >= 3)
    images = vis_dict["vis.images"].permute(0, 2, 3, 1).numpy()
    if 'vis.goal_images' in vis_dict:
        goal_images = vis_dict["vis.goal_images"].permute(0, 2, 3, 1).numpy()
    
    mano_faces_r = vis_dict["meta_info.mano.faces.r"]
    mano_faces_l = vis_dict["meta_info.mano.faces.l"]
    pred_vertices_r_cam = vis_dict["pred.future.v3d.cam.r"]
    pred_vertices_l_cam = vis_dict["pred.future.v3d.cam.l"]
    
    if 'meta_info.mask_timesteps' in vis_dict:
        mask_right_valid = []
        mask_left_valid = []
        mask_pred_vertices_r_cam = []
        mask_pred_vertices_l_cam = []
        
        bz = vis_dict['meta_info.mask_timesteps'].shape[0]
        for b in range(bz):
            curr_mask = vis_dict['meta_info.mask_timesteps'][b]
            mask_right_valid.append(right_valid[b][curr_mask])
            mask_left_valid.append(left_valid[b][curr_mask])
            mask_pred_vertices_r_cam.append(pred_vertices_r_cam[b][curr_mask])
            mask_pred_vertices_l_cam.append(pred_vertices_l_cam[b][curr_mask])
        
        right_valid = mask_right_valid
        left_valid = mask_left_valid
        pred_vertices_r_cam = mask_pred_vertices_r_cam
        pred_vertices_l_cam = mask_pred_vertices_l_cam
    
    K = vis_dict["meta_info.intrinsics"]
    if len(K.shape) == 4:
        K = K[:, 0] # B x T x 3 x 3, same for all timesteps

    # rendering
    im_list = []
    for idx in range(min(len(image_ids), max_examples)):
        r_valid = right_valid[idx]
        l_valid = left_valid[idx]
        K_i = K[idx]
        image_id = image_ids[idx]

        # render gt
        image_list = []
        if 'vis.goal_images' in vis_dict:
            image_list.append(np.vstack([images[idx], goal_images[idx]]))
        else:
            image_list.append(images[idx])
        image_pred = images[idx].copy()

        # render pred
        image_pred = visualize_multiple(
            renderer,
            pred_vertices_r_cam[idx],
            pred_vertices_l_cam[idx],
            mano_faces_r,
            mano_faces_l,
            None,
            None,
            r_valid,
            l_valid,
            K_i,
            image_pred,
            only_hands,
        )
        
        image_list.append(image_pred)

        if not return_pil:
            # stack images into one
            image_pred = vis_utils.im_list_to_plt(
                image_list,
                figsize=(15, 8),
                title_list=["input image", "GT", "pred w/ pred_cam_t"],
            )
        im_list.append(
            {
                "fig_name": f"{image_id}",
                "im": image_pred,
                "inp_img": images[idx],
            }
        )
    return im_list


def future_motion_rends_video(renderer, vis_dict, max_examples, only_hands, return_pil=True):
    # render meshes

    # unpack data
    image_ids = vis_dict["vis.image_ids"]
    right_valid = vis_dict["targets.future_valid_r"].bool()
    left_valid = vis_dict["targets.future_valid_l"].bool()
    if len(right_valid.shape) == 3: # valid bools for each joint is available
        right_valid = (torch.sum(right_valid, dim=-1) >= 3)
        left_valid = (torch.sum(left_valid, dim=-1) >= 3)
    images = vis_dict["vis.images"].permute(0, 2, 3, 1).numpy()
    if 'vis.goal_images' in vis_dict:
        goal_images = vis_dict["vis.goal_images"].permute(0, 2, 3, 1).numpy()
    
    mano_faces_r = vis_dict["meta_info.mano.faces.r"]
    mano_faces_l = vis_dict["meta_info.mano.faces.l"]
    pred_vertices_r_cam = vis_dict["pred.future.v3d.cam.r"]
    pred_vertices_l_cam = vis_dict["pred.future.v3d.cam.l"]
    
    if 'meta_info.mask_timesteps' in vis_dict:
        mask_right_valid = []
        mask_left_valid = []
        mask_pred_vertices_r_cam = []
        mask_pred_vertices_l_cam = []
        
        bz = vis_dict['meta_info.mask_timesteps'].shape[0]
        for b in range(bz):
            curr_mask = vis_dict['meta_info.mask_timesteps'][b]
            mask_right_valid.append(right_valid[b][curr_mask])
            mask_left_valid.append(left_valid[b][curr_mask])
            mask_pred_vertices_r_cam.append(pred_vertices_r_cam[b][curr_mask])
            mask_pred_vertices_l_cam.append(pred_vertices_l_cam[b][curr_mask])
        
        right_valid = mask_right_valid
        left_valid = mask_left_valid
        pred_vertices_r_cam = mask_pred_vertices_r_cam
        pred_vertices_l_cam = mask_pred_vertices_l_cam
    
    K = vis_dict["meta_info.intrinsics"]
    if len(K.shape) == 4:
        K = K[:, 0] # B x T x 3 x 3, same for all timesteps

    # rendering
    im_list = []
    for idx in range(min(len(image_ids), max_examples)):
        r_valid = right_valid[idx]
        l_valid = left_valid[idx]
        K_i = K[idx]
        image_id = image_ids[idx]

        # render gt
        image_list = []
        if 'vis.goal_images' in vis_dict:
            image_list.append(np.vstack([images[idx], goal_images[idx]]))
        else:
            image_list.append(images[idx])
        image_pred = images[idx].copy()

        ts = pred_vertices_r_cam[idx].shape[0]
        for t in range(ts):
            # render pred
            image_pred = visualize_single(
                renderer,
                pred_vertices_r_cam[idx][t],
                pred_vertices_l_cam[idx][t],
                mano_faces_r,
                mano_faces_l,
                None,
                None,
                True,
                True,
                K_i,
                images[idx].copy(),
                only_hands,
            )
            
            image_list.append(image_pred)

        im_list.append(
            {
                "fig_name": f"{image_id}",
                "im": image_list,
                "inp_img": images[idx],
            }
        )
    return im_list


def future_motion_rends_video_views(renderer, vis_dict, max_examples, only_hands, return_pil=True):
    # render meshes

    # unpack data
    image_ids = vis_dict["vis.image_ids"]
    right_valid = vis_dict["targets.future_valid_r"].bool()
    left_valid = vis_dict["targets.future_valid_l"].bool()
    if len(right_valid.shape) == 3: # valid bools for each joint is available
        right_valid = (torch.sum(right_valid, dim=-1) >= 3)
        left_valid = (torch.sum(left_valid, dim=-1) >= 3)
    images = vis_dict["vis.images"].permute(0, 2, 3, 1).numpy()
    if 'vis.goal_images' in vis_dict:
        goal_images = vis_dict["vis.goal_images"].permute(0, 2, 3, 1).numpy()
    
    mano_faces_r = vis_dict["meta_info.mano.faces.r"]
    mano_faces_l = vis_dict["meta_info.mano.faces.l"]
    pred_vertices_r_cam = vis_dict["pred.future.v3d.cam.r"]
    pred_vertices_l_cam = vis_dict["pred.future.v3d.cam.l"]
    
    if 'meta_info.mask_timesteps' in vis_dict:
        mask_right_valid = []
        mask_left_valid = []
        mask_pred_vertices_r_cam = []
        mask_pred_vertices_l_cam = []
        
        bz = vis_dict['meta_info.mask_timesteps'].shape[0]
        for b in range(bz):
            curr_mask = vis_dict['meta_info.mask_timesteps'][b]
            mask_right_valid.append(right_valid[b][curr_mask])
            mask_left_valid.append(left_valid[b][curr_mask])
            mask_pred_vertices_r_cam.append(pred_vertices_r_cam[b][curr_mask])
            mask_pred_vertices_l_cam.append(pred_vertices_l_cam[b][curr_mask])
        
        right_valid = mask_right_valid
        left_valid = mask_left_valid
        pred_vertices_r_cam = mask_pred_vertices_r_cam
        pred_vertices_l_cam = mask_pred_vertices_l_cam
    
    K = vis_dict["meta_info.intrinsics"]
    if len(K.shape) == 4:
        K = K[:, 0] # B x T x 3 x 3, same for all timesteps

    # rendering
    im_list = []
    for idx in range(min(len(image_ids), max_examples)):
        r_valid = right_valid[idx]
        l_valid = left_valid[idx]
        K_i = K[idx]
        image_id = image_ids[idx]

        # render gt
        image_list = []
        if 'vis.goal_images' in vis_dict:
            image_list.append(np.vstack([images[idx], goal_images[idx]]))
        else:
            image_list.append(images[idx])
        image_pred = images[idx].copy()

        ts = pred_vertices_r_cam[idx].shape[0]
        for t in range(ts):
            # render pred
            image_pred = visualize_single_views(
                renderer,
                pred_vertices_r_cam[idx][t],
                pred_vertices_l_cam[idx][t],
                mano_faces_r,
                mano_faces_l,
                None,
                None,
                True,
                True,
                K_i,
                images[idx].copy(),
                only_hands,
            )
            
            image_list.append(image_pred)

        im_list.append(
            {
                "fig_name": f"{image_id}",
                "im": image_list,
                "inp_img": images[idx],
            }
        )
    return im_list


def future_motion_rends_lifting(renderer, vis_dict, max_examples, only_hands, return_pil=True):
    # render meshes

    # unpack data
    image_ids = vis_dict["vis.image_ids"]
    right_valid = vis_dict["targets.future_valid_r"].bool()
    left_valid = vis_dict["targets.future_valid_l"].bool()
    if len(right_valid.shape) == 3: # valid bools for each joint is available
        right_valid = (torch.sum(right_valid, dim=-1) >= 3)
        left_valid = (torch.sum(left_valid, dim=-1) >= 3)
    images = vis_dict["vis.images"].permute(0, 2, 3, 1).numpy()
    if 'vis.goal_images' in vis_dict:
        goal_images = vis_dict["vis.goal_images"].permute(0, 2, 3, 1).numpy()
    
    mano_faces_r = vis_dict["meta_info.mano.faces.r"]
    mano_faces_l = vis_dict["meta_info.mano.faces.l"]
    pred_vertices_r_cam = vis_dict["pred.future.v3d.cam.r"]
    pred_vertices_l_cam = vis_dict["pred.future.v3d.cam.l"]

    future2view = vis_dict['targets.future2view']
    view2future = torch.linalg.inv(future2view)
    bz, ts = view2future.shape[:2]
    # transform vertices to future space
    pred_vertices_r_cam = tf.transform_points_batch(
        view2future.reshape(bz*ts, 4, 4),
        pred_vertices_r_cam.reshape(bz*ts, -1, 3),
    )
    pred_vertices_l_cam = tf.transform_points_batch(
        view2future.reshape(bz*ts, 4, 4),
        pred_vertices_l_cam.reshape(bz*ts, -1, 3),
    )
    pred_vertices_r_cam = pred_vertices_r_cam.reshape(bz, ts, -1, 3)
    pred_vertices_l_cam = pred_vertices_l_cam.reshape(bz, ts, -1, 3)
    
    if 'meta_info.mask_timesteps' in vis_dict:
        mask_right_valid = []
        mask_left_valid = []
        mask_pred_vertices_r_cam = []
        mask_pred_vertices_l_cam = []
        
        bz = vis_dict['meta_info.mask_timesteps'].shape[0]
        for b in range(bz):
            curr_mask = vis_dict['meta_info.mask_timesteps'][b]
            mask_right_valid.append(right_valid[b][curr_mask])
            mask_left_valid.append(left_valid[b][curr_mask])
            mask_pred_vertices_r_cam.append(pred_vertices_r_cam[b][curr_mask])
            mask_pred_vertices_l_cam.append(pred_vertices_l_cam[b][curr_mask])
        
        right_valid = mask_right_valid
        left_valid = mask_left_valid
        pred_vertices_r_cam = mask_pred_vertices_r_cam
        pred_vertices_l_cam = mask_pred_vertices_l_cam
    
    K = vis_dict["meta_info.intrinsics"]
    if len(K.shape) == 4:
        K = K[:, 0] # B x T x 3 x 3, same for all timesteps

    # rendering
    im_list = []
    for idx in range(min(len(image_ids), max_examples)):
        r_valid = right_valid[idx]
        l_valid = left_valid[idx]
        K_i = K[idx]
        image_id = image_ids[idx]

        # render gt
        image_list = []
        if 'vis.goal_images' in vis_dict:
            image_list.append(np.vstack([images[idx], goal_images[idx]]))
        else:
            image_list.append(images[idx])
        image_pred = images[idx].copy()

        ts = pred_vertices_r_cam[idx].shape[0]
        for t in range(ts):
            # render pred
            image_pred = visualize_single(
                renderer,
                pred_vertices_r_cam[idx][t],
                pred_vertices_l_cam[idx][t],
                mano_faces_r,
                mano_faces_l,
                None,
                None,
                True,
                True,
                K_i,
                np.ones_like(images[idx].copy()),
                only_hands,
            )
            
            image_list.append(image_pred)

        im_list.append(
            {
                "fig_name": f"{image_id}",
                "im": image_list,
                "inp_img": images[idx],
            }
        )
    return im_list


def future_motion_2d(renderer, vis_dict, max_examples, only_hands, return_pil=False):
    # render meshes

    # unpack data
    image_ids = vis_dict["vis.image_ids"]
    right_valid = vis_dict["targets.future_valid_r"].bool()
    left_valid = vis_dict["targets.future_valid_l"].bool()
    images = vis_dict["vis.images"].permute(0, 2, 3, 1).numpy()
    gt_vertices_r_cam = vis_dict["targets.future.j2d.r"]
    gt_vertices_l_cam = vis_dict["targets.future.j2d.l"]
    pred_vertices_r_cam = vis_dict["pred.future.j2d.r"]
    pred_vertices_l_cam = vis_dict["pred.future.j2d.l"]
    pred_v3d_r_cam = vis_dict["pred.future.v3d.cam.r"]
    pred_v3d_l_cam = vis_dict["pred.future.v3d.cam.l"]

    bz = gt_vertices_r_cam.shape[0]
    if 'meta_info.mask_timesteps' in vis_dict:
        mask_gt_vertices_r_cam = []
        mask_gt_vertices_l_cam = []
        mask_right_valid = []
        mask_left_valid = []
        mask_pred_vertices_r_cam = []
        mask_pred_vertices_l_cam = []
        mask_pred_v3d_r_cam = []
        mask_pred_v3d_l_cam = []
        bz, ts, num_verts = gt_vertices_r_cam.shape[:3]
        for b in range(bz):
            curr_mask = vis_dict['meta_info.mask_timesteps'][b]
            mask_gt_vertices_r_cam.append(gt_vertices_r_cam[b][curr_mask])
            mask_gt_vertices_l_cam.append(gt_vertices_l_cam[b][curr_mask])
            mask_right_valid.append(right_valid[b][curr_mask])
            mask_left_valid.append(left_valid[b][curr_mask])
            mask_pred_vertices_r_cam.append(pred_vertices_r_cam[b][curr_mask])
            mask_pred_vertices_l_cam.append(pred_vertices_l_cam[b][curr_mask])
            mask_pred_v3d_r_cam.append(pred_v3d_r_cam[b][curr_mask])
            mask_pred_v3d_l_cam.append(pred_v3d_l_cam[b][curr_mask])
        gt_vertices_r_cam = mask_gt_vertices_r_cam
        gt_vertices_l_cam = mask_gt_vertices_l_cam
        right_valid = mask_right_valid
        left_valid = mask_left_valid
        pred_vertices_r_cam = mask_pred_vertices_r_cam
        pred_vertices_l_cam = mask_pred_vertices_l_cam
        pred_v3d_r_cam = mask_pred_v3d_r_cam
        pred_v3d_l_cam = mask_pred_v3d_l_cam

    img_res = max(images.shape[1:3])
    # clip vertices to image size using torch clip
    for b in range(bz):
        gt_vertices_r_cam[b] = torch.clip(gt_vertices_r_cam[b], 0, img_res)
        gt_vertices_l_cam[b] = torch.clip(gt_vertices_l_cam[b], 0, img_res)
        pred_vertices_r_cam[b] = torch.clip(pred_vertices_r_cam[b], 0, img_res)
        pred_vertices_l_cam[b] = torch.clip(pred_vertices_l_cam[b], 0, img_res)

    mano_faces_r = vis_dict["meta_info.mano.faces.r"]
    mano_faces_l = vis_dict["meta_info.mano.faces.l"]
    K = vis_dict["meta_info.intrinsics"]
    if len(K.shape) == 4:
        K = K[:, 0] # B x T x 3 x 3, same for all timesteps
    
    im_list = []
    for idx in range(min(len(image_ids), max_examples)):
        r_valid = right_valid[idx]
        l_valid = left_valid[idx]
        image_id = image_ids[idx]
        image = images[idx]
        
        image_list = []
        # plot GT 2D keypoints
        im_gt = visualize_multiple_kps(
            gt_vertices_r_cam[idx],
            gt_vertices_l_cam[idx],
            None,
            None,
            r_valid,
            l_valid,
            None,
            image,
            only_hands,
            flag="GT",
            only_obj=False,
            fading_factor=0.9,
        )
        image_list.append(im_gt)

        # plot pred 2D keypoints
        im_pred = visualize_multiple_kps(
            pred_vertices_r_cam[idx],
            pred_vertices_l_cam[idx],
            None,
            None,
            r_valid,
            l_valid,
            None,
            image,
            only_hands,
            flag="Pred",
            only_obj=False,
            fading_factor=0.9,
        )
        image_list.append(im_pred)

        # also render the predicted 3D meshes
        K_i = K[idx]
        im_render = visualize_multiple(
                renderer,
                pred_v3d_r_cam[idx],
                pred_v3d_l_cam[idx],
                mano_faces_r,
                mano_faces_l,
                None,
                None,
                r_valid,
                l_valid,
                K_i,
                image.copy(),
                only_hands,
            )
        
        image_list.append(im_render)

        if not return_pil:
            # stack images into one
            image_pred = vis_utils.im_list_to_plt(
                image_list,
                figsize=(15, 8),
                title_list=["GT", "Pred", "Render pred"],
            )
    
        im_list.append(
            {
                # "fig_name": f"{image_id}__rend_rvalid={r_valid}, lvalid={l_valid} ",
                "fig_name": f"{image_id}",
                "im": image_pred,
            }
        )
    return im_list


def visualize_all(vis_dict, max_examples, renderer, postfix, no_tqdm, only_hands=False, return_pil=False):
    # unpack
    image_ids = [
        "/".join(key.split("/")[-5:]).replace(".jpg", "")
        for key in vis_dict["meta_info.imgname"]
    ]
    images = denormalize_images(vis_dict["inputs.img"])
    if images.shape[-1] != renderer.img_res: # for ig_hands models
        images = F.interpolate(images, size=renderer.img_res, mode="bilinear", align_corners=True)

    vis_dict.pop("inputs.img", None)
    vis_dict["vis.images"] = images
    vis_dict["vis.image_ids"] = image_ids

    # render 3D meshes
    im_list = visualize_rends(renderer, vis_dict, max_examples, only_hands, return_pil=return_pil)

    is_j2d_present = False
    # check if j2d is present in any key in predictions
    for key in vis_dict.keys():
        if 'pred' in key and "j2d" in key:
            is_j2d_present = True
            break
    if not return_pil and is_j2d_present: # only visualize keypoints if j2d is present
        # visualize keypoints
        im_list_kp_gt = visualize_kps(vis_dict, "targets", max_examples, only_hands)
        im_list_kp_pred = visualize_kps(vis_dict, "pred", max_examples, only_hands)

        # concat side by side pred and gt
        for im_gt, im_pred in zip(im_list_kp_gt, im_list_kp_pred):
            im = {
                "fig_name": im_gt["fig_name"],
                "im": vis_utils.concat_pil_images([im_gt["im"], im_pred["im"]]),
            }
            im_list.append(im)

    # post fix image list
    im_list_postfix = []
    for im in im_list:
        im["fig_name"] += postfix
        im_list_postfix.append(im)

    return im_list


def visualize_motion(vis_dict, max_examples, renderer, postfix, no_tqdm, only_hands=False, return_pil=False):
    # unpack
    batch_size = len(vis_dict["meta_info.imgname"])
    # for each element in the batch, take the last timestep
    irrelevant_keys = ['meta_info.lengths', 'meta_info.mask_timesteps']
    for k, v in vis_dict.items():
        if 'future' in k:
            continue
        if isinstance(v, torch.Tensor) and k not in irrelevant_keys:
            vis_dict.overwrite(k, v[:,-1])
        elif isinstance(v, list):
            new_list = []
            for item in v:
                new_list.append(item[-1])
            vis_dict.overwrite(k, new_list)
    
    image_ids = [
        "/".join(key.split("/")[-5:]).replace(".jpg", "")
        for key in vis_dict["meta_info.imgname"]
    ]
    images = denormalize_images(vis_dict["inputs.img"])
    if images.shape[-1] != renderer.img_res: # for ig_hands models
        images = F.interpolate(images, size=renderer.img_res, mode="bilinear", align_corners=True)

    vis_dict.pop("inputs.img", None)
    vis_dict["vis.images"] = images
    vis_dict["vis.image_ids"] = image_ids

    if 'inputs.goal_img' in vis_dict:
        goal_images = denormalize_images(vis_dict["inputs.goal_img"])
        if goal_images.shape[-1] != renderer.img_res:
            goal_images = F.interpolate(goal_images, size=renderer.img_res, mode="bilinear", align_corners=True)
        vis_dict.pop("inputs.goal_img", None)
        vis_dict["vis.goal_images"] = goal_images

    im_list = []
    # render 3D meshes
    motion_list = future_motion_rends(renderer, vis_dict, max_examples, only_hands, return_pil=return_pil)
    for im in motion_list:
        im["fig_name"] += postfix
    # combine im_list and motion_list
    im_list = im_list + motion_list

    return im_list


def visualize_motion_viz(vis_dict, max_examples, renderer, postfix, no_tqdm, only_hands=False, return_pil=True):
    # unpack
    # batch_size = len(vis_dict["meta_info.imgname"])
    batch_size = vis_dict['meta_info.intrinsics'].shape[0]
    # for each element in the batch, take the last timestep
    irrelevant_keys = ['meta_info.lengths', 'meta_info.mask_timesteps']
    for k, v in vis_dict.items():
        if 'future' in k:
            continue
        if isinstance(v, torch.Tensor) and k not in irrelevant_keys:
            vis_dict.overwrite(k, v[:,-1])
        elif isinstance(v, list):
            new_list = []
            for item in v:
                new_list.append(item[-1])
            vis_dict.overwrite(k, new_list)
    
    image_ids = [
        "/".join(key.split("/")[-5:]).replace(".jpg", "")
        for key in vis_dict["meta_info.imgname"]
    ]
    images = denormalize_images(vis_dict["inputs.img"])
    if images.shape[-1] != renderer.img_res: # for ig_hands models
        images = F.interpolate(images, size=renderer.img_res, mode="bilinear", align_corners=True)

    vis_dict.pop("inputs.img", None)
    vis_dict["vis.images"] = images
    vis_dict["vis.image_ids"] = image_ids

    if 'inputs.goal_img' in vis_dict:
        goal_images = denormalize_images(vis_dict["inputs.goal_img"])
        if goal_images.shape[-1] != renderer.img_res:
            goal_images = F.interpolate(goal_images, size=renderer.img_res, mode="bilinear", align_corners=True)
        vis_dict.pop("inputs.goal_img", None)
        vis_dict["vis.goal_images"] = goal_images

    im_list = []
    # render 3D meshes
    motion_list = future_motion_rends_viz(renderer, vis_dict, max_examples, only_hands, return_pil=return_pil)
    for im in motion_list:
        im["fig_name"] += postfix
    # combine im_list and motion_list
    im_list = im_list + motion_list

    return im_list


def visualize_motion_video(vis_dict, max_examples, renderer, postfix, no_tqdm, only_hands=False, return_pil=True):
    # unpack
    # batch_size = len(vis_dict["meta_info.imgname"])
    batch_size = vis_dict['meta_info.intrinsics'].shape[0]
    # for each element in the batch, take the last timestep
    irrelevant_keys = ['meta_info.lengths', 'meta_info.mask_timesteps']
    for k, v in vis_dict.items():
        if 'future' in k:
            continue
        if isinstance(v, torch.Tensor) and k not in irrelevant_keys:
            vis_dict.overwrite(k, v[:,-1])
        elif isinstance(v, list):
            new_list = []
            for item in v:
                new_list.append(item[-1])
            vis_dict.overwrite(k, new_list)
    
    image_ids = [
        "/".join(key.split("/")[-5:]).replace(".jpg", "")
        for key in vis_dict["meta_info.imgname"]
    ]
    images = denormalize_images(vis_dict["inputs.img"])
    if images.shape[-1] != renderer.img_res: # for ig_hands models
        images = F.interpolate(images, size=renderer.img_res, mode="bilinear", align_corners=True)

    vis_dict.pop("inputs.img", None)
    vis_dict["vis.images"] = images
    vis_dict["vis.image_ids"] = image_ids

    if 'inputs.goal_img' in vis_dict:
        goal_images = denormalize_images(vis_dict["inputs.goal_img"])
        if goal_images.shape[-1] != renderer.img_res:
            goal_images = F.interpolate(goal_images, size=renderer.img_res, mode="bilinear", align_corners=True)
        vis_dict.pop("inputs.goal_img", None)
        vis_dict["vis.goal_images"] = goal_images

    im_list = []
    # render 3D meshes
    motion_list = future_motion_rends_video(renderer, vis_dict, max_examples, only_hands, return_pil=return_pil)
    for im in motion_list:
        im["fig_name"] += postfix
    # combine im_list and motion_list
    im_list = im_list + motion_list

    return im_list


def visualize_motion_video_views(vis_dict, max_examples, renderer, postfix, no_tqdm, only_hands=False, return_pil=True):
    # unpack
    # batch_size = len(vis_dict["meta_info.imgname"])
    batch_size = vis_dict['meta_info.intrinsics'].shape[0]
    # for each element in the batch, take the last timestep
    irrelevant_keys = ['meta_info.lengths', 'meta_info.mask_timesteps']
    for k, v in vis_dict.items():
        if 'future' in k:
            continue
        if isinstance(v, torch.Tensor) and k not in irrelevant_keys:
            vis_dict.overwrite(k, v[:,-1])
        elif isinstance(v, list):
            new_list = []
            for item in v:
                new_list.append(item[-1])
            vis_dict.overwrite(k, new_list)
    
    image_ids = [
        "/".join(key.split("/")[-5:]).replace(".jpg", "")
        for key in vis_dict["meta_info.imgname"]
    ]
    images = denormalize_images(vis_dict["inputs.img"])
    if images.shape[-1] != renderer.img_res: # for ig_hands models
        images = F.interpolate(images, size=renderer.img_res, mode="bilinear", align_corners=True)

    vis_dict.pop("inputs.img", None)
    vis_dict["vis.images"] = images
    vis_dict["vis.image_ids"] = image_ids

    if 'inputs.goal_img' in vis_dict:
        goal_images = denormalize_images(vis_dict["inputs.goal_img"])
        if goal_images.shape[-1] != renderer.img_res:
            goal_images = F.interpolate(goal_images, size=renderer.img_res, mode="bilinear", align_corners=True)
        vis_dict.pop("inputs.goal_img", None)
        vis_dict["vis.goal_images"] = goal_images

    im_list = []
    # render 3D meshes
    motion_list = future_motion_rends_video_views(renderer, vis_dict, max_examples, only_hands, return_pil=return_pil)
    for im in motion_list:
        im["fig_name"] += postfix
    # combine im_list and motion_list
    im_list = im_list + motion_list

    return im_list


def visualize_motion_lifting(vis_dict, max_examples, renderer, postfix, no_tqdm, only_hands=False, return_pil=True):
    # unpack
    # batch_size = len(vis_dict["meta_info.imgname"])
    batch_size = vis_dict['meta_info.intrinsics'].shape[0]
    # for each element in the batch, take the last timestep
    irrelevant_keys = ['meta_info.lengths', 'meta_info.mask_timesteps']
    for k, v in vis_dict.items():
        if 'future' in k:
            continue
        if isinstance(v, torch.Tensor) and k not in irrelevant_keys:
            vis_dict.overwrite(k, v[:,-1])
        elif isinstance(v, list):
            new_list = []
            for item in v:
                new_list.append(item[-1])
            vis_dict.overwrite(k, new_list)
    
    image_ids = [
        "/".join(key.split("/")[-5:]).replace(".jpg", "")
        for key in vis_dict["meta_info.imgname"]
    ]
    images = denormalize_images(vis_dict["inputs.img"])
    if images.shape[-1] != renderer.img_res: # for ig_hands models
        images = F.interpolate(images, size=renderer.img_res, mode="bilinear", align_corners=True)

    vis_dict.pop("inputs.img", None)
    vis_dict["vis.images"] = images
    vis_dict["vis.image_ids"] = image_ids

    if 'inputs.goal_img' in vis_dict:
        goal_images = denormalize_images(vis_dict["inputs.goal_img"])
        if goal_images.shape[-1] != renderer.img_res:
            goal_images = F.interpolate(goal_images, size=renderer.img_res, mode="bilinear", align_corners=True)
        vis_dict.pop("inputs.goal_img", None)
        vis_dict["vis.goal_images"] = goal_images

    im_list = []
    # render 3D meshes
    motion_list = future_motion_rends_lifting(renderer, vis_dict, max_examples, only_hands, return_pil=return_pil)
    for im in motion_list:
        im["fig_name"] += postfix
    # combine im_list and motion_list
    im_list = im_list + motion_list

    return im_list


def visualize_motion_2d(vis_dict, max_examples, renderer, postfix, no_tqdm, only_hands=False, return_pil=False):
    # unpack
    batch_size = len(vis_dict["meta_info.imgname"])
    # for each element in the batch, take the last timestep
    irrelevant_keys = ['meta_info.lengths', 'meta_info.mask_timesteps']
    for k, v in vis_dict.items():
        if 'future' in k:
            continue
        if isinstance(v, torch.Tensor) and k not in irrelevant_keys:
            vis_dict.overwrite(k, v[:,-1])
        elif isinstance(v, list):
            new_list = []
            for item in v:
                new_list.append(item[-1])
            vis_dict.overwrite(k, new_list)
    
    image_ids = [
        "/".join(key.split("/")[-5:]).replace(".jpg", "")
        for key in vis_dict["meta_info.imgname"]
    ]
    images = denormalize_images(vis_dict["inputs.img"])
    if images.shape[-1] != renderer.img_res: # for ig_hands models
        images = F.interpolate(images, size=renderer.img_res, mode="bilinear", align_corners=True)

    vis_dict.pop("inputs.img", None)
    vis_dict["vis.images"] = images
    vis_dict["vis.image_ids"] = image_ids

    im_list = []
    motion_list = future_motion_2d(renderer, vis_dict, max_examples, only_hands, return_pil=return_pil)
    for im in motion_list:
        im["fig_name"] += postfix
    # combine im_list and motion_list
    im_list = im_list + motion_list

    return im_list


def visualize_obj(vis_dict, max_examples, renderer, postfix, no_tqdm, only_hands=False, return_pil=False):
    # unpack
    image_ids = [
        "/".join(key.split("/")[-5:]).replace(".jpg", "")
        for key in vis_dict["meta_info.imgname"]
    ]
    images = denormalize_images(vis_dict["inputs.img"])
    if images.shape[-1] != renderer.img_res: # for ig_hands models
        images = F.interpolate(images, size=renderer.img_res, mode="bilinear", align_corners=True)

    vis_dict.pop("inputs.img", None)
    vis_dict["vis.images"] = images
    vis_dict["vis.image_ids"] = image_ids

    # render 3D meshes
    im_list = visualize_rends_obj(renderer, vis_dict, max_examples, only_hands, return_pil=return_pil)

    if not return_pil:
        # visualize keypoints
        im_list_kp_gt = visualize_kps(vis_dict, "targets", max_examples, only_hands)
        im_list_kp_pred = visualize_kps(vis_dict, "pred", max_examples, only_hands)

        # concat side by side pred and gt
        for im_gt, im_pred in zip(im_list_kp_gt, im_list_kp_pred):
            im = {
                "fig_name": im_gt["fig_name"],
                "im": vis_utils.concat_pil_images([im_gt["im"], im_pred["im"]]),
            }
            im_list.append(im)

    # post fix image list
    im_list_postfix = []
    for im in im_list:
        im["fig_name"] += postfix
        im_list_postfix.append(im)

    return im_list

def visualize_rends_obj(renderer, vis_dict, max_examples, only_hands, return_pil):
    # render meshes

    # unpack data
    image_ids = vis_dict["vis.image_ids"]
    right_valid = vis_dict["targets.right_valid"].bool()
    left_valid = vis_dict["targets.left_valid"].bool()
    images = vis_dict["vis.images"].permute(0, 2, 3, 1).numpy()
    gt_vertices_r_cam = vis_dict["targets.mano.v3d.cam.r"]
    gt_vertices_l_cam = vis_dict["targets.mano.v3d.cam.l"]
    mano_faces_r = vis_dict["meta_info.mano.faces.r"]
    mano_faces_l = vis_dict["meta_info.mano.faces.l"]
    pred_vertices_r_cam = vis_dict["pred.mano.v3d.cam.r"]
    pred_vertices_l_cam = vis_dict["pred.mano.v3d.cam.l"]

    # object
    gt_obj_v_cam = unpad_vtensor(
        vis_dict["targets.object.v.cam"], vis_dict["targets.object.v_len"]
    )  # meter
    pred_obj_v_cam = unpad_vtensor(
        vis_dict["pred.object.v.cam"], vis_dict["pred.object.v_len"]
    )
    pred_obj_f = unpad_vtensor(vis_dict["pred.object.f"], vis_dict["pred.object.f_len"])
    K = vis_dict["meta_info.intrinsics"]

    # rendering
    im_list = []
    for idx in range(min(len(image_ids), max_examples)):
        r_valid = right_valid[idx]
        l_valid = left_valid[idx]
        K_i = K[idx]
        image_id = image_ids[idx]

        # render gt
        image_list = []
        image_list.append(images[idx])
        # image_gt = visualize_rend(
        #     renderer,
        #     gt_vertices_r_cam[idx],
        #     gt_vertices_l_cam[idx],
        #     mano_faces_r,
        #     mano_faces_l,
        #     gt_obj_v_cam[idx],
        #     pred_obj_f[idx],
        #     r_valid,
        #     l_valid,
        #     K_i,
        #     images[idx],
        #     only_hands,
        #     only_obj=True,
        # )
        # image_list.append(image_gt)

        # render pred
        image_pred = visualize_rend(
            renderer,
            pred_vertices_r_cam[idx],
            pred_vertices_l_cam[idx],
            mano_faces_r,
            mano_faces_l,
            pred_obj_v_cam[idx],
            pred_obj_f[idx],
            r_valid,
            l_valid,
            K_i,
            images[idx],
            only_hands,
            only_obj=True,
        )
        image_list.append(image_pred)

        if not return_pil:
            # stack images into one
            image_pred = vis_utils.im_list_to_plt(
                image_list,
                figsize=(15, 8),
                title_list=["input image", "GT", "pred w/ pred_cam_t"],
            )
        im_list.append(
            {
                "fig_name": f"{image_id}__rend_rvalid={r_valid}, lvalid={l_valid} ",
                "im": image_pred,
            }
        )
    return im_list


def visualize_gif(vis_dict, max_examples, renderer, postfix, no_tqdm, only_hands=False, return_pil=False, **kwargs):
    # unpack
    image_ids = [
        "/".join(key.split("/")[-5:]).replace(".jpg", "")
        for key in vis_dict["meta_info.imgname"]
    ]
    images = denormalize_images(vis_dict["inputs.img"])
    if images.shape[-1] != renderer.img_res: # for ig_hands models
        images = F.interpolate(images, size=renderer.img_res, mode="bilinear", align_corners=True)

    vis_dict.pop("inputs.img", None)
    vis_dict["vis.images"] = images
    vis_dict["vis.image_ids"] = image_ids

    # render 3D meshes
    gif_list = visualize_rends_gifs(renderer, vis_dict, max_examples, only_hands, return_pil=return_pil, **kwargs) # for gifs

    return [], gif_list


def visualize_rends_gifs(renderer, vis_dict, max_examples, only_hands, return_pil=False, **kwargs):
    # render meshes

    # unpack data
    image_ids = vis_dict["vis.image_ids"]
    right_valid = vis_dict["targets.right_valid"].bool()
    left_valid = vis_dict["targets.left_valid"].bool()
    images = vis_dict["vis.images"].permute(0, 2, 3, 1).numpy()
    gt_vertices_r_cam = vis_dict["targets.mano.v3d.cam.r"]
    gt_vertices_l_cam = vis_dict["targets.mano.v3d.cam.l"]
    mano_faces_r = vis_dict["meta_info.mano.faces.r"]
    mano_faces_l = vis_dict["meta_info.mano.faces.l"]
    pred_vertices_r_cam = vis_dict["pred.mano.v3d.cam.r"]
    pred_vertices_l_cam = vis_dict["pred.mano.v3d.cam.l"]

    crop_viz = kwargs.get('crop_viz', False)
    if crop_viz:
        # bbox_r = vis_dict["inputs.r_bbox"] # this is need for rendering hamer predictions
        # bbox_l = vis_dict["inputs.l_bbox"]
        crop_img_r = vis_dict['crop_r'].numpy().astype(np.uint8)
        crop_img_l = vis_dict['crop_l'].numpy().astype(np.uint8)
        mano_faces_l = mano_faces_r # render using right mano and flip

    if not only_hands:
        # object
        gt_obj_v_cam = unpad_vtensor(
            vis_dict["targets.object.v.cam"], vis_dict["targets.object.v_len"]
        )  # meter
        pred_obj_v_cam = unpad_vtensor(
            vis_dict["pred.object.v.cam"], vis_dict["pred.object.v_len"]
        )
        pred_obj_f = unpad_vtensor(vis_dict["pred.object.f"], vis_dict["pred.object.f_len"])
    K = vis_dict["meta_info.intrinsics"]

    # rendering
    im_list = []
    for idx in range(min(len(image_ids), max_examples)):
        r_valid = right_valid[idx]
        l_valid = left_valid[idx]
        K_i = K[idx]
        image_id = image_ids[idx]

        if crop_viz:
            crop_r = Image.fromarray(crop_img_r[idx])
            crop_l = Image.fromarray(crop_img_l[idx])
            crop_r = crop_r.resize((renderer.img_res, renderer.img_res))
            crop_l = crop_l.resize((renderer.img_res, renderer.img_res))
            crop_r = np.array(crop_r) / 255
            crop_l = np.array(crop_l) / 255
            kwargs['crop_r'] = crop_r
            kwargs['crop_l'] = crop_l

        # render gt
        image_list = []
        image_list.append(images[idx])
        image_hand_r = visualize_rend_gif(
            renderer,
            pred_vertices_r_cam[idx],
            pred_vertices_l_cam[idx],
            mano_faces_r,
            mano_faces_l,
            None,
            None,
            r_valid,
            l_valid,
            K_i,
            images[idx],
            only_hands,
            only_right=True,
            **kwargs,
        )

        image_hand_l = visualize_rend_gif(
            renderer,
            pred_vertices_r_cam[idx],
            pred_vertices_l_cam[idx],
            mano_faces_r,
            mano_faces_l,
            None,
            None,
            r_valid,
            l_valid,
            K_i,
            images[idx],
            only_hands,
            only_left=True,
            **kwargs,
        )

        im_list.append(
            {
                "fig_name": f"{image_id}__rend_rvalid={r_valid}, lvalid={l_valid} ",
                "im_right": image_hand_r,
                "im_left": image_hand_l,
            }
        )
    return im_list

def scale_box(bbox, scale=1.2, res=224):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w >= res-4 or h >= res-4:
        return bbox

    x1 -= w*(scale-1)/2
    y1 -= h*(scale-1)/2
    x2 += w*(scale-1)/2
    y2 += h*(scale-1)/2
    # clip the values in [0,res]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(res, x2)
    y2 = min(res, y2)
    return [x1, y1, x2, y2]

def visualize_rend_gif(
    renderer,
    vertices_r,
    vertices_l,
    mano_faces_r,
    mano_faces_l,
    vertices_o,
    faces_o,
    r_valid,
    l_valid,
    K,
    img,
    only_hands,
    only_obj=False,
    only_right=False,
    only_left=False,
    **kwargs,
):
    # render 3d meshes
    mesh_r = Mesh(v=vertices_r, f=mano_faces_r)
    mesh_l = Mesh(v=vertices_l, f=mano_faces_l)
    if not only_hands:
        mesh_o = Mesh(v=thing.thing2np(vertices_o), f=thing.thing2np(faces_o))

    # render only valid meshes
    meshes = []
    mesh_names = []
    if r_valid and not only_left:
        meshes.append(mesh_r)
        mesh_names.append("right")

    if l_valid and not only_right:
        meshes.append(mesh_l)
        mesh_names.append("left")
    if not only_hands:
        meshes = meshes + [mesh_o]
        mesh_names = mesh_names + ["object"]

    if only_obj:
        meshes = [mesh_o]
        mesh_names = ["object"]

    materials = [color2material(mesh_color_dict[name], metallicFactor=0.5) for name in mesh_names]

    if len(meshes) > 0:
        # render in image space
        if kwargs.get('crop_viz', False):
            render_img_img = renderer.render_meshes_pose(
                cam_transl=None,
                meshes=meshes,
                image=kwargs['crop_r'] if only_right else kwargs['crop_l'],
                materials=materials,
                sideview_angle=None,
                K=K,
            )
        else:
            render_img_img = renderer.render_meshes_pose(
                cam_transl=None,
                meshes=meshes,
                image=img,
                materials=materials,
                sideview_angle=None,
                K=K,
            )
        render_img_list = [render_img_img]

        cam_transl = np.zeros((3,))
        intrx = K.clone()
        # if len(meshes) > 0:
        assert len(meshes) == 1 # only one hand at a time
        verts = meshes[0].v.copy()
        cam_transl[0] = -verts[:,0].mean() # different convention?
        cam_transl[1] = -verts[:,1].mean() # different convention?
        z = verts[:,2].mean()
        scale_factor = z / 0.2 # rendering depth
        if kwargs.get('scale_factor', None) is not None:
            scale_factor = kwargs['scale_factor']
        intrx[0,0] *= scale_factor
        intrx[1,1] *= scale_factor

        # render rotated meshes
        for angle in list(np.linspace(0, 360, 21)):
            render_img_angle = renderer.render_meshes_pose(
                cam_transl=cam_transl,
                meshes=meshes,
                image=None,
                materials=materials,
                sideview_angle=angle,
                K=intrx,
            )
            render_img_list.append(render_img_angle)

        # render only hand in the camera frame
        render_img_hand = renderer.render_meshes_pose(
                cam_transl=None,
                meshes=meshes,
                image=None,
                materials=materials,
                sideview_angle=None,
                K=K,
            )
        render_img_list.append(render_img_hand)

        # cat all images
        render_img = np.concatenate(render_img_list, axis=0)
        return render_img

    else:
        # dummy image
        render_img = np.concatenate([img] * 22, axis=0) # 21 rotations + 1 hand only in camera frame
        return render_img