import torch
import torch.nn.functional as F
import time
import json
from gaussian_splatting.utils.loss_utils import l1_loss

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]

def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    if config["Training"]["monocular"] and config["Dataset"]["depth_loss"]:
        # mono_depth 在非关键帧时可能为 None（延迟计算优化），此时回退到 RGB
        if viewpoint.mono_depth is not None:
            loss = get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)
        else:
            loss = get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    elif config["Training"]["monocular"]:
        loss = get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    else:
        loss = get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)

    # 位姿平滑正则化
    smooth_weight = config.get("Training", {}).get("pose_smooth_weight", 0.0)
    if smooth_weight > 0:
        loss = loss + smooth_weight * pose_smoothness_loss(viewpoint, None)

    return loss

def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    
    return l1.mean()

def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.mono_depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()

def get_loss_mapping(config, image,  viewpoint, depth=None, initialization=False, monodepth = True):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
        
    if config["Training"]["monocular"] and monodepth:
        return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)

def get_loss_mapping_rgb(config, image, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()

def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.mono_depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()

def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()

def flow_loss(flow_pred, flow_gt, height, width):
    """
    计算光流损失，归一化后使用Huber loss（对大运动更鲁棒）

    Args:
        flow_pred: 预测的光流，形状为 (2, H, W)
        flow_gt: 真实的光流，形状为 (2, H, W)
        height: 图像高度
        width: 图像宽度

    Returns:
        loss: Huber loss
    """
    flow_pred = flow_pred.clone()
    flow_gt = flow_gt.clone()

    flow_pred[0] /= width
    flow_pred[1] /= height

    flow_gt[0] /= width
    flow_gt[1] /= height

    # 使用Huber loss替代L1+clamp，对大运动不截断而是线性增长
    return F.smooth_l1_loss(flow_pred, flow_gt, beta=0.5)

def pose_smoothness_loss(viewpoint, prev_viewpoint):
    """
    位姿平滑正则化：约束相邻帧位姿变化的一致性，减少帧间抖动

    Args:
        viewpoint: 当前帧的Camera对象
        prev_viewpoint: 上一帧的Camera对象

    Returns:
        loss: 旋转和平移delta的L2正则
    """
    rot_delta = viewpoint.cam_rot_delta
    trans_delta = viewpoint.cam_trans_delta
    return rot_delta.pow(2).sum() + trans_delta.pow(2).sum()