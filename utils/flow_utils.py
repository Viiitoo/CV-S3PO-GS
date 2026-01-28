"""
光流工具函数
用于初始化和计算光流
"""

import torch
import torch.nn.functional as F

# 延迟导入gmflow，避免在未启用光流时也需要yacs依赖
_gmflow_available = None
_gmflow_cfg = None
_gmflow_build = None

def _check_gmflow_available():
    """检查gmflow是否可用"""
    global _gmflow_available, _gmflow_cfg, _gmflow_build
    if _gmflow_available is None:
        try:
            from gmflow.config import get_cfg as get_gmflow_cfg
            from gmflow.gmflow import build_gmflow
            _gmflow_cfg = get_gmflow_cfg
            _gmflow_build = build_gmflow
            _gmflow_available = True
        except ImportError as e:
            _gmflow_available = False
            raise ImportError(
                f"GMFlow模块不可用: {e}\n"
                "请安装yacs依赖: pip install yacs==0.1.8\n"
                "或者如果不需要光流功能，请在配置文件中设置 use_optical_flow: False"
            ) from e
    return _gmflow_available


def init_optical_flow_model(model_path=None):
    """
    初始化GMFlow光流模型
    
    Args:
        model_path: 模型权重路径，如果为None则使用默认路径
    
    Returns:
        flownet: 初始化好的GMFlow模型（已设置为eval模式）
    """
    # 检查gmflow是否可用
    _check_gmflow_available()
    
    # 获取GMFlow配置
    cfg = _gmflow_cfg()
    
    # 如果提供了模型路径，使用提供的路径；否则使用配置中的默认路径
    if model_path is not None:
        cfg.model = model_path
    
    # 构建GMFlow模型
    flownet = _gmflow_build(cfg)
    
    # 加载预训练权重
    checkpoint = torch.load(cfg.model, map_location='cpu')
    # 处理不同的checkpoint格式
    if 'model' in checkpoint:
        weights = checkpoint['model']
    else:
        weights = checkpoint
    
    # 加载权重
    flownet.load_state_dict(weights, strict=False)
    
    # 移动到GPU并设置为eval模式
    flownet = flownet.cuda()
    flownet.eval()
    
    return flownet


def compute_optical_flow(img1, img2, flownet):
    """
    计算两帧之间的光流
    
    Args:
        img1: 第一帧图像，形状为 (C, H, W)，值范围 [0, 1]
        img2: 第二帧图像，形状为 (C, H, W)，值范围 [0, 1]
        flownet: GMFlow模型
    
    Returns:
        flow: 光流张量，形状为 (2, H, W)
    """
    # 确保输入是tensor
    if not isinstance(img1, torch.Tensor):
        img1 = torch.from_numpy(img1).float()
    if not isinstance(img2, torch.Tensor):
        img2 = torch.from_numpy(img2).float()
    
    # 确保在正确的设备上
    device = next(flownet.parameters()).device
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    # 获取原始尺寸
    C, H, W = img1.shape
    
    # 添加batch维度: (C, H, W) -> (1, C, H, W)
    img1_batch = img1.unsqueeze(0)  # (1, C, H, W)
    img2_batch = img2.unsqueeze(0)  # (1, C, H, W)
    
    # GMFlow期望输入是[0, 255]范围，所以需要乘以255
    img1_batch = img1_batch * 255.0
    img2_batch = img2_batch * 255.0
    
    # 计算光流（模型内部会处理normalization和padding）
    with torch.no_grad():
        flow_preds = flownet(img1_batch, img2_batch)
    
    # flow_preds是一个列表，在eval模式下通常只有一个元素
    # 检查列表是否为空
    if len(flow_preds) == 0:
        raise RuntimeError(
            f"GMFlow返回空列表，可能是模型配置问题。\n"
            f"输入尺寸: {img1_batch.shape}, 模型状态: training={flownet.training}\n"
            f"请检查模型是否正确设置为eval模式：flownet.eval()"
        )
    
    # 取最后一个元素（最终预测），形状为 (1, 2, H_flow, W_flow)
    # 在eval模式下，通常只有一个元素；在training模式下，最后一个是最终预测
    # 注意：MotionGS使用flow_preds[0]，但为了兼容training模式，我们使用[-1]
    flow = flow_preds[-1]  # (1, 2, H_flow, W_flow)
    
    # 移除batch维度: (1, 2, H_flow, W_flow) -> (2, H_flow, W_flow)
    flow = flow.squeeze(0)  # (2, H_flow, W_flow)
    
    # 如果输出尺寸与输入尺寸不匹配，需要插值
    H_flow, W_flow = flow.shape[-2:]
    if H_flow != H or W_flow != W:
        # 使用双线性插值调整尺寸
        flow = flow.unsqueeze(0)  # (1, 2, H_flow, W_flow)
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        flow = flow.squeeze(0)  # (2, H, W)
        
        # 根据尺寸比例缩放光流值
        flow[0] *= W / W_flow  # x方向
        flow[1] *= H / H_flow  # y方向
    
    return flow  # (2, H, W)


def calculate_gs_flow(gs_per_pixel=None, weight_per_gs_pixel=None, next_conic_2D=None, 
                     conic_2D_inv=None, proj_2D=None, next_proj_2D=None, x_mu=None,
                     depth1=None, depth2=None, cam1=None, cam2=None):
    """
    计算GS Flow（高斯点的2D光流）
    
    支持两种模式：
    1. 完整模式（需要proj_2D、conic_2D等中间信息）：使用MotionGS的精确方法
    2. 简化模式（使用深度和相机参数）：通过深度图计算像素对应关系
    
    Args:
        # 完整模式参数（如果render函数返回这些信息）
        gs_per_pixel: 每个像素的高斯索引，形状为 (K, H, W)
        weight_per_gs_pixel: 每个像素的高斯权重，形状为 (K, H, W)
        next_conic_2D: 下一帧的2D协方差矩阵，形状为 (K, 3)
        conic_2D_inv: 当前帧的2D协方差逆矩阵，形状为 (K, 3)
        proj_2D: 当前帧的2D投影位置，形状为 (K, 2)
        next_proj_2D: 下一帧的2D投影位置，形状为 (K, 2)
        x_mu: 像素坐标，形状为 (K, H, W, 2)
        
        # 简化模式参数
        depth1: 当前帧深度图，形状为 (1, H, W) 或 (H, W)
        depth2: 下一帧深度图（可选，当前未使用）
        cam1: 当前帧相机对象，需要有intrinsic和extrinsic属性
        cam2: 下一帧相机对象，需要有intrinsic和extrinsic属性
    
    Returns:
        gs_flow: GS Flow张量，形状为 (2, H, W)
    """
    # 如果提供了完整模式的参数，使用精确方法
    if (gs_per_pixel is not None and weight_per_gs_pixel is not None and 
        next_conic_2D is not None and conic_2D_inv is not None and
        proj_2D is not None and next_proj_2D is not None and x_mu is not None):
        return _calculate_gs_flow_full(gs_per_pixel, weight_per_gs_pixel, next_conic_2D,
                                       conic_2D_inv, proj_2D, next_proj_2D, x_mu)
    
    # 否则使用简化模式（基于深度）
    elif depth1 is not None and cam1 is not None and cam2 is not None:
        return _calculate_gs_flow_simplified(depth1, cam1, cam2)
    
    else:
        raise ValueError("calculate_gs_flow需要提供完整模式或简化模式的参数")


def _calculate_gs_flow_full(gs_per_pixel, weight_per_gs_pixel, next_conic_2D, 
                            conic_2D_inv, proj_2D, next_proj_2D, x_mu):
    """
    完整模式的GS Flow计算（参考MotionGS）
    """
    conic_2D_inv = conic_2D_inv.detach()  # K 3
    
    gs_per_pixel = gs_per_pixel.long()  # K H W
    
    # 计算协方差矩阵的乘积
    conv_conv = torch.zeros([conic_2D_inv.shape[0], 2, 2], device=conic_2D_inv.device)  # K 2 2
    conv_conv[:, 0, 0] = next_conic_2D[:, 0] * conic_2D_inv[:, 0] + next_conic_2D[:, 1] * conic_2D_inv[:, 1]
    conv_conv[:, 0, 1] = next_conic_2D[:, 0] * conic_2D_inv[:, 1] + next_conic_2D[:, 1] * conic_2D_inv[:, 2]
    conv_conv[:, 1, 0] = next_conic_2D[:, 1] * conic_2D_inv[:, 0] + next_conic_2D[:, 2] * conic_2D_inv[:, 1]
    conv_conv[:, 1, 1] = next_conic_2D[:, 1] * conic_2D_inv[:, 1] + next_conic_2D[:, 2] * conic_2D_inv[:, 2]
    
    # 计算各向异性GS Flow
    conv_multi = (conv_conv[gs_per_pixel] @ x_mu.permute(0,2,3,1).unsqueeze(-1).detach()).squeeze()  # K H W 2
    flow_per_pixel = (conv_multi + next_proj_2D[gs_per_pixel] - proj_2D[gs_per_pixel].detach() - 
                     x_mu.permute(0,2,3,1).detach())  # K H W 2
    
    # 加权平均
    weight_per_gs_pixel = weight_per_gs_pixel / (weight_per_gs_pixel.sum(dim=0, keepdim=True) + 1e-7)  # K H W
    flow_gs = torch.einsum("khw, khwa -> ahw", [weight_per_gs_pixel.detach(), flow_per_pixel])  # 2 H W
    
    return flow_gs


def _calculate_gs_flow_simplified(depth1, cam1, cam2):
    """
    简化版本的GS Flow计算函数
    通过渲染两帧的深度图，计算像素对应关系来近似GS Flow
    
    注意：这是简化版本，因为当前render函数不返回proj_2D、conic_2D等中间信息。
    如需更精确的GS Flow，需要修改render函数使用MotionGS的rasterization版本。
    
    Args:
        depth1: 当前帧深度图，形状为 (1, H, W) 或 (H, W)
        cam1: 当前帧相机对象，需要有intrinsic和extrinsic属性
        cam2: 下一帧相机对象，需要有intrinsic和extrinsic属性
    
    Returns:
        gs_flow: GS Flow张量，形状为 (2, H, W)
    """
    from utils.warp_utils import BackprojectDepth, Project3D
    
    # 确保深度图格式正确
    if depth1.dim() == 2:
        depth1 = depth1.unsqueeze(0)  # (1, H, W)
    
    H, W = depth1.shape[-2:]
    
    # 使用BackprojectDepth和Project3D计算像素对应关系
    backprojdepth = BackprojectDepth(1, H, W).cuda()
    project3d = Project3D(1, H, W).cuda()
    
    # 反投影当前帧深度到3D点
    inv_K1 = torch.linalg.inv(cam1.intrinsic.cuda())[None]  # B 4 4
    points_3d = backprojdepth(depth1, inv_K1)  # B 4 HW
    
    # 投影3D点到下一帧
    K2 = cam2.intrinsic.cuda()[None]  # B 4 4
    T12 = torch.matmul(torch.linalg.inv(cam2.extrinsic.cuda()), 
                     cam1.extrinsic.cuda())[None]  # B 4 4
    _, pixel_coords = project3d(points_3d, K2, T12)  # B H W 2
    
    # 计算原始像素坐标
    pixel_coords = pixel_coords.permute(0, 3, 1, 2)  # B 2 H W
    ori_coords = backprojdepth.pix_coords.view(1, 3, H, W)[:, :2]  # B 2 H W
    
    # GS Flow = 下一帧投影位置 - 当前帧原始位置
    gs_flow = pixel_coords - ori_coords  # B 2 H W
    
    return gs_flow[0]  # 2 H W

