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

