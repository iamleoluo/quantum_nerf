"""
數學工具模組

提供 NeRF 訓練過程中的各種數學工具函數：
- 安全歸一化
- PSNR 計算
- 其他數學輔助函數
"""

import torch
import numpy as np
from typing import Union, Tuple


def safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    安全地對向量進行歸一化，避免除零錯誤
    
    Args:
        x: 輸入向量
        eps: 小數值，用於避免除零
        
    Returns:
        歸一化後的向量
    """
    norm = torch.norm(x, dim=-1, keepdim=True)
    return x / (norm + eps)


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0
) -> float:
    """
    計算峰值信噪比 (PSNR)
    
    Args:
        pred: 預測值
        target: 目標值
        max_val: 最大值範圍
        
    Returns:
        PSNR 值（單位：dB）
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5
) -> float:
    """
    計算結構相似性指數 (SSIM)
    
    Args:
        pred: 預測圖像 [B, C, H, W]
        target: 目標圖像 [B, C, H, W]
        window_size: 高斯窗口大小
        sigma: 高斯核標準差
        
    Returns:
        SSIM 值
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 創建高斯窗口
    window = torch.exp(-(torch.arange(window_size) - window_size//2)**2 / (2*sigma**2))
    window = window / window.sum()
    window = window.unsqueeze(0).unsqueeze(0)
    
    # 計算均值
    mu_pred = torch.nn.functional.conv2d(pred, window, padding=window_size//2)
    mu_target = torch.nn.functional.conv2d(target, window, padding=window_size//2)
    
    # 計算方差和協方差
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = torch.nn.functional.conv2d(pred**2, window, padding=window_size//2) - mu_pred_sq
    sigma_target_sq = torch.nn.functional.conv2d(target**2, window, padding=window_size//2) - mu_target_sq
    sigma_pred_target = torch.nn.functional.conv2d(pred*target, window, padding=window_size//2) - mu_pred_target
    
    # 計算 SSIM
    ssim_map = ((2*mu_pred_target + C1)*(2*sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1)*(sigma_pred_sq + sigma_target_sq + C2))
    
    return ssim_map.mean().item()


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: torch.nn.Module = None
) -> float:
    """
    計算感知相似度 (LPIPS)
    
    Args:
        pred: 預測圖像 [B, C, H, W]
        target: 目標圖像 [B, C, H, W]
        net: LPIPS 網絡（如果為 None，則使用默認網絡）
        
    Returns:
        LPIPS 值
    """
    if net is None:
        try:
            import lpips
            net = lpips.LPIPS(net='alex')
        except ImportError:
            print("警告：未安裝 lpips 包，無法計算 LPIPS")
            return 0.0
    
    with torch.no_grad():
        return net(pred, target).mean().item()


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor
) -> dict:
    """
    計算多個圖像質量指標
    
    Args:
        pred: 預測圖像
        target: 目標圖像
        
    Returns:
        包含多個指標的字典
    """
    metrics = {
        'psnr': compute_psnr(pred, target),
        'ssim': compute_ssim(pred, target),
        'lpips': compute_lpips(pred, target)
    }
    return metrics


def compute_ray_intersection(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    bounds: Tuple[float, float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    計算射線與邊界框的交點
    
    Args:
        ray_o: 射線起點 [N, 3]
        ray_d: 射線方向 [N, 3]
        bounds: 邊界框範圍 (min, max)
        
    Returns:
        near: 近交點距離 [N]
        far: 遠交點距離 [N]
    """
    # 計算與每個平面的交點
    t_min = (bounds[0] - ray_o) / (ray_d + 1e-8)
    t_max = (bounds[1] - ray_o) / (ray_d + 1e-8)
    
    # 處理方向為負的情況
    t_min, t_max = torch.min(t_min, t_max), torch.max(t_min, t_max)
    
    # 取所有平面的最大近點和最小遠點
    near = torch.max(t_min, dim=-1)[0]
    far = torch.min(t_max, dim=-1)[0]
    
    # 處理無交點的情況
    mask = far > near
    near = torch.where(mask, near, torch.zeros_like(near))
    far = torch.where(mask, far, torch.ones_like(far))
    
    return near, far 