"""
體積渲染器模組

實現 NeRF 的體積渲染功能：
- 體積渲染方程
- 透明度累積
- 顏色合成
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class VolumeRenderer:
    """
    體積渲染器
    
    實現 NeRF 的體積渲染方程
    """
    
    def __init__(self, config: dict):
        """
        初始化體積渲染器
        
        Args:
            config: 配置字典，包含：
                - white_bkgd: 是否使用白色背景
                - raw_noise_std: 原始密度噪聲標準差
        """
        self.white_bkgd = config.get('white_bkgd', True)
        self.raw_noise_std = config.get('raw_noise_std', 0.0)
    
    def render_volume(
        self,
        rgb: torch.Tensor,
        density: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        執行體積渲染
        
        Args:
            rgb: [batch, n_samples, 3] 預測的 RGB 顏色
            density: [batch, n_samples, 1] 預測的體積密度
            z_vals: [batch, n_samples] 採樣點的深度值
            rays_d: [batch, 3] 射線方向
            raw_noise_std: 密度噪聲標準差（可選）
            
        Returns:
            rgb_map: [batch, 3] 渲染的 RGB 顏色
            depth_map: [batch] 渲染的深度圖
            acc_map: [batch] 累積的不透明度
        """
        # 添加噪聲到密度（如果指定）
        if raw_noise_std is None:
            raw_noise_std = self.raw_noise_std
        
        if raw_noise_std > 0.:
            noise = torch.randn_like(density) * raw_noise_std
            density = density + noise
        
        # 計算相鄰採樣點之間的距離
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], -1)
        
        # 考慮射線方向
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        
        # 計算 alpha 值
        alpha = 1. - torch.exp(-density[..., 0] * dists)
        
        # 計算透射率
        transmittance = torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], -1), -1
        )[:, :-1]
        
        # 計算權重
        weights = alpha * transmittance
        
        # 計算 RGB 圖
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
        
        # 計算深度圖
        depth_map = torch.sum(weights * z_vals, dim=-1)
        
        # 計算累積不透明度
        acc_map = torch.sum(weights, dim=-1)
        
        # 如果使用白色背景，則混合背景色
        if self.white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])
        
        return rgb_map, depth_map, acc_map
    
    def render_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        z_vals: torch.Tensor,
        rgb: torch.Tensor,
        density: torch.Tensor,
        raw_noise_std: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        渲染單條射線
        
        Args:
            rays_o: [batch, 3] 射線起點
            rays_d: [batch, 3] 射線方向
            z_vals: [batch, n_samples] 採樣點的深度值
            rgb: [batch, n_samples, 3] 預測的 RGB 顏色
            density: [batch, n_samples, 1] 預測的體積密度
            raw_noise_std: 密度噪聲標準差（可選）
            
        Returns:
            rgb_map: [batch, 3] 渲染的 RGB 顏色
            depth_map: [batch] 渲染的深度圖
            acc_map: [batch] 累積的不透明度
        """
        return self.render_volume(rgb, density, z_vals, rays_d, raw_noise_std)
    
    def render_image(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        z_vals: torch.Tensor,
        rgb: torch.Tensor,
        density: torch.Tensor,
        raw_noise_std: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        渲染完整圖像
        
        Args:
            rays_o: [H*W, 3] 射線起點
            rays_d: [H*W, 3] 射線方向
            z_vals: [H*W, n_samples] 採樣點的深度值
            rgb: [H*W, n_samples, 3] 預測的 RGB 顏色
            density: [H*W, n_samples, 1] 預測的體積密度
            raw_noise_std: 密度噪聲標準差（可選）
            
        Returns:
            rgb_map: [H, W, 3] 渲染的 RGB 圖像
            depth_map: [H, W] 渲染的深度圖
            acc_map: [H, W] 累積的不透明度圖
        """
        rgb_map, depth_map, acc_map = self.render_rays(
            rays_o, rays_d, z_vals, rgb, density, raw_noise_std
        )
        
        # 重塑為圖像形狀
        H = int(torch.sqrt(torch.tensor(rays_o.shape[0])))
        W = H
        
        rgb_map = rgb_map.reshape(H, W, 3)
        depth_map = depth_map.reshape(H, W)
        acc_map = acc_map.reshape(H, W)
        
        return rgb_map, depth_map, acc_map 