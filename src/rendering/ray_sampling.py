"""
射線採樣模組

實現 NeRF 的射線採樣功能：
- 分層採樣
- 重要性採樣
- 射線生成
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List


def stratified_sampling(
    near: torch.Tensor,
    far: torch.Tensor,
    n_samples: int,
    perturb: bool = True
) -> torch.Tensor:
    """
    分層採樣
    
    Args:
        near: [batch] 近平面距離
        far: [batch] 遠平面距離
        n_samples: 採樣點數量
        perturb: 是否添加擾動
        
    Returns:
        z_vals: [batch, n_samples] 採樣點的深度值
    """
    # 創建均勻間隔的採樣點
    t_vals = torch.linspace(0., 1., steps=n_samples, device=near.device)
    t_vals = t_vals.expand(list(near.shape) + [n_samples])
    
    # 添加擾動
    if perturb:
        mids = .5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand(t_vals.shape, device=near.device)
        t_vals = lower + (upper - lower) * t_rand
    
    # 轉換到實際深度範圍
    z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals
    
    return z_vals


def importance_sampling(
    z_vals: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    perturb: bool = True
) -> torch.Tensor:
    """
    重要性採樣
    
    Args:
        z_vals: [batch, n_samples] 原始採樣點的深度值
        weights: [batch, n_samples] 採樣點權重
        n_samples: 新的採樣點數量
        perturb: 是否添加擾動
        
    Returns:
        z_vals_fine: [batch, n_samples] 新的採樣點深度值
    """
    # 計算累積分布函數
    cdf = torch.cumsum(weights, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    
    # 均勻採樣
    u = torch.linspace(0., 1., steps=n_samples, device=z_vals.device)
    u = u.expand(list(z_vals.shape[:-1]) + [n_samples])
    
    # 反轉 CDF 得到新的採樣點
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)
    
    # 線性插值
    cdf_g = torch.gather(cdf, -1, inds_g)
    z_vals_g = torch.gather(z_vals, -1, inds_g)
    
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    z_vals_fine = z_vals_g[..., 0] + t * (z_vals_g[..., 1] - z_vals_g[..., 0])
    
    return z_vals_fine


class RaySampler:
    """
    射線採樣器
    
    生成和管理射線採樣
    """
    
    def __init__(self, config: dict):
        """
        初始化射線採樣器
        
        Args:
            config: 配置字典，包含：
                - n_samples: 每條射線的採樣點數量
                - perturb: 是否添加擾動
                - n_fine: 細採樣的點數量
        """
        self.n_samples = config.get('n_samples', 64)
        self.perturb = config.get('perturb', True)
        self.n_fine = config.get('n_fine', 128)
    
    def get_rays(
        self,
        H: int,
        W: int,
        focal: float,
        c2w: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成射線
        
        Args:
            H: 圖像高度
            W: 圖像寬度
            focal: 焦距
            c2w: [4, 4] 相機到世界座標的變換矩陣
            
        Returns:
            rays_o: [H*W, 3] 射線起點
            rays_d: [H*W, 3] 射線方向
        """
        # 生成像素座標
        i, j = torch.meshgrid(
            torch.linspace(0, W-1, W, device=c2w.device),
            torch.linspace(0, H-1, H, device=c2w.device)
        )
        i = i.t()
        j = j.t()
        
        # 轉換到相機座標系
        dirs = torch.stack([
            (i - W*.5) / focal,
            -(j - H*.5) / focal,
            -torch.ones_like(i)
        ], -1)
        
        # 轉換到世界座標系
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        
        return rays_o, rays_d
    
    def sample_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        採樣射線上的點
        
        Args:
            rays_o: [batch, 3] 射線起點
            rays_d: [batch, 3] 射線方向
            near: 近平面距離
            far: 遠平面距離
            
        Returns:
            z_vals: [batch, n_samples] 採樣點的深度值
            points: [batch, n_samples, 3] 採樣點的世界座標
            viewdirs: [batch, 3] 標準化的視角方向
        """
        # 生成採樣點
        z_vals = stratified_sampling(
            torch.full_like(rays_o[..., 0], near),
            torch.full_like(rays_o[..., 0], far),
            self.n_samples,
            self.perturb
        )
        
        # 計算採樣點的世界座標
        points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        
        # 標準化視角方向
        viewdirs = F.normalize(rays_d, dim=-1)
        
        return z_vals, points, viewdirs
    
    def sample_fine_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        z_vals: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        在重要區域進行細採樣
        
        Args:
            rays_o: [batch, 3] 射線起點
            rays_d: [batch, 3] 射線方向
            z_vals: [batch, n_samples] 原始採樣點的深度值
            weights: [batch, n_samples] 採樣點權重
            
        Returns:
            z_vals_fine: [batch, n_fine] 新的採樣點深度值
            points_fine: [batch, n_fine, 3] 新的採樣點世界座標
            viewdirs: [batch, 3] 標準化的視角方向
        """
        # 重要性採樣
        z_vals_fine = importance_sampling(
            z_vals, weights, self.n_fine, self.perturb
        )
        
        # 計算新的採樣點世界座標
        points_fine = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_fine[..., :, None]
        
        # 標準化視角方向
        viewdirs = F.normalize(rays_d, dim=-1)
        
        return z_vals_fine, points_fine, viewdirs 