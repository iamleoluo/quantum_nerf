"""
渲染工具模組

提供 NeRF 渲染過程中的輔助功能：
- 射線渲染
- 圖像渲染
- 相機參數處理
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from .volume_renderer import VolumeRenderer
from .ray_sampling import RaySampler


def render_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    network: torch.nn.Module,
    encoder: torch.nn.Module,
    renderer: VolumeRenderer,
    sampler: RaySampler,
    raw_noise_std: float = 0.0,
    use_viewdirs: bool = True,
    use_fine: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    渲染單條射線
    
    Args:
        rays_o: [batch, 3] 射線起點
        rays_d: [batch, 3] 射線方向
        near: 近平面距離
        far: 遠平面距離
        network: NeRF 網絡
        encoder: 位置編碼器
        renderer: 體積渲染器
        sampler: 射線採樣器
        raw_noise_std: 密度噪聲標準差
        use_viewdirs: 是否使用視角方向
        use_fine: 是否使用細採樣
        
    Returns:
        rgb_map: [batch, 3] 渲染的 RGB 顏色
        depth_map: [batch] 渲染的深度圖
        acc_map: [batch] 累積的不透明度
        extras: 額外信息字典
    """
    # 初始化額外信息字典
    extras = {}
    
    # 粗採樣
    z_vals, points, viewdirs = sampler.sample_rays(rays_o, rays_d, near, far)
    
    # 位置編碼
    points_flat = points.reshape(-1, 3)
    points_encoded = encoder(points_flat)
    points_encoded = points_encoded.reshape(points.shape[0], points.shape[1], -1)
    
    # 方向編碼
    if use_viewdirs:
        viewdirs_flat = viewdirs.unsqueeze(1).expand(-1, points.shape[1], -1).reshape(-1, 3)
        viewdirs_encoded = encoder(viewdirs_flat)
        viewdirs_encoded = viewdirs_encoded.reshape(points.shape[0], points.shape[1], -1)
    else:
        viewdirs_encoded = None
    
    # 網絡前向傳播
    rgb, density = network(points_encoded, viewdirs_encoded)
    
    # 體積渲染
    rgb_map_coarse, depth_map_coarse, acc_map_coarse = renderer.render_volume(
        rgb, density, z_vals, rays_d, raw_noise_std
    )
    
    extras['rgb_map_coarse'] = rgb_map_coarse
    extras['depth_map_coarse'] = depth_map_coarse
    extras['acc_map_coarse'] = acc_map_coarse
    
    # 細採樣
    if use_fine:
        # 計算權重
        weights = renderer.compute_weights(density, z_vals, rays_d)
        
        # 細採樣
        z_vals_fine, points_fine, viewdirs_fine = sampler.sample_fine_rays(
            rays_o, rays_d, z_vals, weights
        )
        
        # 位置編碼
        points_fine_flat = points_fine.reshape(-1, 3)
        points_fine_encoded = encoder(points_fine_flat)
        points_fine_encoded = points_fine_encoded.reshape(points_fine.shape[0], points_fine.shape[1], -1)
        
        # 方向編碼
        if use_viewdirs:
            viewdirs_fine_flat = viewdirs_fine.unsqueeze(1).expand(-1, points_fine.shape[1], -1).reshape(-1, 3)
            viewdirs_fine_encoded = encoder(viewdirs_fine_flat)
            viewdirs_fine_encoded = viewdirs_fine_encoded.reshape(points_fine.shape[0], points_fine.shape[1], -1)
        else:
            viewdirs_fine_encoded = None
        
        # 網絡前向傳播
        rgb_fine, density_fine = network(points_fine_encoded, viewdirs_fine_encoded)
        
        # 體積渲染
        rgb_map_fine, depth_map_fine, acc_map_fine = renderer.render_volume(
            rgb_fine, density_fine, z_vals_fine, rays_d, raw_noise_std
        )
        
        extras['rgb_map_fine'] = rgb_map_fine
        extras['depth_map_fine'] = depth_map_fine
        extras['acc_map_fine'] = acc_map_fine
        
        return rgb_map_fine, depth_map_fine, acc_map_fine, extras
    
    return rgb_map_coarse, depth_map_coarse, acc_map_coarse, extras


def render_image(
    H: int,
    W: int,
    focal: float,
    c2w: torch.Tensor,
    near: float,
    far: float,
    network: torch.nn.Module,
    encoder: torch.nn.Module,
    renderer: VolumeRenderer,
    sampler: RaySampler,
    chunk: int = 1024,
    raw_noise_std: float = 0.0,
    use_viewdirs: bool = True,
    use_fine: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    渲染完整圖像
    
    Args:
        H: 圖像高度
        W: 圖像寬度
        focal: 焦距
        c2w: [4, 4] 相機到世界座標的變換矩陣
        near: 近平面距離
        far: 遠平面距離
        network: NeRF 網絡
        encoder: 位置編碼器
        renderer: 體積渲染器
        sampler: 射線採樣器
        chunk: 每批處理的射線數量
        raw_noise_std: 密度噪聲標準差
        use_viewdirs: 是否使用視角方向
        use_fine: 是否使用細採樣
        
    Returns:
        rgb_map: [H, W, 3] 渲染的 RGB 圖像
        depth_map: [H, W] 渲染的深度圖
        acc_map: [H, W] 累積的不透明度圖
        extras: 額外信息字典
    """
    # 生成射線
    rays_o, rays_d = sampler.get_rays(H, W, focal, c2w)
    
    # 初始化結果
    rgb_maps = []
    depth_maps = []
    acc_maps = []
    extras = {}
    
    # 分批處理射線
    for i in range(0, rays_o.shape[0], chunk):
        # 獲取當前批次的射線
        rays_o_chunk = rays_o[i:i+chunk]
        rays_d_chunk = rays_d[i:i+chunk]
        
        # 渲染當前批次的射線
        rgb_map, depth_map, acc_map, chunk_extras = render_rays(
            rays_o_chunk, rays_d_chunk, near, far,
            network, encoder, renderer, sampler,
            raw_noise_std, use_viewdirs, use_fine
        )
        
        # 保存結果
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        acc_maps.append(acc_map)
        
        # 保存額外信息
        for k, v in chunk_extras.items():
            if k not in extras:
                extras[k] = []
            extras[k].append(v)
    
    # 合併結果
    rgb_map = torch.cat(rgb_maps, 0).reshape(H, W, 3)
    depth_map = torch.cat(depth_maps, 0).reshape(H, W)
    acc_map = torch.cat(acc_maps, 0).reshape(H, W)
    
    # 合併額外信息
    for k in extras:
        extras[k] = torch.cat(extras[k], 0).reshape(H, W, -1)
    
    return rgb_map, depth_map, acc_map, extras


def create_camera_rays(
    H: int,
    W: int,
    focal: float,
    c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    創建相機射線
    
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