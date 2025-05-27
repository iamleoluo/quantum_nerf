"""
NeRF 網絡模組

提供多種 NeRF 網絡實現：
- 標準 NeRF 網絡
- 分層 NeRF 網絡
- 輕量級 NeRF 網絡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from .base import BaseModel, QuantumReadyMixin


class NeRFNetwork(BaseModel, QuantumReadyMixin):
    """
    標準 NeRF 網絡
    
    實現原始 NeRF 論文中的網絡架構
    """
    
    def __init__(self, config: dict):
        """
        初始化 NeRF 網絡
        
        Args:
            config: 配置字典，包含：
                - pos_encode_dim: 位置編碼維度
                - dir_encode_dim: 方向編碼維度
                - hidden_dim: 隱藏層維度
                - num_layers: 層數
                - skip_connections: 跳躍連接層索引
        """
        super().__init__(config)
        QuantumReadyMixin.__init__(self)
        
        # 獲取配置
        self.pos_encode_dim = config.get('pos_encode_dim', 63)
        self.dir_encode_dim = config.get('dir_encode_dim', 27)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 8)
        self.skip_connections = config.get('skip_connections', [4])
        
        # 位置處理層
        self.pos_layers = nn.ModuleList()
        self.pos_layers.append(nn.Linear(self.pos_encode_dim, self.hidden_dim))
        
        for i in range(1, self.num_layers):
            if i in self.skip_connections:
                self.pos_layers.append(nn.Linear(self.hidden_dim + self.pos_encode_dim, self.hidden_dim))
            else:
                self.pos_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        
        # 密度預測頭
        self.density_head = nn.Linear(self.hidden_dim, 1)
        
        # 特徵提取層
        self.feature_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # 顏色預測層
        self.color_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim + self.dir_encode_dim, self.hidden_dim // 2)
        ])
        self.rgb_head = nn.Linear(self.hidden_dim // 2, 3)
    
    def forward(self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            pos_encoded: [batch, pos_encode_dim] 編碼後的位置
            dir_encoded: [batch, dir_encode_dim] 編碼後的方向
            
        Returns:
            rgb: [batch, 3] 預測的顏色
            density: [batch, 1] 預測的密度
        """
        # 處理位置
        h = pos_encoded
        for i, layer in enumerate(self.pos_layers):
            h = layer(h)
            h = F.relu(h)
            
            # 跳躍連接
            if i in self.skip_connections:
                h = torch.cat([h, pos_encoded], dim=-1)
        
        # 預測密度
        density = F.relu(self.density_head(h))
        
        # 提取特徵
        features = self.feature_layer(h)
        
        # 結合特徵和方向
        color_input = torch.cat([features, dir_encoded], dim=-1)
        
        # 處理顏色
        for layer in self.color_layers:
            color_input = F.relu(layer(color_input))
        
        # 預測 RGB
        rgb = torch.sigmoid(self.rgb_head(color_input))
        
        return rgb, density
    
    def quantum_forward(self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        量子增強的前向傳播
        
        Args:
            pos_encoded: [batch, pos_encode_dim] 編碼後的位置
            dir_encoded: [batch, dir_encode_dim] 編碼後的方向
            
        Returns:
            rgb: [batch, 3] 預測的顏色
            density: [batch, 1] 預測的密度
        """
        if not self.use_quantum:
            return self.forward(pos_encoded, dir_encoded)
        
        # 使用量子層處理位置和方向
        pos_encoded = super().quantum_forward(pos_encoded)
        dir_encoded = super().quantum_forward(dir_encoded)
        
        # 繼續標準前向傳播
        return self.forward(pos_encoded, dir_encoded)


class HierarchicalNeRF(BaseModel, QuantumReadyMixin):
    """
    分層 NeRF 網絡
    
    實現粗細兩階段採樣的 NeRF 網絡
    """
    
    def __init__(self, config: dict):
        """
        初始化分層 NeRF 網絡
        
        Args:
            config: 配置字典，包含：
                - pos_encode_dim: 位置編碼維度
                - dir_encode_dim: 方向編碼維度
                - hidden_dim: 隱藏層維度
                - num_layers: 層數
                - skip_connections: 跳躍連接層索引
        """
        super().__init__(config)
        QuantumReadyMixin.__init__(self)
        
        # 創建粗網絡和細網絡
        self.coarse_net = NeRFNetwork(config)
        self.fine_net = NeRFNetwork(config)
    
    def forward(self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor,
                z_vals: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            pos_encoded: [batch, n_samples, pos_encode_dim] 編碼後的位置
            dir_encoded: [batch, n_samples, dir_encode_dim] 編碼後的方向
            z_vals: [batch, n_samples] 採樣點的深度值
            rays_d: [batch, 3] 射線方向
            
        Returns:
            rgb_coarse: [batch, 3] 粗網絡預測的顏色
            rgb_fine: [batch, 3] 細網絡預測的顏色
            weights: [batch, n_samples] 重要性權重
        """
        # 粗網絡前向傳播
        rgb_coarse, density_coarse = self.coarse_net(pos_encoded, dir_encoded)
        
        # 計算粗網絡的權重
        weights = self.compute_weights(density_coarse, z_vals, rays_d)
        
        # 重要性採樣
        z_vals_fine = self.importance_sampling(z_vals, weights)
        
        # 細網絡前向傳播
        pos_encoded_fine = self.encode_positions(z_vals_fine, rays_d)
        dir_encoded_fine = self.encode_directions(rays_d)
        rgb_fine, _ = self.fine_net(pos_encoded_fine, dir_encoded_fine)
        
        return rgb_coarse, rgb_fine, weights
    
    def compute_weights(self, density: torch.Tensor, z_vals: torch.Tensor,
                       rays_d: torch.Tensor) -> torch.Tensor:
        """
        計算體積渲染權重
        
        Args:
            density: [batch, n_samples, 1] 密度值
            z_vals: [batch, n_samples] 深度值
            rays_d: [batch, 3] 射線方向
            
        Returns:
            weights: [batch, n_samples] 權重
        """
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
        
        return weights
    
    def importance_sampling(self, z_vals: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        重要性採樣
        
        Args:
            z_vals: [batch, n_samples] 原始深度值
            weights: [batch, n_samples] 權重
            
        Returns:
            z_vals_fine: [batch, n_samples] 新的深度值
        """
        # 計算累積分布函數
        cdf = torch.cumsum(weights, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
        
        # 均勻採樣
        u = torch.linspace(0., 1., steps=z_vals.shape[-1], device=z_vals.device)
        u = u.expand(list(z_vals.shape[:-1]) + [z_vals.shape[-1]])
        
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


class CompactNeRF(BaseModel, QuantumReadyMixin):
    """
    輕量級 NeRF 網絡
    
    實現一個更小更快的 NeRF 網絡
    """
    
    def __init__(self, config: dict):
        """
        初始化輕量級 NeRF 網絡
        
        Args:
            config: 配置字典，包含：
                - pos_encode_dim: 位置編碼維度
                - dir_encode_dim: 方向編碼維度
                - hidden_dim: 隱藏層維度
                - num_layers: 層數
        """
        super().__init__(config)
        QuantumReadyMixin.__init__(self)
        
        # 獲取配置
        self.pos_encode_dim = config.get('pos_encode_dim', 63)
        self.dir_encode_dim = config.get('dir_encode_dim', 27)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 4)
        
        # 共享特徵提取層
        self.feature_net = nn.Sequential(
            nn.Linear(self.pos_encode_dim, self.hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU()
            ) for _ in range(self.num_layers-2)],
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 密度預測頭
        self.density_head = nn.Linear(self.hidden_dim, 1)
        
        # 顏色預測頭
        self.color_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.dir_encode_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3)
        )
    
    def forward(self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            pos_encoded: [batch, pos_encode_dim] 編碼後的位置
            dir_encoded: [batch, dir_encode_dim] 編碼後的方向
            
        Returns:
            rgb: [batch, 3] 預測的顏色
            density: [batch, 1] 預測的密度
        """
        # 提取特徵
        features = self.feature_net(pos_encoded)
        
        # 預測密度
        density = F.relu(self.density_head(features))
        
        # 預測顏色
        color_input = torch.cat([features, dir_encoded], dim=-1)
        rgb = torch.sigmoid(self.color_head(color_input))
        
        return rgb, density
    
    def quantum_forward(self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        量子增強的前向傳播
        
        Args:
            pos_encoded: [batch, pos_encode_dim] 編碼後的位置
            dir_encoded: [batch, dir_encode_dim] 編碼後的方向
            
        Returns:
            rgb: [batch, 3] 預測的顏色
            density: [batch, 1] 預測的密度
        """
        if not self.use_quantum:
            return self.forward(pos_encoded, dir_encoded)
        
        # 使用量子層處理位置和方向
        pos_encoded = super().quantum_forward(pos_encoded)
        dir_encoded = super().quantum_forward(dir_encoded)
        
        # 繼續標準前向傳播
        return self.forward(pos_encoded, dir_encoded)


def create_nerf_network(config: dict) -> BaseModel:
    """
    根據配置創建 NeRF 網絡
    
    Args:
        config: 網絡配置字典
        
    Returns:
        network: NeRF 網絡實例
    """
    network_type = config.get('type', 'standard')
    
    if network_type == 'standard':
        return NeRFNetwork(config)
    elif network_type == 'hierarchical':
        return HierarchicalNeRF(config)
    elif network_type == 'compact':
        return CompactNeRF(config)
    else:
        raise ValueError(f"未知的網絡類型: {network_type}")


# 輔助函數
def test_network_forward(network: BaseModel, batch_size: int = 1024):
    """
    測試網絡前向傳播
    
    Args:
        network: NeRF 網絡
        batch_size: 批次大小
    """
    print(f"🧪 測試網絡前向傳播:")
    
    # 創建測試輸入
    pos_encoded = torch.randn(batch_size, network.pos_encode_dim)
    dir_encoded = torch.randn(batch_size, network.dir_encode_dim)
    
    # 前向傳播
    with torch.no_grad():
        rgb, density = network(pos_encoded, dir_encoded)
    
    print(f"   - 輸入形狀: pos {pos_encoded.shape}, dir {dir_encoded.shape}")
    print(f"   - 輸出形狀: rgb {rgb.shape}, density {density.shape}")
    print(f"   - RGB 範圍: [{rgb.min():.3f}, {rgb.max():.3f}]")
    print(f"   - 密度範圍: [{density.min():.3f}, {density.max():.3f}]")
    
    return rgb, density


def analyze_network_gradients(network: BaseModel, pos_encoded: torch.Tensor, 
                            dir_encoded: torch.Tensor, target_rgb: torch.Tensor):
    """
    分析網絡梯度
    
    Args:
        network: NeRF 網絡
        pos_encoded: 編碼位置
        dir_encoded: 編碼方向
        target_rgb: 目標顏色
    """
    print(f"📊 分析網絡梯度:")
    
    # 前向傳播
    rgb, density = network(pos_encoded, dir_encoded)
    
    # 計算損失
    loss = F.mse_loss(rgb, target_rgb)
    
    # 反向傳播
    loss.backward()
    
    # 分析梯度
    total_grad_norm = 0
    param_count = 0
    
    for name, param in network.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            param_count += 1
            
            if 'weight' in name:
                print(f"   - {name}: 梯度範數 = {grad_norm:.6f}")
    
    avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
    print(f"   - 平均梯度範數: {avg_grad_norm:.6f}")
    print(f"   - 損失值: {loss.item():.6f}")
    
    return loss.item(), avg_grad_norm 