"""
NeRF 神經網絡模組

實現核心的 Neural Radiance Fields 網絡：
- 多層感知機 (MLP) 架構
- 跳躍連接
- 位置和方向分離處理
- 量子層預留接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from .base import BaseModel, QuantumReadyMixin


class NeRFNetwork(BaseModel, QuantumReadyMixin):
    """
    NeRF 神經網絡
    
    核心的神經輻射場網絡，預測 3D 點的顏色和密度
    """
    
    def __init__(self, config: dict):
        """
        初始化 NeRF 網絡
        
        Args:
            config: 網絡配置字典
        """
        super().__init__(config)
        
        # 網絡參數
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
                # 跳躍連接層
                self.pos_layers.append(nn.Linear(self.hidden_dim + self.pos_encode_dim, self.hidden_dim))
            else:
                self.pos_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        
        # 密度預測頭
        self.density_head = nn.Linear(self.hidden_dim, 1)
        
        # 特徵提取層 (用於顏色預測)
        self.feature_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # 顏色預測層 (依賴於觀看方向)
        self.color_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim + self.dir_encode_dim, self.hidden_dim // 2)
        ])
        self.rgb_head = nn.Linear(self.hidden_dim // 2, 3)
        
        # 量子層佔位符
        self.quantum_layers = nn.ModuleList()
        
        # 初始化權重
        self._initialize_weights()
        
        print(f"🧠 NeRF 網絡初始化:")
        print(f"   - 位置編碼維度: {self.pos_encode_dim}")
        print(f"   - 方向編碼維度: {self.dir_encode_dim}")
        print(f"   - 隱藏層維度: {self.hidden_dim}")
        print(f"   - 網絡層數: {self.num_layers}")
        print(f"   - 跳躍連接: {self.skip_connections}")
        print(f"   - 總參數量: {self.count_parameters():,}")
    
    def _initialize_weights(self):
        """初始化網絡權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            pos_encoded: [batch, pos_encode_dim] 編碼後的位置
            dir_encoded: [batch, dir_encode_dim] 編碼後的方向
            
        Returns:
            rgb: [batch, 3] 預測的 RGB 顏色
            density: [batch, 1] 預測的體積密度
        """
        # 處理位置信息
        h = pos_encoded
        for i, layer in enumerate(self.pos_layers):
            h = layer(h)
            h = F.relu(h)
            
            # 跳躍連接
            if i in self.skip_connections:
                h = torch.cat([h, pos_encoded], dim=-1)
        
        # 預測密度 (與觀看方向無關)
        density = F.relu(self.density_head(h))
        
        # 提取特徵用於顏色預測
        features = self.feature_layer(h)
        
        # 結合特徵和觀看方向
        color_input = torch.cat([features, dir_encoded], dim=-1)
        
        # 處理顏色預測
        for layer in self.color_layers:
            color_input = F.relu(layer(color_input))
        
        # 預測 RGB 顏色
        rgb = torch.sigmoid(self.rgb_head(color_input))
        
        return rgb, density
    
    def quantum_forward(self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        量子增強前向傳播 (佔位符)
        
        Args:
            pos_encoded: [batch, pos_encode_dim] 編碼後的位置
            dir_encoded: [batch, dir_encode_dim] 編碼後的方向
            
        Returns:
            rgb: [batch, 3] 預測的 RGB 顏色
            density: [batch, 1] 預測的體積密度
        """
        if self.is_quantum_enabled() and len(self.quantum_layers) > 0:
            # TODO: 實現量子層處理
            # 可能的量子增強：
            # 1. 量子神經網絡層
            # 2. 量子注意力機制
            # 3. 量子糾纏特徵處理
            pass
        
        # 回退到經典處理
        return self.forward(pos_encoded, dir_encoded)
    
    def get_density(self, pos_encoded: torch.Tensor) -> torch.Tensor:
        """
        僅預測密度 (用於快速採樣)
        
        Args:
            pos_encoded: [batch, pos_encode_dim] 編碼後的位置
            
        Returns:
            density: [batch, 1] 預測的體積密度
        """
        h = pos_encoded
        for i, layer in enumerate(self.pos_layers):
            h = layer(h)
            h = F.relu(h)
            
            if i in self.skip_connections:
                h = torch.cat([h, pos_encoded], dim=-1)
        
        density = F.relu(self.density_head(h))
        return density
    
    def get_features(self, pos_encoded: torch.Tensor) -> torch.Tensor:
        """
        提取位置特徵 (用於顏色預測)
        
        Args:
            pos_encoded: [batch, pos_encode_dim] 編碼後的位置
            
        Returns:
            features: [batch, hidden_dim] 位置特徵
        """
        h = pos_encoded
        for i, layer in enumerate(self.pos_layers):
            h = layer(h)
            h = F.relu(h)
            
            if i in self.skip_connections:
                h = torch.cat([h, pos_encoded], dim=-1)
        
        features = self.feature_layer(h)
        return features


class HierarchicalNeRF(BaseModel):
    """
    分層 NeRF 網絡
    
    包含粗糙和精細兩個網絡，用於分層體積採樣
    """
    
    def __init__(self, config: dict):
        """
        初始化分層 NeRF
        
        Args:
            config: 網絡配置字典
        """
        super().__init__(config)
        
        # 粗糙網絡 (較小)
        coarse_config = config.copy()
        coarse_config['hidden_dim'] = config.get('coarse_hidden_dim', 128)
        coarse_config['num_layers'] = config.get('coarse_num_layers', 6)
        self.coarse_network = NeRFNetwork(coarse_config)
        
        # 精細網絡 (較大)
        fine_config = config.copy()
        fine_config['hidden_dim'] = config.get('fine_hidden_dim', 256)
        fine_config['num_layers'] = config.get('fine_num_layers', 8)
        self.fine_network = NeRFNetwork(fine_config)
        
        print(f"🏗️ 分層 NeRF 初始化:")
        print(f"   - 粗糙網絡參數: {self.coarse_network.count_parameters():,}")
        print(f"   - 精細網絡參數: {self.fine_network.count_parameters():,}")
        print(f"   - 總參數量: {self.count_parameters():,}")
    
    def forward(self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor, 
                network_type: str = 'fine') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            pos_encoded: [batch, pos_encode_dim] 編碼後的位置
            dir_encoded: [batch, dir_encode_dim] 編碼後的方向
            network_type: 'coarse' 或 'fine'
            
        Returns:
            rgb: [batch, 3] 預測的 RGB 顏色
            density: [batch, 1] 預測的體積密度
        """
        if network_type == 'coarse':
            return self.coarse_network(pos_encoded, dir_encoded)
        else:
            return self.fine_network(pos_encoded, dir_encoded)
    
    def get_coarse_prediction(self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """獲取粗糙網絡預測"""
        return self.coarse_network(pos_encoded, dir_encoded)
    
    def get_fine_prediction(self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """獲取精細網絡預測"""
        return self.fine_network(pos_encoded, dir_encoded)


class CompactNeRF(BaseModel):
    """
    緊湊型 NeRF 網絡
    
    針對移動設備或快速推理優化的輕量級版本
    """
    
    def __init__(self, config: dict):
        """
        初始化緊湊型 NeRF
        
        Args:
            config: 網絡配置字典
        """
        super().__init__(config)
        
        self.pos_encode_dim = config.get('pos_encode_dim', 39)  # 較少的編碼維度
        self.dir_encode_dim = config.get('dir_encode_dim', 15)
        self.hidden_dim = config.get('hidden_dim', 64)  # 較小的隱藏層
        self.num_layers = config.get('num_layers', 4)   # 較少的層數
        
        # 共享主幹網絡
        self.backbone = nn.ModuleList()
        self.backbone.append(nn.Linear(self.pos_encode_dim, self.hidden_dim))
        
        for i in range(1, self.num_layers):
            self.backbone.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        
        # 密度和顏色頭
        self.density_head = nn.Linear(self.hidden_dim, 1)
        self.color_head = nn.Linear(self.hidden_dim + self.dir_encode_dim, 3)
        
        print(f"📱 緊湊型 NeRF 初始化:")
        print(f"   - 隱藏層維度: {self.hidden_dim}")
        print(f"   - 網絡層數: {self.num_layers}")
        print(f"   - 總參數量: {self.count_parameters():,}")
    
    def forward(self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            pos_encoded: [batch, pos_encode_dim] 編碼後的位置
            dir_encoded: [batch, dir_encode_dim] 編碼後的方向
            
        Returns:
            rgb: [batch, 3] 預測的 RGB 顏色
            density: [batch, 1] 預測的體積密度
        """
        # 主幹網絡處理
        h = pos_encoded
        for layer in self.backbone:
            h = F.relu(layer(h))
        
        # 預測密度
        density = F.relu(self.density_head(h))
        
        # 預測顏色 (結合方向信息)
        color_input = torch.cat([h, dir_encoded], dim=-1)
        rgb = torch.sigmoid(self.color_head(color_input))
        
        return rgb, density


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