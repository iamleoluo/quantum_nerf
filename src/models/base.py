"""
基礎模型模組

提供所有模型的基礎類和量子就緒混合類：
- BaseModel: 所有模型的基礎類
- QuantumReadyMixin: 為模型添加量子計算能力的混合類
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """
    所有模型的基礎類
    
    提供基本的模型接口和通用功能：
    - 模型配置管理
    - 權重保存和加載
    - 訓練和評估模式切換
    - 設備管理
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化基礎模型
        
        Args:
            config: 模型配置字典
        """
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        前向傳播（必須由子類實現）
        """
        pass
    
    def save_weights(self, path: str):
        """
        保存模型權重
        
        Args:
            path: 保存路徑
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        print(f"✅ 模型權重已保存到: {path}")
    
    def load_weights(self, path: str):
        """
        加載模型權重
        
        Args:
            path: 權重文件路徑
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 模型權重已從 {path} 加載")
    
    def get_num_parameters(self) -> int:
        """
        獲取模型參數數量
        
        Returns:
            參數總數
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """
        獲取可訓練參數數量
        
        Returns:
            可訓練參數總數
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_model_summary(self):
        """
        打印模型摘要信息
        """
        print(f"模型名稱: {self.__class__.__name__}")
        print(f"總參數數量: {self.get_num_parameters():,}")
        print(f"可訓練參數: {self.get_trainable_parameters():,}")
        print(f"設備: {self.device}")
        print("\n模型結構:")
        print(self)


class QuantumReadyMixin:
    """
    量子就緒混合類
    
    為模型添加量子計算能力：
    - 量子層管理
    - 量子-經典混合計算
    - 量子硬件接口
    """
    
    def __init__(self):
        """
        初始化量子就緒混合類
        """
        self.quantum_layers = []
        self.quantum_device = None
        self.use_quantum = False
    
    def add_quantum_layer(self, layer: nn.Module):
        """
        添加量子層
        
        Args:
            layer: 量子層模組
        """
        self.quantum_layers.append(layer)
    
    def set_quantum_device(self, device: str):
        """
        設置量子設備
        
        Args:
            device: 量子設備名稱
        """
        self.quantum_device = device
        print(f"✅ 量子設備已設置為: {device}")
    
    def enable_quantum(self, enable: bool = True):
        """
        啟用/禁用量子計算
        
        Args:
            enable: 是否啟用量子計算
        """
        self.use_quantum = enable
        print(f"✅ 量子計算已{'啟用' if enable else '禁用'}")
    
    def quantum_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        量子前向傳播
        
        Args:
            x: 輸入張量
            
        Returns:
            經過量子處理的張量
        """
        if not self.use_quantum or not self.quantum_layers:
            return x
        
        # 在量子層之間傳遞數據
        for layer in self.quantum_layers:
            x = layer(x)
        
        return x
    
    def get_quantum_circuit(self) -> Optional[Any]:
        """
        獲取量子電路
        
        Returns:
            量子電路對象（如果可用）
        """
        if not self.quantum_layers:
            return None
        
        # 返回第一個量子層的電路
        return self.quantum_layers[0].circuit
    
    def measure_quantum_state(self) -> torch.Tensor:
        """
        測量量子態
        
        Returns:
            測量結果
        """
        if not self.use_quantum:
            return torch.zeros(1)
        
        # 執行量子測量
        results = []
        for layer in self.quantum_layers:
            if hasattr(layer, 'measure'):
                results.append(layer.measure())
        
        return torch.stack(results) if results else torch.zeros(1) 