"""
基礎模型類

定義所有模型的共同接口和基礎功能
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseModel(nn.Module, ABC):
    """
    所有模型的基礎類
    
    提供共同的接口和功能：
    - 模型保存和加載
    - 參數統計
    - 設備管理
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """前向傳播 - 子類必須實現"""
        pass
    
    def count_parameters(self) -> int:
        """計算模型參數數量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, path: str, epoch: Optional[int] = None, 
                   optimizer_state: Optional[Dict] = None):
        """
        保存模型
        
        Args:
            path: 保存路徑
            epoch: 當前訓練輪數
            optimizer_state: 優化器狀態
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }
        
        if epoch is not None:
            save_dict['epoch'] = epoch
            
        if optimizer_state is not None:
            save_dict['optimizer_state_dict'] = optimizer_state
            
        torch.save(save_dict, path)
        print(f"✅ 模型已保存至: {path}")
    
    def load_model(self, path: str, strict: bool = True) -> Dict[str, Any]:
        """
        加載模型
        
        Args:
            path: 模型路徑
            strict: 是否嚴格匹配參數
            
        Returns:
            加載的額外信息（如 epoch, optimizer_state 等）
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        print(f"✅ 模型已從 {path} 加載")
        
        # 返回額外信息
        extra_info = {}
        for key in ['epoch', 'optimizer_state_dict', 'config']:
            if key in checkpoint:
                extra_info[key] = checkpoint[key]
                
        return extra_info
    
    def to_device(self, device: Optional[torch.device] = None):
        """移動模型到指定設備"""
        if device is None:
            device = self.device
        self.to(device)
        self.device = device
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型信息"""
        return {
            'model_class': self.__class__.__name__,
            'parameters': self.count_parameters(),
            'device': str(self.device),
            'config': self.config
        }
    
    def print_model_info(self):
        """打印模型信息"""
        info = self.get_model_info()
        print(f"📊 模型信息:")
        print(f"   類別: {info['model_class']}")
        print(f"   參數數量: {info['parameters']:,}")
        print(f"   設備: {info['device']}")
        print(f"   配置: {info['config']}")


class QuantumReadyMixin:
    """
    量子就緒混合類
    
    為模型添加量子計算整合的接口
    """
    
    def __init__(self):
        self.quantum_enabled = False
        self.quantum_components = {}
    
    def enable_quantum(self, components: Dict[str, Any]):
        """啟用量子組件"""
        self.quantum_enabled = True
        self.quantum_components = components
        print("🌌 量子組件已啟用")
    
    def disable_quantum(self):
        """禁用量子組件"""
        self.quantum_enabled = False
        self.quantum_components = {}
        print("💻 切換回經典計算")
    
    def is_quantum_enabled(self) -> bool:
        """檢查是否啟用量子計算"""
        return self.quantum_enabled 