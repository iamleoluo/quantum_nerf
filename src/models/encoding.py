"""
位置編碼器模組

提供多種位置編碼方式：
- 標準位置編碼
- 多尺度位置編碼
- 自適應位置編碼
- 量子位置編碼（預留）
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from .base import BaseModel, QuantumReadyMixin


class PositionalEncoder(BaseModel):
    """
    標準位置編碼器
    
    使用正弦和餘弦函數進行位置編碼
    """
    
    def __init__(self, config: dict):
        """
        初始化位置編碼器
        
        Args:
            config: 配置字典，包含：
                - input_dims: 輸入維度
                - num_freqs: 頻率數量
                - include_input: 是否包含原始輸入
        """
        super().__init__(config)
        
        self.input_dims = config.get('input_dims', 3)
        self.num_freqs = config.get('num_freqs', 10)
        self.include_input = config.get('include_input', True)
        
        # 創建頻率帶
        freq_bands = 2.**torch.linspace(0., self.num_freqs-1, self.num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        
        # 計算輸出維度
        self.output_dims = self.input_dims * (2 * self.num_freqs + (1 if self.include_input else 0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入張量 [..., input_dims]
            
        Returns:
            編碼後的張量 [..., output_dims]
        """
        outputs = []
        
        if self.include_input:
            outputs.append(x)
        
        for freq in self.freq_bands:
            outputs.append(torch.sin(x * freq))
            outputs.append(torch.cos(x * freq))
        
        return torch.cat(outputs, dim=-1)


class MultiScalePositionalEncoder(BaseModel):
    """
    多尺度位置編碼器
    
    使用不同尺度的頻率進行編碼
    """
    
    def __init__(self, config: dict):
        """
        初始化多尺度位置編碼器
        
        Args:
            config: 配置字典，包含：
                - input_dims: 輸入維度
                - num_scales: 尺度數量
                - freqs_per_scale: 每個尺度的頻率數量
                - include_input: 是否包含原始輸入
        """
        super().__init__(config)
        
        self.input_dims = config.get('input_dims', 3)
        self.num_scales = config.get('num_scales', 3)
        self.freqs_per_scale = config.get('freqs_per_scale', 4)
        self.include_input = config.get('include_input', True)
        
        # 創建不同尺度的頻率帶
        self.freq_bands = []
        for scale in range(self.num_scales):
            base_freq = 2. ** scale
            freqs = base_freq * 2.**torch.linspace(0., self.freqs_per_scale-1, self.freqs_per_scale)
            self.register_buffer(f'freq_bands_{scale}', freqs)
            self.freq_bands.append(freqs)
        
        # 計算輸出維度
        self.output_dims = self.input_dims * (
            2 * self.num_scales * self.freqs_per_scale + 
            (1 if self.include_input else 0)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入張量 [..., input_dims]
            
        Returns:
            編碼後的張量 [..., output_dims]
        """
        outputs = []
        
        if self.include_input:
            outputs.append(x)
        
        for scale_freqs in self.freq_bands:
            for freq in scale_freqs:
                outputs.append(torch.sin(x * freq))
                outputs.append(torch.cos(x * freq))
        
        return torch.cat(outputs, dim=-1)


class AdaptivePositionalEncoder(BaseModel, QuantumReadyMixin):
    """
    自適應位置編碼器
    
    根據輸入數據動態調整編碼頻率
    """
    
    def __init__(self, config: dict):
        """
        初始化自適應位置編碼器
        
        Args:
            config: 配置字典，包含：
                - input_dims: 輸入維度
                - max_freqs: 最大頻率數量
                - min_freqs: 最小頻率數量
                - include_input: 是否包含原始輸入
        """
        super().__init__(config)
        QuantumReadyMixin.__init__(self)
        
        self.input_dims = config.get('input_dims', 3)
        self.max_freqs = config.get('max_freqs', 10)
        self.min_freqs = config.get('min_freqs', 4)
        self.include_input = config.get('include_input', True)
        
        # 頻率預測網絡
        self.freq_predictor = nn.Sequential(
            nn.Linear(self.input_dims, 64),
            nn.ReLU(),
            nn.Linear(64, self.max_freqs)
        )
        
        # 初始化頻率帶
        self.register_buffer('base_freqs', 
            2.**torch.linspace(0., self.max_freqs-1, self.max_freqs))
        
        # 計算輸出維度
        self.output_dims = self.input_dims * (
            2 * self.max_freqs + 
            (1 if self.include_input else 0)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入張量 [..., input_dims]
            
        Returns:
            編碼後的張量 [..., output_dims]
        """
        # 預測頻率權重
        freq_weights = torch.sigmoid(self.freq_predictor(x))
        
        # 選擇頻率
        selected_freqs = self.base_freqs * freq_weights
        
        outputs = []
        if self.include_input:
            outputs.append(x)
        
        for freq in selected_freqs:
            outputs.append(torch.sin(x * freq))
            outputs.append(torch.cos(x * freq))
        
        return torch.cat(outputs, dim=-1)
    
    def quantum_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        量子增強的前向傳播
        
        Args:
            x: 輸入張量 [..., input_dims]
            
        Returns:
            編碼後的張量 [..., output_dims]
        """
        if not self.use_quantum:
            return self.forward(x)
        
        # 使用量子層處理輸入
        x = super().quantum_forward(x)
        
        # 繼續標準編碼
        return self.forward(x)


class QuantumPositionalEncoder(BaseModel, QuantumReadyMixin):
    """
    量子位置編碼器（預留）
    
    使用量子電路進行位置編碼
    """
    
    def __init__(self, config: dict):
        """
        初始化量子位置編碼器
        
        Args:
            config: 配置字典，包含：
                - input_dims: 輸入維度
                - num_qubits: 量子比特數量
                - include_input: 是否包含原始輸入
        """
        super().__init__(config)
        QuantumReadyMixin.__init__(self)
        
        self.input_dims = config.get('input_dims', 3)
        self.num_qubits = config.get('num_qubits', 8)
        self.include_input = config.get('include_input', True)
        
        # 量子電路參數
        self.quantum_params = nn.Parameter(
            torch.randn(self.num_qubits * 3)  # 每個量子比特的旋轉參數
        )
        
        # 計算輸出維度
        self.output_dims = self.input_dims * (
            2 * self.num_qubits + 
            (1 if self.include_input else 0)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播（經典回退）
        
        Args:
            x: 輸入張量 [..., input_dims]
            
        Returns:
            編碼後的張量 [..., output_dims]
        """
        if not self.use_quantum:
            # 使用經典編碼作為回退
            outputs = []
            if self.include_input:
                outputs.append(x)
            
            # 使用量子參數進行經典編碼
            for i in range(self.num_qubits):
                freq = 2. ** i
                outputs.append(torch.sin(x * freq))
                outputs.append(torch.cos(x * freq))
            
            return torch.cat(outputs, dim=-1)
        
        # 量子編碼（待實現）
        return self.quantum_forward(x)
    
    def quantum_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        量子前向傳播（待實現）
        
        Args:
            x: 輸入張量 [..., input_dims]
            
        Returns:
            編碼後的張量 [..., output_dims]
        """
        # TODO: 實現量子編碼
        # 目前回退到經典編碼
        return self.forward(x) 