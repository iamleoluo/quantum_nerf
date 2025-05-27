"""
位置編碼模組

實現 NeRF 中的位置編碼功能：
- 正弦餘弦位置編碼
- 多頻率特徵提取
- 量子編碼接口預留
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List
from .base import BaseModel, QuantumReadyMixin


class PositionalEncoder(QuantumReadyMixin):
    """
    位置編碼器
    
    使用正弦和餘弦函數對 3D 座標進行編碼，
    將低維座標映射到高維特徵空間。
    """
    
    def __init__(self, input_dims: int = 3, max_freq_log2: int = 10, 
                 num_freqs: int = 10, include_input: bool = True,
                 log_sampling: bool = True):
        """
        初始化位置編碼器
        
        Args:
            input_dims: 輸入維度 (通常是 3 for 3D 座標)
            max_freq_log2: 最大頻率的 log2 值
            num_freqs: 頻率數量
            include_input: 是否包含原始輸入
            log_sampling: 是否使用對數採樣頻率
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = log_sampling
        
        # 週期函數
        self.periodic_fns = [torch.sin, torch.cos]
        
        # 創建頻率帶
        if log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq_log2, steps=num_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq_log2, steps=num_freqs)
        
        # 註冊為緩衝區 (不會被當作參數訓練)
        self.register_buffer('freq_bands', freq_bands)
        
        # 計算輸出維度
        out_dim = 0
        if include_input:
            out_dim += input_dims
        out_dim += input_dims * len(self.periodic_fns) * num_freqs
        self.out_dim = out_dim
        
        print(f"🔢 位置編碼器初始化:")
        print(f"   - 輸入維度: {input_dims}")
        print(f"   - 頻率數量: {num_freqs}")
        print(f"   - 輸出維度: {out_dim}")
        print(f"   - 頻率範圍: [1, {2**max_freq_log2}]")
    
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        對輸入座標進行位置編碼
        
        Args:
            inputs: [..., input_dims] 輸入座標
            
        Returns:
            encoded: [..., out_dim] 編碼後的特徵
        """
        outputs = []
        
        # 包含原始輸入
        if self.include_input:
            outputs.append(inputs)
        
        # 對每個頻率應用正弦和餘弦函數
        for freq in self.freq_bands:
            for p_fn in self.periodic_fns:
                outputs.append(p_fn(inputs * freq))
        
        return torch.cat(outputs, dim=-1)
    
    def encode_with_gradients(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        編碼並保留梯度信息 (用於訓練)
        
        Args:
            inputs: [..., input_dims] 輸入座標
            
        Returns:
            encoded: [..., out_dim] 編碼後的特徵
        """
        return self.encode(inputs)
    
    def get_frequency_bands(self) -> torch.Tensor:
        """獲取頻率帶"""
        return self.freq_bands
    
    def get_encoding_info(self) -> dict:
        """獲取編碼器信息"""
        return {
            'input_dims': self.input_dims,
            'output_dims': self.out_dim,
            'num_freqs': self.num_freqs,
            'max_freq': 2**self.max_freq_log2,
            'include_input': self.include_input,
            'log_sampling': self.log_sampling
        }
    
    # 量子編碼接口 (未來實現)
    def quantum_encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        量子位置編碼 (佔位符)
        
        未來可以實現：
        - 量子傅立葉變換 (QFT)
        - 變分量子電路編碼
        - 量子特徵映射
        
        Args:
            inputs: [..., input_dims] 輸入座標
            
        Returns:
            encoded: [..., quantum_out_dim] 量子編碼特徵
        """
        if self.is_quantum_enabled():
            # TODO: 實現量子編碼
            # 可能的實現：
            # 1. 量子傅立葉變換
            # 2. 參數化量子電路
            # 3. 量子核方法
            pass
        
        # 回退到經典編碼
        return self.encode(inputs)


class MultiScalePositionalEncoder(PositionalEncoder):
    """
    多尺度位置編碼器
    
    在不同尺度上應用位置編碼，適用於多解析度訓練
    """
    
    def __init__(self, input_dims: int = 3, scales: List[int] = [1, 2, 4, 8],
                 num_freqs_per_scale: int = 4, include_input: bool = True):
        """
        初始化多尺度編碼器
        
        Args:
            input_dims: 輸入維度
            scales: 不同的尺度列表
            num_freqs_per_scale: 每個尺度的頻率數量
            include_input: 是否包含原始輸入
        """
        self.scales = scales
        self.num_freqs_per_scale = num_freqs_per_scale
        
        # 計算總頻率數
        total_freqs = len(scales) * num_freqs_per_scale
        max_freq_log2 = int(np.log2(max(scales))) + num_freqs_per_scale - 1
        
        super().__init__(
            input_dims=input_dims,
            max_freq_log2=max_freq_log2,
            num_freqs=total_freqs,
            include_input=include_input
        )
        
        # 重新計算頻率帶
        freq_bands = []
        for scale in scales:
            scale_freqs = 2.**torch.linspace(
                np.log2(scale), 
                np.log2(scale) + num_freqs_per_scale - 1, 
                steps=num_freqs_per_scale
            )
            freq_bands.append(scale_freqs)
        
        freq_bands = torch.cat(freq_bands)
        self.register_buffer('freq_bands', freq_bands)
        
        print(f"🔢 多尺度位置編碼器:")
        print(f"   - 尺度: {scales}")
        print(f"   - 每尺度頻率數: {num_freqs_per_scale}")
        print(f"   - 總頻率數: {total_freqs}")


class AdaptivePositionalEncoder(PositionalEncoder):
    """
    自適應位置編碼器
    
    可以根據訓練進度動態調整編碼強度
    """
    
    def __init__(self, input_dims: int = 3, max_freq_log2: int = 10,
                 num_freqs: int = 10, include_input: bool = True,
                 warmup_steps: int = 1000):
        """
        初始化自適應編碼器
        
        Args:
            warmup_steps: 預熱步數，在此期間逐漸增加頻率
        """
        super().__init__(input_dims, max_freq_log2, num_freqs, include_input)
        
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        print(f"🔢 自適應位置編碼器:")
        print(f"   - 預熱步數: {warmup_steps}")
    
    def encode(self, inputs: torch.Tensor, step: Optional[int] = None) -> torch.Tensor:
        """
        自適應編碼
        
        Args:
            inputs: [..., input_dims] 輸入座標
            step: 當前訓練步數
            
        Returns:
            encoded: [..., out_dim] 編碼後的特徵
        """
        if step is not None:
            self.current_step = step
        
        outputs = []
        
        # 包含原始輸入
        if self.include_input:
            outputs.append(inputs)
        
        # 計算當前應該使用的頻率數量
        if self.current_step < self.warmup_steps:
            active_freqs = max(1, int(self.num_freqs * self.current_step / self.warmup_steps))
        else:
            active_freqs = self.num_freqs
        
        # 對活躍的頻率應用編碼
        for i, freq in enumerate(self.freq_bands[:active_freqs]):
            for p_fn in self.periodic_fns:
                outputs.append(p_fn(inputs * freq))
        
        # 對未激活的頻率填充零
        if active_freqs < self.num_freqs:
            remaining_dims = (self.num_freqs - active_freqs) * len(self.periodic_fns) * self.input_dims
            zeros = torch.zeros(*inputs.shape[:-1], remaining_dims, device=inputs.device)
            outputs.append(zeros)
        
        return torch.cat(outputs, dim=-1)


def create_positional_encoder(config: dict) -> PositionalEncoder:
    """
    根據配置創建位置編碼器
    
    Args:
        config: 編碼器配置
        
    Returns:
        encoder: 位置編碼器實例
    """
    encoder_type = config.get('type', 'standard')
    
    if encoder_type == 'standard':
        return PositionalEncoder(
            input_dims=config.get('input_dims', 3),
            max_freq_log2=config.get('max_freq_log2', 10),
            num_freqs=config.get('num_freqs', 10),
            include_input=config.get('include_input', True)
        )
    elif encoder_type == 'multiscale':
        return MultiScalePositionalEncoder(
            input_dims=config.get('input_dims', 3),
            scales=config.get('scales', [1, 2, 4, 8]),
            num_freqs_per_scale=config.get('num_freqs_per_scale', 4),
            include_input=config.get('include_input', True)
        )
    elif encoder_type == 'adaptive':
        return AdaptivePositionalEncoder(
            input_dims=config.get('input_dims', 3),
            max_freq_log2=config.get('max_freq_log2', 10),
            num_freqs=config.get('num_freqs', 10),
            include_input=config.get('include_input', True),
            warmup_steps=config.get('warmup_steps', 1000)
        )
    else:
        raise ValueError(f"未知的編碼器類型: {encoder_type}")


# 輔助函數
def visualize_encoding(encoder: PositionalEncoder, inputs: torch.Tensor, 
                      save_path: Optional[str] = None):
    """
    可視化位置編碼效果
    
    Args:
        encoder: 位置編碼器
        inputs: 輸入座標
        save_path: 保存路徑 (可選)
    """
    import matplotlib.pyplot as plt
    
    encoded = encoder.encode(inputs)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始座標
    for i, coord_name in enumerate(['X', 'Y', 'Z'][:inputs.shape[-1]]):
        if i < 2:
            axes[0, i].plot(inputs[:, i].numpy())
            axes[0, i].set_title(f'原始 {coord_name} 座標')
            axes[0, i].set_ylabel(f'{coord_name} 值')
    
    # 編碼特徵
    axes[1, 0].imshow(encoded.T.numpy(), aspect='auto', cmap='viridis')
    axes[1, 0].set_title('編碼特徵矩陣')
    axes[1, 0].set_xlabel('樣本索引')
    axes[1, 0].set_ylabel('特徵維度')
    
    # 頻率響應
    freq_bands = encoder.get_frequency_bands()
    axes[1, 1].plot(freq_bands.numpy(), 'o-')
    axes[1, 1].set_title('頻率帶')
    axes[1, 1].set_xlabel('頻率索引')
    axes[1, 1].set_ylabel('頻率值')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def test_encoding_properties(encoder: PositionalEncoder):
    """
    測試編碼器的性質
    
    Args:
        encoder: 位置編碼器
    """
    print(f"🧪 測試位置編碼器性質:")
    
    # 測試輸入
    test_inputs = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
        [-1.0, -1.0, -1.0]
    ])
    
    encoded = encoder.encode(test_inputs)
    
    print(f"   - 輸入形狀: {test_inputs.shape}")
    print(f"   - 輸出形狀: {encoded.shape}")
    print(f"   - 輸出範圍: [{encoded.min():.3f}, {encoded.max():.3f}]")
    
    # 測試確定性
    encoded2 = encoder.encode(test_inputs)
    is_deterministic = torch.allclose(encoded, encoded2)
    print(f"   - 確定性: {'✅' if is_deterministic else '❌'}")
    
    # 測試不同輸入產生不同輸出
    different_inputs = test_inputs + 0.1
    encoded_diff = encoder.encode(different_inputs)
    is_different = not torch.allclose(encoded, encoded_diff)
    print(f"   - 區分性: {'✅' if is_different else '❌'}")
    
    return {
        'deterministic': is_deterministic,
        'different_outputs': is_different,
        'output_range': (encoded.min().item(), encoded.max().item())
    } 