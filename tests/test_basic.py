"""
基礎測試模組

測試專案的基本功能和組件
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class TestProjectStructure:
    """測試專案結構"""
    
    def test_project_directories_exist(self):
        """測試必要的目錄是否存在"""
        required_dirs = [
            'src',
            'src/models',
            'src/utils', 
            'src/training',
            'src/rendering',
            'src/quantum',
            'data',
            'configs',
            'tests',
            'notebooks',
            'outputs'
        ]
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"目錄 {dir_name} 不存在"
            assert dir_path.is_dir(), f"{dir_name} 不是目錄"
    
    def test_config_files_exist(self):
        """測試配置文件是否存在"""
        config_files = [
            'configs/basic_config.yaml',
            'requirements.txt',
            'README.md'
        ]
        
        for file_name in config_files:
            file_path = project_root / file_name
            assert file_path.exists(), f"文件 {file_name} 不存在"
            assert file_path.is_file(), f"{file_name} 不是文件"


class TestBasicFunctionality:
    """測試基本功能"""
    
    def test_torch_installation(self):
        """測試 PyTorch 是否正確安裝"""
        assert torch.__version__ is not None
        
        # 測試基本張量操作
        x = torch.randn(3, 4)
        y = torch.randn(4, 5)
        z = torch.mm(x, y)
        
        assert z.shape == (3, 5)
    
    def test_cuda_availability(self):
        """測試 CUDA 可用性（如果有的話）"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            x = torch.randn(10, 10, device=device)
            assert x.device.type == 'cuda'
        else:
            # 如果沒有 CUDA，確保 CPU 工作正常
            device = torch.device('cpu')
            x = torch.randn(10, 10, device=device)
            assert x.device.type == 'cpu'
    
    def test_numpy_integration(self):
        """測試 NumPy 整合"""
        # PyTorch 到 NumPy
        torch_tensor = torch.randn(5, 5)
        numpy_array = torch_tensor.numpy()
        assert isinstance(numpy_array, np.ndarray)
        
        # NumPy 到 PyTorch
        numpy_array = np.random.randn(3, 3)
        torch_tensor = torch.from_numpy(numpy_array)
        assert isinstance(torch_tensor, torch.Tensor)


class TestSimplePositionalEncoder:
    """測試簡化的位置編碼器"""
    
    def setup_method(self):
        """設置測試環境"""
        self.input_dims = 3
        self.num_freqs = 4
        self.encoder = self._create_simple_encoder()
    
    def _create_simple_encoder(self):
        """創建簡化的位置編碼器"""
        class SimplePositionalEncoder:
            def __init__(self, input_dims=3, num_freqs=4):
                self.input_dims = input_dims
                self.num_freqs = num_freqs
                self.freq_bands = 2.0 ** torch.linspace(0., num_freqs-1, steps=num_freqs)
                self.out_dim = input_dims * (1 + 2 * num_freqs)
            
            def encode(self, inputs):
                outputs = [inputs]
                for freq in self.freq_bands:
                    outputs.append(torch.sin(inputs * freq))
                    outputs.append(torch.cos(inputs * freq))
                return torch.cat(outputs, dim=-1)
        
        return SimplePositionalEncoder(self.input_dims, self.num_freqs)
    
    def test_encoder_initialization(self):
        """測試編碼器初始化"""
        assert self.encoder.input_dims == self.input_dims
        assert self.encoder.num_freqs == self.num_freqs
        assert self.encoder.out_dim == self.input_dims * (1 + 2 * self.num_freqs)
    
    def test_encoding_shape(self):
        """測試編碼輸出形狀"""
        batch_size = 10
        inputs = torch.randn(batch_size, self.input_dims)
        
        encoded = self.encoder.encode(inputs)
        
        expected_shape = (batch_size, self.encoder.out_dim)
        assert encoded.shape == expected_shape
    
    def test_encoding_deterministic(self):
        """測試編碼的確定性"""
        inputs = torch.tensor([[1.0, 2.0, 3.0]])
        
        encoded1 = self.encoder.encode(inputs)
        encoded2 = self.encoder.encode(inputs)
        
        assert torch.allclose(encoded1, encoded2)
    
    def test_encoding_different_inputs(self):
        """測試不同輸入產生不同編碼"""
        inputs1 = torch.tensor([[1.0, 2.0, 3.0]])
        inputs2 = torch.tensor([[4.0, 5.0, 6.0]])
        
        encoded1 = self.encoder.encode(inputs1)
        encoded2 = self.encoder.encode(inputs2)
        
        assert not torch.allclose(encoded1, encoded2)


class TestMathematicalOperations:
    """測試數學運算"""
    
    def test_volume_rendering_basics(self):
        """測試體積渲染的基本數學運算"""
        # 模擬密度和顏色值
        batch_size = 5
        n_samples = 10
        
        density = torch.rand(batch_size, n_samples, 1)
        rgb = torch.rand(batch_size, n_samples, 3)
        z_vals = torch.linspace(2.0, 6.0, n_samples).expand(batch_size, n_samples)
        
        # 計算距離
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[..., :1].shape)], -1)
        
        # 計算 alpha 值
        alpha = 1. - torch.exp(-density[..., 0] * dists)
        
        # 計算透射率
        transmittance = torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1
        )[:, :-1]
        
        # 計算權重
        weights = alpha * transmittance
        
        # 檢查權重的性質
        assert weights.shape == (batch_size, n_samples)
        assert torch.all(weights >= 0)  # 權重應該非負
        assert torch.all(torch.sum(weights, dim=-1) <= 1.0 + 1e-6)  # 權重和應該 <= 1
    
    def test_ray_generation(self):
        """測試射線生成"""
        height, width = 32, 32
        focal = 25.0
        
        # 創建像素座標
        i, j = torch.meshgrid(
            torch.linspace(0, width-1, width),
            torch.linspace(0, height-1, height),
            indexing='xy'
        )
        
        # 轉換為相機座標
        dirs = torch.stack([
            (i - width * 0.5) / focal,
            -(j - height * 0.5) / focal,
            -torch.ones_like(i)
        ], -1)
        
        assert dirs.shape == (height, width, 3)
        
        # 檢查 Z 分量
        assert torch.all(dirs[..., 2] == -1.0)


if __name__ == "__main__":
    # 運行測試
    pytest.main([__file__, "-v"]) 