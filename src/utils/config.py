"""
配置管理模組

處理 NeRF 專案的配置文件：
- YAML 配置解析
- 參數驗證
- 動態配置更新
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union


class ConfigManager:
    """
    配置管理器
    
    負責加載、驗證和管理專案配置
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路徑
        """
        self.config = {}
        self.config_path = config_path
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加載配置文件
        
        Args:
            config_path: 配置文件路徑
            
        Returns:
            config: 配置字典
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        self.config_path = str(config_path)
        print(f"✅ 配置文件已加載: {config_path}")
        
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        獲取配置值
        
        Args:
            key: 配置鍵 (支持點分隔的嵌套鍵)
            default: 默認值
            
        Returns:
            value: 配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        設置配置值
        
        Args:
            key: 配置鍵 (支持點分隔的嵌套鍵)
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """
        批量更新配置
        
        Args:
            updates: 更新字典
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def save_config(self, save_path: Optional[str] = None):
        """
        保存配置文件
        
        Args:
            save_path: 保存路徑 (默認覆蓋原文件)
        """
        if save_path is None:
            save_path = self.config_path
        
        if save_path is None:
            raise ValueError("未指定保存路徑")
        
        save_path = Path(save_path)
        
        if save_path.suffix.lower() in ['.yaml', '.yml']:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        elif save_path.suffix.lower() == '.json':
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的保存格式: {save_path.suffix}")
        
        print(f"✅ 配置已保存: {save_path}")
    
    def validate_config(self) -> bool:
        """
        驗證配置的有效性
        
        Returns:
            is_valid: 配置是否有效
        """
        required_keys = [
            'model.hidden_dim',
            'model.num_layers',
            'training.num_epochs',
            'training.batch_size',
            'training.learning_rate'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                print(f"❌ 缺少必需的配置項: {key}")
                return False
        
        print("✅ 配置驗證通過")
        return True
    
    def get_model_config(self) -> Dict[str, Any]:
        """獲取模型配置"""
        return self.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """獲取訓練配置"""
        return self.get('training', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """獲取數據配置"""
        return self.get('data', {})
    
    def get_rendering_config(self) -> Dict[str, Any]:
        """獲取渲染配置"""
        return self.get('rendering', {})
    
    def print_config(self):
        """打印配置信息"""
        print("📋 當前配置:")
        print(yaml.dump(self.config, default_flow_style=False, allow_unicode=True))


def load_config(config_path: str) -> ConfigManager:
    """
    快速加載配置文件
    
    Args:
        config_path: 配置文件路徑
        
    Returns:
        config_manager: 配置管理器實例
    """
    return ConfigManager(config_path)


def create_default_config() -> Dict[str, Any]:
    """
    創建默認配置
    
    Returns:
        default_config: 默認配置字典
    """
    return {
        'experiment': {
            'name': 'nerf_experiment',
            'description': 'NeRF 訓練實驗',
            'output_dir': 'outputs',
            'seed': 42
        },
        'model': {
            'type': 'standard',
            'hidden_dim': 256,
            'num_layers': 8,
            'skip_connections': [4],
            'pos_encoding': {
                'input_dims': 3,
                'max_freq_log2': 10,
                'num_freqs': 10,
                'include_input': True
            },
            'dir_encoding': {
                'input_dims': 3,
                'max_freq_log2': 4,
                'num_freqs': 4,
                'include_input': True
            }
        },
        'training': {
            'num_epochs': 10000,
            'batch_size': 1024,
            'learning_rate': 5e-4,
            'lr_scheduler': {
                'type': 'exponential',
                'gamma': 0.1,
                'step_size': 5000
            },
            'weight_decay': 0.0,
            'log_every': 100,
            'save_every': 1000,
            'validate_every': 500
        },
        'rendering': {
            'near': 2.0,
            'far': 6.0,
            'n_samples': 64,
            'n_importance': 128,
            'white_background': True,
            'chunk_size': 1024
        },
        'data': {
            'data_dir': 'data',
            'scene_name': 'lego',
            'image_height': 800,
            'image_width': 800,
            'focal_length': 525.0,
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        },
        'hardware': {
            'use_cuda': True,
            'gpu_id': 0,
            'num_workers': 4,
            'pin_memory': True
        }
    } 