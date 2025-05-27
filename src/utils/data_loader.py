"""
數據加載器模組

提供 NeRF 訓練所需的數據加載和處理功能：
- 數據集加載
- 數據預處理
- 批次生成
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, Optional


class NeRFDataset(Dataset):
    """NeRF 數據集"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Any] = None
    ):
        """
        初始化數據集
        
        Args:
            data_dir: 數據集目錄
            split: 數據集分割 ('train' 或 'val')
            transform: 數據轉換
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 加載相機參數
        with open(os.path.join(data_dir, f'transforms_{split}.json'), 'r') as f:
            self.meta = json.load(f)
        
        # 加載圖像
        self.images = []
        self.poses = []
        self.focal = None
        
        for frame in self.meta['frames']:
            # 加載圖像
            image_path = os.path.join(data_dir, frame['file_path'])
            image = torch.from_numpy(np.load(image_path))
            self.images.append(image)
            
            # 加載相機姿態
            pose = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
            self.poses.append(pose)
        
        # 獲取焦距
        self.focal = 0.5 * self.images[0].shape[1] / np.tan(0.5 * self.meta['camera_angle_x'])
        
        # 轉換為張量
        self.images = torch.stack(self.images)
        self.poses = torch.stack(self.poses)
    
    def __len__(self) -> int:
        """返回數據集大小"""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        獲取數據樣本
        
        Args:
            idx: 樣本索引
            
        Returns:
            sample: 數據樣本字典
        """
        image = self.images[idx]
        pose = self.poses[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'pose': pose,
            'focal': self.focal
        }


def create_data_loaders(
    data_dir: str,
    batch_size: int = 1024,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    創建數據加載器
    
    Args:
        data_dir: 數據集目錄
        batch_size: 批次大小
        num_workers: 工作進程數
        
    Returns:
        train_loader: 訓練數據加載器
        val_loader: 驗證數據加載器
    """
    # 創建數據集
    train_dataset = NeRFDataset(data_dir, split='train')
    val_dataset = NeRFDataset(data_dir, split='val')
    
    # 創建數據加載器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 