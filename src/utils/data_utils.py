"""
數據處理工具模組

處理 NeRF 訓練所需的各種數據：
- 圖像加載和預處理
- 相機參數解析
- 射線生成
- 數據集分割
"""

import torch
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import imageio
from PIL import Image


class NeRFDataLoader:
    """
    NeRF 數據加載器
    
    負責加載和預處理 NeRF 訓練所需的所有數據
    """
    
    def __init__(self, data_dir: str, scene_name: str = "lego", 
                 image_scale: float = 1.0, white_background: bool = True):
        """
        初始化數據加載器
        
        Args:
            data_dir: 數據目錄路徑
            scene_name: 場景名稱
            image_scale: 圖像縮放比例
            white_background: 是否使用白色背景
        """
        self.data_dir = Path(data_dir)
        self.scene_name = scene_name
        self.image_scale = image_scale
        self.white_background = white_background
        
        # 數據存儲
        self.images = {}
        self.poses = {}
        self.intrinsics = {}
        self.image_size = None
        
        print(f"📂 初始化數據加載器: {scene_name}")
    
    def load_images(self, split: str = "train") -> torch.Tensor:
        """
        加載圖像數據
        
        Args:
            split: 數據分割 ("train", "val", "test")
            
        Returns:
            images: [N, H, W, 3] 圖像張量
        """
        print(f"📸 加載 {split} 圖像...")
        
        image_dir = self.data_dir / split
        if not image_dir.exists():
            raise FileNotFoundError(f"圖像目錄不存在: {image_dir}")
        
        # 獲取所有圖像文件
        image_files = sorted(list(image_dir.glob("*.jpg")) + 
                           list(image_dir.glob("*.png")))
        
        if len(image_files) == 0:
            raise ValueError(f"在 {image_dir} 中未找到圖像文件")
        
        images = []
        for img_file in image_files:
            # 加載圖像
            img = imageio.imread(img_file)
            
            # 轉換為 RGB (如果是 RGBA)
            if img.shape[-1] == 4:
                if self.white_background:
                    # 白色背景混合
                    img = img[..., :3] * img[..., -1:] / 255.0 + (1.0 - img[..., -1:] / 255.0)
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img[..., :3]
            
            # 縮放圖像
            if self.image_scale != 1.0:
                h, w = img.shape[:2]
                new_h, new_w = int(h * self.image_scale), int(w * self.image_scale)
                img = np.array(Image.fromarray(img).resize((new_w, new_h)))
            
            # 標準化到 [0, 1]
            img = img.astype(np.float32) / 255.0
            images.append(img)
        
        images = np.stack(images, axis=0)
        self.images[split] = torch.from_numpy(images)
        self.image_size = images.shape[1:3]  # (H, W)
        
        print(f"✅ 加載了 {len(images)} 張 {split} 圖像，尺寸: {self.image_size}")
        return self.images[split]
    
    def load_camera_parameters(self, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        加載相機參數
        
        Args:
            split: 數據分割
            
        Returns:
            poses: [N, 4, 4] 相機位姿矩陣
            intrinsics: [N, 3, 3] 相機內參矩陣
        """
        print(f"📷 加載 {split} 相機參數...")
        
        # 加載位姿數據
        poses_file = self.data_dir / f"transforms_{split}.json"
        if not poses_file.exists():
            raise FileNotFoundError(f"相機參數文件不存在: {poses_file}")
        
        with open(poses_file, 'r') as f:
            meta = json.load(f)
        
        poses = []
        intrinsics = []
        
        # 提取相機參數
        camera_angle_x = meta.get('camera_angle_x', 0.6911112070083618)
        
        for frame in meta['frames']:
            # 位姿矩陣
            pose = np.array(frame['transform_matrix'], dtype=np.float32)
            poses.append(pose)
            
            # 內參矩陣
            if self.image_size is not None:
                h, w = self.image_size
                focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
                
                intrinsic = np.array([
                    [focal, 0, w/2],
                    [0, focal, h/2],
                    [0, 0, 1]
                ], dtype=np.float32)
                intrinsics.append(intrinsic)
        
        poses = np.stack(poses, axis=0)
        intrinsics = np.stack(intrinsics, axis=0) if intrinsics else None
        
        self.poses[split] = torch.from_numpy(poses)
        if intrinsics is not None:
            self.intrinsics[split] = torch.from_numpy(intrinsics)
        
        print(f"✅ 加載了 {len(poses)} 個相機位姿")
        return self.poses[split], self.intrinsics.get(split)
    
    def create_rays(self, poses: torch.Tensor, intrinsics: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        為給定的相機位姿創建射線
        
        Args:
            poses: [N, 4, 4] 相機位姿
            intrinsics: [N, 3, 3] 相機內參 (可選)
            
        Returns:
            rays_o: [N, H, W, 3] 射線起點
            rays_d: [N, H, W, 3] 射線方向
        """
        if self.image_size is None:
            raise ValueError("請先加載圖像以確定圖像尺寸")
        
        h, w = self.image_size
        n_poses = poses.shape[0]
        
        # 創建像素座標網格
        i, j = torch.meshgrid(
            torch.linspace(0, w-1, w),
            torch.linspace(0, h-1, h),
            indexing='xy'
        )
        
        rays_o_list = []
        rays_d_list = []
        
        for idx in range(n_poses):
            pose = poses[idx]
            
            if intrinsics is not None:
                # 使用提供的內參
                intrinsic = intrinsics[idx]
                focal_x, focal_y = intrinsic[0, 0], intrinsic[1, 1]
                cx, cy = intrinsic[0, 2], intrinsic[1, 2]
            else:
                # 使用默認內參
                focal_x = focal_y = 0.5 * w / np.tan(0.5 * 0.6911112070083618)
                cx, cy = w/2, h/2
            
            # 轉換為相機座標系
            dirs = torch.stack([
                (i - cx) / focal_x,
                -(j - cy) / focal_y,
                -torch.ones_like(i)
            ], dim=-1)
            
            # 轉換到世界座標系
            rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
            rays_o = pose[:3, -1].expand(rays_d.shape)
            
            rays_o_list.append(rays_o)
            rays_d_list.append(rays_d)
        
        rays_o = torch.stack(rays_o_list, dim=0)
        rays_d = torch.stack(rays_d_list, dim=0)
        
        return rays_o, rays_d
    
    def get_training_data(self, batch_size: int = 1024) -> Dict[str, torch.Tensor]:
        """
        獲取訓練數據批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            batch_data: 包含射線和目標顏色的字典
        """
        if "train" not in self.images or "train" not in self.poses:
            raise ValueError("請先加載訓練數據")
        
        images = self.images["train"]
        poses = self.poses["train"]
        intrinsics = self.intrinsics.get("train")
        
        # 創建射線
        rays_o, rays_d = self.create_rays(poses, intrinsics)
        
        # 隨機選擇圖像
        n_images = images.shape[0]
        img_idx = torch.randint(0, n_images, (1,)).item()
        
        # 獲取該圖像的數據
        target_img = images[img_idx]  # [H, W, 3]
        img_rays_o = rays_o[img_idx]  # [H, W, 3]
        img_rays_d = rays_d[img_idx]  # [H, W, 3]
        
        # 平坦化
        h, w = target_img.shape[:2]
        target_rgb = target_img.reshape(-1, 3)  # [H*W, 3]
        rays_o_flat = img_rays_o.reshape(-1, 3)  # [H*W, 3]
        rays_d_flat = img_rays_d.reshape(-1, 3)  # [H*W, 3]
        
        # 隨機採樣射線
        total_rays = h * w
        ray_indices = torch.randperm(total_rays)[:batch_size]
        
        batch_data = {
            'rays_o': rays_o_flat[ray_indices],      # [batch_size, 3]
            'rays_d': rays_d_flat[ray_indices],      # [batch_size, 3]
            'target_rgb': target_rgb[ray_indices],   # [batch_size, 3]
            'img_idx': img_idx,
            'ray_indices': ray_indices
        }
        
        return batch_data
    
    def get_validation_data(self, img_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        獲取驗證數據
        
        Args:
            img_idx: 圖像索引
            
        Returns:
            val_data: 驗證數據字典
        """
        if "val" not in self.images or "val" not in self.poses:
            raise ValueError("請先加載驗證數據")
        
        images = self.images["val"]
        poses = self.poses["val"]
        intrinsics = self.intrinsics.get("val")
        
        # 創建射線
        rays_o, rays_d = self.create_rays(poses, intrinsics)
        
        # 獲取指定圖像的數據
        target_img = images[img_idx]
        img_rays_o = rays_o[img_idx]
        img_rays_d = rays_d[img_idx]
        
        val_data = {
            'rays_o': img_rays_o,        # [H, W, 3]
            'rays_d': img_rays_d,        # [H, W, 3]
            'target_rgb': target_img,    # [H, W, 3]
            'img_idx': img_idx
        }
        
        return val_data


def create_synthetic_dataset(n_images: int = 100, image_size: Tuple[int, int] = (64, 64),
                           output_dir: str = "data/synthetic") -> Dict[str, np.ndarray]:
    """
    創建合成數據集用於演示
    
    Args:
        n_images: 圖像數量
        image_size: 圖像尺寸 (H, W)
        output_dir: 輸出目錄
        
    Returns:
        dataset: 包含圖像和相機參數的字典
    """
    print(f"🎨 創建合成數據集: {n_images} 張圖像，尺寸 {image_size}")
    
    h, w = image_size
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 創建相機軌跡 (圓形)
    angles = np.linspace(0, 2*np.pi, n_images, endpoint=False)
    radius = 4.0
    height = 1.0
    
    images = []
    poses = []
    
    for i, angle in enumerate(angles):
        # 相機位置
        cam_pos = np.array([
            radius * np.cos(angle),
            height,
            radius * np.sin(angle)
        ])
        
        # 朝向原點
        look_at = np.array([0., 0., 0.])
        up = np.array([0., 1., 0.])
        
        # 創建相機到世界的變換矩陣
        z_axis = cam_pos - look_at
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        pose = np.eye(4)
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = cam_pos
        
        poses.append(pose)
        
        # 創建簡單的合成圖像 (彩色方塊)
        img = np.zeros((h, w, 3))
        
        # 添加一些彩色區域
        quarter_h, quarter_w = h//4, w//4
        
        # 紅色區域
        img[quarter_h:quarter_h*2, quarter_w:quarter_w*2] = [1.0, 0.0, 0.0]
        # 綠色區域  
        img[quarter_h:quarter_h*2, quarter_w*2:quarter_w*3] = [0.0, 1.0, 0.0]
        # 藍色區域
        img[quarter_h*2:quarter_h*3, quarter_w:quarter_w*2] = [0.0, 0.0, 1.0]
        # 黃色區域
        img[quarter_h*2:quarter_h*3, quarter_w*2:quarter_w*3] = [1.0, 1.0, 0.0]
        
        # 添加一些隨機噪聲
        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)
        
        images.append(img)
    
    images = np.stack(images)
    poses = np.stack(poses)
    
    # 保存數據
    dataset = {
        'images': images,
        'poses': poses,
        'image_size': image_size,
        'n_images': n_images
    }
    
    # 保存為 numpy 格式
    np.savez(output_path / 'synthetic_dataset.npz', **dataset)
    
    # 創建 transforms.json 格式的相機參數
    camera_angle_x = 0.6911112070083618
    
    transforms = {
        'camera_angle_x': camera_angle_x,
        'frames': []
    }
    
    for i, pose in enumerate(poses):
        frame = {
            'file_path': f'./images/img_{i:03d}',
            'transform_matrix': pose.tolist()
        }
        transforms['frames'].append(frame)
    
    # 分割數據集
    n_train = int(0.8 * n_images)
    n_val = int(0.1 * n_images)
    
    # 訓練集
    train_transforms = {
        'camera_angle_x': camera_angle_x,
        'frames': transforms['frames'][:n_train]
    }
    
    # 驗證集
    val_transforms = {
        'camera_angle_x': camera_angle_x,
        'frames': transforms['frames'][n_train:n_train+n_val]
    }
    
    # 測試集
    test_transforms = {
        'camera_angle_x': camera_angle_x,
        'frames': transforms['frames'][n_train+n_val:]
    }
    
    # 保存 JSON 文件
    with open(output_path / 'transforms_train.json', 'w') as f:
        json.dump(train_transforms, f, indent=2)
    
    with open(output_path / 'transforms_val.json', 'w') as f:
        json.dump(val_transforms, f, indent=2)
        
    with open(output_path / 'transforms_test.json', 'w') as f:
        json.dump(test_transforms, f, indent=2)
    
    # 保存圖像文件
    for split, split_transforms in [('train', train_transforms), 
                                   ('val', val_transforms), 
                                   ('test', test_transforms)]:
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        for i, frame in enumerate(split_transforms['frames']):
            img_idx = int(frame['file_path'].split('_')[-1])
            img = (images[img_idx] * 255).astype(np.uint8)
            imageio.imwrite(split_dir / f'img_{i:03d}.png', img)
    
    print(f"✅ 合成數據集已保存到: {output_path}")
    print(f"   - 訓練集: {len(train_transforms['frames'])} 張圖像")
    print(f"   - 驗證集: {len(val_transforms['frames'])} 張圖像") 
    print(f"   - 測試集: {len(test_transforms['frames'])} 張圖像")
    
    return dataset


# 輔助函數
def create_rays(height: int, width: int, focal: float, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    為單個相機位姿創建射線
    
    Args:
        height, width: 圖像尺寸
        focal: 焦距
        pose: [4, 4] 相機位姿矩陣
        
    Returns:
        rays_o: [H*W, 3] 射線起點
        rays_d: [H*W, 3] 射線方向
    """
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
    
    # 轉換到世界座標系
    rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], -1)
    rays_o = pose[:3, -1].expand(rays_d.shape)
    
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def normalize_rays(rays_d: torch.Tensor) -> torch.Tensor:
    """
    標準化射線方向向量
    
    Args:
        rays_d: [..., 3] 射線方向
        
    Returns:
        normalized_rays_d: [..., 3] 標準化後的射線方向
    """
    return rays_d / torch.norm(rays_d, dim=-1, keepdim=True)


def sample_rays_batch(rays_o: torch.Tensor, rays_d: torch.Tensor, 
                     target_rgb: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
    """
    從射線中採樣批次數據
    
    Args:
        rays_o: [N, 3] 射線起點
        rays_d: [N, 3] 射線方向  
        target_rgb: [N, 3] 目標顏色
        batch_size: 批次大小
        
    Returns:
        batch_data: 批次數據字典
    """
    n_rays = rays_o.shape[0]
    indices = torch.randperm(n_rays)[:batch_size]
    
    return {
        'rays_o': rays_o[indices],
        'rays_d': rays_d[indices],
        'target_rgb': target_rgb[indices],
        'indices': indices
    } 