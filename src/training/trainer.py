"""
訓練器模組

實現 NeRF 的訓練流程：
- 訓練循環
- 驗證循環
- 模型保存和加載
- 訓練狀態管理
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import logging
from tqdm import tqdm
import json

from ..models.base import BaseModel
from ..models.encoding import PositionalEncoder
from ..rendering.volume_renderer import VolumeRenderer
from ..rendering.ray_sampling import RaySampler
from .losses import NeRFLoss
from .optimizers import create_optimizer, create_scheduler


class NeRFTrainer:
    """NeRF 訓練器"""
    
    def __init__(
        self,
        model: BaseModel,
        encoder: PositionalEncoder,
        renderer: VolumeRenderer,
        sampler: RaySampler,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化訓練器
        
        Args:
            model: NeRF 模型
            encoder: 位置編碼器
            renderer: 體積渲染器
            sampler: 射線採樣器
            train_loader: 訓練數據加載器
            val_loader: 驗證數據加載器
            config: 配置字典
        """
        self.model = model
        self.encoder = encoder
        self.renderer = renderer
        self.sampler = sampler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        
        # 設置設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.encoder.to(self.device)
        
        # 創建損失函數
        self.criterion = NeRFLoss(
            rgb_weight=self.config.get('rgb_weight', 1.0),
            depth_weight=self.config.get('depth_weight', 0.1),
            l1_weight=self.config.get('l1_weight', 0.0),
            l2_weight=self.config.get('l2_weight', 0.0),
            tv_weight=self.config.get('tv_weight', 0.0)
        )
        
        # 創建優化器
        self.optimizer = create_optimizer(
            list(self.model.parameters()),
            self.config
        )
        
        # 創建學習率調度器
        self.scheduler = create_scheduler(
            self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer,
            self.config
        )
        
        # 初始化訓練狀態
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')
        
        # 設置日誌
        self.setup_logging()
    
    def setup_logging(self):
        """設置日誌"""
        log_dir = self.config.get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'train.log')),
                logging.StreamHandler()
            ]
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """
        訓練一個 epoch
        
        Returns:
            metrics: 訓練指標字典
        """
        self.model.train()
        total_loss = 0
        metrics = {}
        
        with tqdm(self.train_loader, desc=f'Epoch {self.epoch}') as pbar:
            for batch in pbar:
                # 將數據移到設備
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # 前向傳播
                pred = self.model(
                    self.encoder(batch['points']),
                    self.encoder(batch['viewdirs']) if 'viewdirs' in batch else None
                )
                
                # 計算損失
                loss, loss_dict = self.criterion(
                    pred,
                    batch,
                    dict(self.model.named_parameters())
                )
                
                # 反向傳播和優化
                self.optimizer.step(loss)
                
                # 更新學習率
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # 更新指標
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    if k not in metrics:
                        metrics[k] = 0
                    metrics[k] += v
                
                # 更新進度條
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.get_lr():.6f}'
                })
                
                self.step += 1
        
        # 計算平均指標
        num_batches = len(self.train_loader)
        metrics = {k: v / num_batches for k, v in metrics.items()}
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        驗證模型
        
        Returns:
            metrics: 驗證指標字典
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        metrics = {}
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            # 將數據移到設備
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # 前向傳播
            pred = self.model(
                self.encoder(batch['points']),
                self.encoder(batch['viewdirs']) if 'viewdirs' in batch else None
            )
            
            # 計算損失
            loss, loss_dict = self.criterion(
                pred,
                batch,
                dict(self.model.named_parameters())
            )
            
            # 更新指標
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k not in metrics:
                    metrics[k] = 0
                metrics[k] += v
        
        # 計算平均指標
        num_batches = len(self.val_loader)
        metrics = {k: v / num_batches for k, v in metrics.items()}
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """
        保存檢查點
        
        Args:
            is_best: 是否為最佳模型
        """
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型狀態
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # 保存最新檢查點
        torch.save(
            checkpoint,
            os.path.join(checkpoint_dir, 'latest.pth')
        )
        
        # 保存最佳模型
        if is_best:
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, 'best.pth')
            )
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加載檢查點
        
        Args:
            checkpoint_path: 檢查點路徑
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加載模型狀態
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加載訓練狀態
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']
    
    def train(self, num_epochs: int):
        """
        訓練模型
        
        Args:
            num_epochs: 訓練輪數
        """
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # 訓練一個 epoch
            train_metrics = self.train_epoch()
            
            # 驗證
            val_metrics = self.validate()
            
            # 更新最佳驗證損失
            if val_metrics and val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.save_checkpoint(is_best=True)
            
            # 保存檢查點
            self.save_checkpoint()
            
            # 記錄指標
            metrics = {
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            }
            
            logging.info(
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['total_loss']:.4f}, "
                f"val_loss={val_metrics.get('total_loss', float('inf')):.4f}"
            )
            
            # 保存指標
            metrics_path = os.path.join(
                self.config.get('log_dir', 'logs'),
                'metrics.json'
            )
            
            with open(metrics_path, 'a') as f:
                json.dump(metrics, f)
                f.write('\n') 