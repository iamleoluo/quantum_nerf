"""
優化器模組

提供 NeRF 訓練所需的優化器：
- Adam 優化器
- 學習率調度器
- 梯度裁剪
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional, List, Union


class AdamOptimizer:
    """Adam 優化器封裝"""
    
    def __init__(
        self,
        params: List[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        max_grad_norm: Optional[float] = None
    ):
        """
        初始化 Adam 優化器
        
        Args:
            params: 模型參數列表
            lr: 學習率
            betas: Adam 優化器的 beta 參數
            eps: 數值穩定性參數
            weight_decay: 權重衰減係數
            amsgrad: 是否使用 AMSGrad 變體
            max_grad_norm: 梯度裁剪閾值
        """
        self.optimizer = optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        self.max_grad_norm = max_grad_norm
    
    def step(self, loss: torch.Tensor) -> None:
        """
        執行一步優化
        
        Args:
            loss: 損失值
        """
        # 計算梯度
        loss.backward()
        
        # 梯度裁剪
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.optimizer.param_groups[0]['params'],
                self.max_grad_norm
            )
        
        # 更新參數
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def get_lr(self) -> float:
        """
        獲取當前學習率
        
        Returns:
            lr: 當前學習率
        """
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self) -> Dict[str, Any]:
        """
        獲取優化器狀態
        
        Returns:
            state: 優化器狀態字典
        """
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        加載優化器狀態
        
        Args:
            state_dict: 優化器狀態字典
        """
        self.optimizer.load_state_dict(state_dict)


class WarmupCosineScheduler(_LRScheduler):
    """帶預熱的餘弦學習率調度器"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        初始化調度器
        
        Args:
            optimizer: 優化器
            warmup_steps: 預熱步數
            max_steps: 總步數
            min_lr: 最小學習率
            last_epoch: 上一個 epoch
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """
        獲取當前學習率
        
        Returns:
            lr: 當前學習率列表
        """
        if self.last_epoch < self.warmup_steps:
            # 線性預熱
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # 餘弦衰減
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            alpha = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
            return [self.min_lr + (base_lr - self.min_lr) * alpha for base_lr in self.base_lrs]


class ExponentialScheduler(_LRScheduler):
    """指數衰減學習率調度器"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.1,
        last_epoch: int = -1
    ):
        """
        初始化調度器
        
        Args:
            optimizer: 優化器
            gamma: 衰減係數
            last_epoch: 上一個 epoch
        """
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """
        獲取當前學習率
        
        Returns:
            lr: 當前學習率列表
        """
        return [base_lr * (self.gamma ** self.last_epoch) for base_lr in self.base_lrs]


def create_optimizer(
    params: List[torch.nn.Parameter],
    config: Dict[str, Any]
) -> Union[AdamOptimizer, torch.optim.Optimizer]:
    """
    創建優化器
    
    Args:
        params: 模型參數列表
        config: 配置字典
        
    Returns:
        optimizer: 優化器實例
    """
    optimizer_type = config.get('optimizer_type', 'adam')
    
    if optimizer_type == 'adam':
        return AdamOptimizer(
            params,
            lr=config.get('lr', 1e-3),
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=config.get('weight_decay', 0.0),
            amsgrad=config.get('amsgrad', False),
            max_grad_norm=config.get('max_grad_norm', None)
        )
    else:
        raise ValueError(f"不支持的優化器類型: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any]
) -> Optional[_LRScheduler]:
    """
    創建學習率調度器
    
    Args:
        optimizer: 優化器實例
        config: 配置字典
        
    Returns:
        scheduler: 學習率調度器實例
    """
    scheduler_type = config.get('scheduler_type', None)
    
    if scheduler_type == 'warmup_cosine':
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=config.get('warmup_steps', 1000),
            max_steps=config.get('max_steps', 100000),
            min_lr=config.get('min_lr', 0.0)
        )
    elif scheduler_type == 'exponential':
        return ExponentialScheduler(
            optimizer,
            gamma=config.get('gamma', 0.1)
        )
    else:
        return None 