"""
損失函數模組

提供 NeRF 訓練所需的各種損失函數：
- RGB 損失
- 深度損失
- 正則化損失
- 總損失計算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class RGBLoss(nn.Module):
    """RGB 損失函數"""
    
    def __init__(self, reduction: str = 'mean'):
        """
        初始化 RGB 損失函數
        
        Args:
            reduction: 損失歸約方式 ('mean' 或 'sum')
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        計算 RGB 損失
        
        Args:
            pred: [B, 3] 預測的 RGB 值
            target: [B, 3] 目標 RGB 值
            mask: [B] 可選的遮罩
            
        Returns:
            loss: 標量損失值
        """
        loss = F.mse_loss(pred, target, reduction='none')
        
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


class DepthLoss(nn.Module):
    """深度損失函數"""
    
    def __init__(self, reduction: str = 'mean'):
        """
        初始化深度損失函數
        
        Args:
            reduction: 損失歸約方式 ('mean' 或 'sum')
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        計算深度損失
        
        Args:
            pred: [B] 預測的深度值
            target: [B] 目標深度值
            mask: [B] 可選的遮罩
            
        Returns:
            loss: 標量損失值
        """
        loss = F.l1_loss(pred, target, reduction='none')
        
        if mask is not None:
            loss = loss * mask
        
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


class RegularizationLoss(nn.Module):
    """正則化損失函數"""
    
    def __init__(
        self,
        l1_weight: float = 0.0,
        l2_weight: float = 0.0,
        tv_weight: float = 0.0
    ):
        """
        初始化正則化損失函數
        
        Args:
            l1_weight: L1 正則化權重
            l2_weight: L2 正則化權重
            tv_weight: 總變差正則化權重
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.tv_weight = tv_weight
    
    def forward(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        計算正則化損失
        
        Args:
            params: 模型參數字典
            
        Returns:
            loss: 標量損失值
        """
        loss = 0.0
        
        # L1 正則化
        if self.l1_weight > 0:
            l1_loss = sum(p.abs().sum() for p in params.values())
            loss += self.l1_weight * l1_loss
        
        # L2 正則化
        if self.l2_weight > 0:
            l2_loss = sum(p.pow(2).sum() for p in params.values())
            loss += self.l2_weight * l2_loss
        
        # 總變差正則化
        if self.tv_weight > 0:
            tv_loss = 0.0
            for p in params.values():
                if len(p.shape) == 4:  # 卷積層權重
                    tv_loss += (
                        torch.sum(torch.abs(p[:, :, :, :-1] - p[:, :, :, 1:])) +
                        torch.sum(torch.abs(p[:, :, :-1, :] - p[:, :, 1:, :]))
                    )
            loss += self.tv_weight * tv_loss
        
        return loss


class NeRFLoss(nn.Module):
    """NeRF 總損失函數"""
    
    def __init__(
        self,
        rgb_weight: float = 1.0,
        depth_weight: float = 0.1,
        l1_weight: float = 0.0,
        l2_weight: float = 0.0,
        tv_weight: float = 0.0
    ):
        """
        初始化 NeRF 總損失函數
        
        Args:
            rgb_weight: RGB 損失權重
            depth_weight: 深度損失權重
            l1_weight: L1 正則化權重
            l2_weight: L2 正則化權重
            tv_weight: 總變差正則化權重
        """
        super().__init__()
        self.rgb_loss = RGBLoss()
        self.depth_loss = DepthLoss()
        self.reg_loss = RegularizationLoss(l1_weight, l2_weight, tv_weight)
        
        self.rgb_weight = rgb_weight
        self.depth_weight = depth_weight
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        params: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        計算總損失
        
        Args:
            pred: 預測值字典，包含 'rgb' 和 'depth'
            target: 目標值字典，包含 'rgb' 和 'depth'
            params: 模型參數字典
            mask: 可選的遮罩
            
        Returns:
            total_loss: 總損失值
            loss_dict: 各項損失的字典
        """
        # RGB 損失
        rgb_loss = self.rgb_loss(pred['rgb'], target['rgb'], mask)
        
        # 深度損失
        depth_loss = self.depth_loss(pred['depth'], target['depth'], mask)
        
        # 正則化損失
        reg_loss = self.reg_loss(params)
        
        # 總損失
        total_loss = (
            self.rgb_weight * rgb_loss +
            self.depth_weight * depth_loss +
            reg_loss
        )
        
        # 損失字典
        loss_dict = {
            'total_loss': total_loss.item(),
            'rgb_loss': rgb_loss.item(),
            'depth_loss': depth_loss.item(),
            'reg_loss': reg_loss.item()
        }
        
        return total_loss, loss_dict