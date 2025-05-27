"""
Quantum NeRF 專案核心模組

這個包包含了 NeRF 模型的所有核心組件，設計為模組化和可擴展的架構。

主要模組：
- models: NeRF 模型定義
- utils: 工具函數和輔助類
- training: 訓練相關功能
- rendering: 渲染相關功能
- quantum: 量子計算整合模組
"""

__version__ = "0.1.0"
__author__ = "Quantum NeRF Team"

# 導入主要組件
from . import models
from . import utils
from . import training
from . import rendering
from . import quantum

__all__ = [
    "models",
    "utils", 
    "training",
    "rendering",
    "quantum"
] 