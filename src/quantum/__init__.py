"""
量子計算整合模組

為未來的量子增強功能預留接口：
- 量子位置編碼
- 量子神經層
- 量子採樣策略
- 量子優化算法
"""

# 當前為佔位符實現，未來將整合真正的量子組件
from .quantum_encoding import QuantumPositionalEncoder
from .quantum_layers import QuantumNeRFLayer
from .quantum_sampling import QuantumSampler
from .quantum_optimizer import QuantumOptimizer

__all__ = [
    "QuantumPositionalEncoder",
    "QuantumNeRFLayer",
    "QuantumSampler", 
    "QuantumOptimizer"
]

# 量子功能狀態
QUANTUM_AVAILABLE = False  # 當量子硬體/模擬器可用時設為 True

def is_quantum_available():
    """檢查量子功能是否可用"""
    return QUANTUM_AVAILABLE

def enable_quantum():
    """啟用量子功能（需要量子硬體支持）"""
    global QUANTUM_AVAILABLE
    # TODO: 檢查量子硬體/模擬器
    QUANTUM_AVAILABLE = True
    print("🌌 量子功能已啟用")

def disable_quantum():
    """禁用量子功能，回到經典計算"""
    global QUANTUM_AVAILABLE
    QUANTUM_AVAILABLE = False
    print("💻 已切換回經典計算") 