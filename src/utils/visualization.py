"""
可視化工具模組

提供 NeRF 訓練過程中的各種可視化功能：
- 訓練曲線繪製
- 渲染視頻生成
- 3D 場景可視化
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Optional, Tuple
import imageio
from pathlib import Path


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "訓練指標"
) -> None:
    """
    繪製訓練過程中的各種指標曲線
    
    Args:
        metrics: 包含各種指標的字典，每個指標是一個數值列表
        save_path: 保存圖像的路徑（可選）
        title: 圖表標題
    """
    plt.figure(figsize=(12, 6))
    
    for name, values in metrics.items():
        plt.plot(values, label=name)
    
    plt.title(title)
    plt.xlabel("迭代次數")
    plt.ylabel("指標值")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def render_video(
    frames: List[np.ndarray],
    save_path: str,
    fps: int = 30
) -> None:
    """
    將一系列幀渲染為視頻
    
    Args:
        frames: 圖像幀列表，每個幀是一個 numpy 數組
        save_path: 保存視頻的路徑
        fps: 每秒幀數
    """
    # 確保所有幀都是 uint8 類型
    frames = [frame.astype(np.uint8) for frame in frames]
    
    # 創建保存目錄
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存視頻
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"✅ 視頻已保存到: {save_path}")


def visualize_3d_points(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> None:
    """
    可視化 3D 點雲
    
    Args:
        points: [N, 3] 點雲座標
        colors: [N, 3] 點雲顏色（可選）
        save_path: 保存圖像的路徑（可選）
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is not None:
        scatter = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors, s=1, alpha=0.5
        )
    else:
        scatter = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=1, alpha=0.5
        )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D 點雲可視化')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_camera_poses(
    poses: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    可視化相機位姿
    
    Args:
        poses: [N, 4, 4] 相機位姿矩陣
        save_path: 保存圖像的路徑（可選）
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取相機位置
    positions = poses[:, :3, 3]
    
    # 繪製相機位置
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', s=50)
    
    # 繪製相機方向
    for i in range(len(poses)):
        # 相機的 z 軸方向
        direction = poses[i, :3, 2]
        # 繪製方向箭頭
        ax.quiver(
            positions[i, 0], positions[i, 1], positions[i, 2],
            direction[0], direction[1], direction[2],
            length=0.1, color='r'
        )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('相機位姿可視化')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_rays(
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    可視化射線
    
    Args:
        rays_o: [N, 3] 射線起點
        rays_d: [N, 3] 射線方向
        save_path: 保存圖像的路徑（可選）
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 繪製射線起點
    ax.scatter(rays_o[:, 0], rays_o[:, 1], rays_o[:, 2], c='b', s=50)
    
    # 繪製射線方向
    for i in range(len(rays_o)):
        ax.quiver(
            rays_o[i, 0], rays_o[i, 1], rays_o[i, 2],
            rays_d[i, 0], rays_d[i, 1], rays_d[i, 2],
            length=0.1, color='r'
        )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('射線可視化')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 