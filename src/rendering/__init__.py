"""
渲染相關模組

包含體積渲染的所有組件：
- 體積渲染器
- 射線採樣
- 渲染方程實現
"""

from .volume_renderer import VolumeRenderer
from .ray_sampling import RaySampler, stratified_sampling
from .render_utils import render_rays, render_image

__all__ = [
    "VolumeRenderer",
    "RaySampler",
    "stratified_sampling",
    "render_rays",
    "render_image"
] 