"""
核心流水线: 字体渲染 → 复杂度计算 → 像素化 → 自适应分辨率 → 磷光点仿真
"""

from .render import CharRenderer, load_chars_from_file, batch_render
from .complexity import compute_complexity, assign_levels, batch_compute
from .pixelize import pixelize, batch_pixelize
from .adaptive import adaptive_pixelize, pixelize_cell
from .phosphene import gaussian_phosphene, normalize_to_uint8, render_and_save, batch_render as batch_render_spv
