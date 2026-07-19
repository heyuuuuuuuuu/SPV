"""分析工具: 混淆分析 + 对比评估"""

from .confusion import (
    load_and_preprocess, cosine_distance, analyze_resolution,
    compute_unified_threshold, apply_threshold, build_summary,
    print_char_report, plot_results,
)
from .compare import load_pixelized, reconstruct, compute_metrics as compare_metrics
