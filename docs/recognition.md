# recognition.py — 自排除最近邻混淆分析

## 概述

对 SPV（Simulated Prosthetic Vision）磷光点仿真图像进行 **自排除最近邻混淆分析（Self-Excluded Nearest-Neighbor Confusion Analysis）**，评估不同分辨率下汉字之间的可区分性（混淆风险）。

- **核心问题**：在给定分辨率下，某个汉字的 SPV 图像是否容易被误认为另一个汉字？
- **核心方法**：排除自身后，用余弦距离寻找最近邻，以 8×8 分布确定统一阈值，跨分辨率比较混淆率。

---

## 算法流程

```
Step 1: 逐分辨率加载 SPV 图像
        ↓ 转为 128×128 灰度 → 归一化 [0,1] → flatten 成向量
Step 2: 对每个汉字，排除自身
        ↓ 计算与其他所有汉字的余弦距离
Step 3: 找 nearest_other（最近邻）和 second_other（第二近邻）
        ↓ 记录 nearest_distance、second_distance、margin
Step 4: 以 8×8 的 nearest_distance 分布
        ↓ 取前 risk_percentile% 分位数 → 统一阈值 T
Step 5: 同一 T 应用于全部分辨率
        ↓ nearest_distance < T → is_high_confusion_risk = True
Step 6: 输出 CSV + 可视化图表
```

---


### 自排除（Self-Exclusion）

每个汉字在计算最近邻时，**不将自身纳入候选集**，避免自身距离=0 的问题。

```python
dists[i] = np.inf   # 排除自身，距离设为无穷大
```

### 余弦距离（Cosine Distance）

\[
\text{cosine\_distance}(A, B) = 1 - \frac{A \cdot B}{\|A\| \cdot \|B\|}
\]

- 取值范围 [0, 2]
- 0 = 完全相同方向（最相似）
- 2 = 完全相反方向

### 统一阈值 T

以 8×8（最常用参考分辨率）的最近邻距离分布为基础，取距离最小的前 `risk_percentile`% 的分位数作为阈值 T。同一 T 固定用于所有分辨率，保证可比性。

```
T = percentile(nearest_distances_8x8, risk_percentile)
   = 距离最小的前 risk_percentile% 汉字对应的最近邻距离
```

**计算公式**：
1. 收集 8×8 分辨率下所有 N 个汉字的 `nearest_distance` → 排序：`d₁ ≤ d₂ ≤ ... ≤ dₙ`
2. 取索引 `k = floor(N × risk_percentile / 100)`
3. `T = dₖ`（第 k 个最小距离，即第 risk_percentile 百分位数）

**实例（T = 0.0438）**：
- 8×8 分辨率下共 N = 999 个汉字
- `risk_percentile` = 20，`k = floor(999 × 20 / 100) = floor(199.8) = 199`
- T = 排序后第 199 个（0-indexed）最小 nearest_distance ≈ **0.043804**（四舍五入 0.0438）
- 含义：999 个字符中，约 200 个（前 20%）具有 ≤ 0.0438 的最近邻距离，这些被定义为"高风险混淆"（`is_high_confusion_risk = True`）

### 混淆判定

```python
is_high_confusion_risk = nearest_distance < T
```

---

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input-dir` | str | `E:/dataset/char_spv_fixed` | SPV 图像根目录 |
| `--output-dir` | str | `E:/results/confusion` | 输出目录 |
| `--resolutions` | list | `6x6 8x8 10x10 12x14` | 分析的分辨率列表 |
| `--threshold-resolution` | str | `8x8` | 确定阈值 T 的参考分辨率（基准） |
| `--risk-percentile` | float | `20.0` | 高风险分位数百分比 |
| `--target-size` | int | `128` | resize 目标尺寸 |

---

## 输出文件

### 1. `per_character_confusion.csv`

逐汉字详细结果：

| 列名 | 类型 | 说明 |
|------|------|------|
| `resolution` | str | 分辨率（如 `6x6`） |
| `char` | str | 当前汉字 |
| `nearest_other` | str | 最近邻汉字 |
| `nearest_distance` | float | 到最近邻的余弦距离 |
| `second_other` | str | 第二近邻汉字 |
| `second_distance` | float | 到第二近邻的余弦距离 |
| `margin` | float | `second_distance - nearest_distance` |
| `threshold` | float | 统一阈值 T |
| `is_high_confusion_risk` | bool | 是否高风险混淆 |

**示例**：

| resolution | char | nearest_other | nearest_distance | second_other | second_distance | margin | threshold | is_high_confusion_risk |
|:----------:|:----:|:-------------:|:----------------:|:------------:|:---------------:|:------:|:---------:|:----------------------:|
| 6x6 | 一 | 千 | 0.3145 | 午 | 0.3452 | 0.0307 | 0.0438 | False |
| 6x6 | 丁 | 了 | 0.0270 | 于 | 0.1730 | 0.1461 | 0.0438 | True |
| 8x8 | 睡 | 履 | 0.0372 | 蒙 | 0.0487 | 0.0115 | 0.0438 | True |

### 2. `resolution_summary.csv`

分辨率汇总：

| 列名 | 类型 | 说明 |
|------|------|------|
| `resolution` | str | 分辨率 |
| `num_chars` | int | 汉字总数 |
| `num_high_confusion` | int | 高风险混淆字数 |
| `confusion_rate` | float | 混淆率 |
| `mean_nearest_distance` | float | 平均最近邻距离 |
| `median_nearest_distance` | float | 中位数最近邻距离 |
| `mean_margin` | float | 平均 margin |

### 3. 可视化图表

| 文件 | 说明 |
|------|------|
| `confusion_rate_bar.png` | 各分辨率混淆率柱状图 |
| `mean_nearest_distance_line.png` | 平均最近邻距离折线图 |
| `nearest_distance_histogram.png` | 各分辨率距离分布直方图（红区=高风险） |

---

## 使用示例

```bash
# 默认参数运行
python recognition.py

# 自定义阈值分位数（前 15% 为高风险）
python recognition.py --risk-percentile 15

# 自定义输入输出
python recognition.py \
    --input-dir E:/dataset/char_spv_fixed \
    --output-dir E:/results/confusion_custom \
    --resolutions 6x6 8x8 10x10 12x14 \
    --risk-percentile 20

# 仅分析 8x8 和 12x14
python recognition.py --resolutions 8x8 12x14
```

---

## 结果解读

典型运行结果（8×8 基准）：

```
T = 0.043804 (基于 8x8, top 20%)

分辨率     混淆率    高风险     平均距离    平均margin
6x6        82.3%    822/999    0.0171      0.0022
8x8        20.0%    200/999    0.0538      0.0104
10x10       1.5%     15/999    0.0927      0.0148
12x14       0.0%      0/999    0.1312      0.0195
```

**解读**：
- **6×6**：82.3% 汉字存在高风险混淆，平均最近邻距离仅 0.017 → 分辨率严重不足，几乎不具识别能力
- **8×8**：混淆率 20.0%（这正是 `risk_percentile` 定义所致），平均距离 0.054 → 约 1/5 汉字存在高风险，整体可用性有限
- **10×10**：混淆率降至 1.5%，平均距离 0.093 → 仅极少数复杂字仍有混淆风险
- **12×14**：零混淆，平均距离 0.131 → 分辨率充足，所有汉字均安全

**相比于 6×6 基准的策略意义**：
> 以 6×6 为基准时，T ≈ 0.0165 过于严格，导致 8×8 仅标记 1.1% 为高风险，实际上许多混淆被隐藏。
> 以 8×8 为基准时，T ≈ 0.0438 更准确地反映了 8×8 下真实的混淆结构：约 1/5 的汉字在此分辨率下存在识别风险。
> 这为"复杂度自适应分辨率分配"提供了更保守、更可靠的量化依据。

**论文价值**：
> 以 8×8 为基准重新校准阈值，揭示 6×6 下 82.3% 的极高混淆率 → 6×6 分辨率不可行；8×8 下 20% 混淆率 → 可用于简单字但对复杂字不足；10×10 下仅 1.5% 混淆率 → 基本安全；12×14 完全消除混淆。这为"复杂度自适应分辨率"提供分层依据：简单字可用 8×8，中等字需 10×10，复杂字需 ≥12×14。