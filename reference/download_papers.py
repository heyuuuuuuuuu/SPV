#!/usr/bin/env python3
"""
下载 SPV (Simulated Prosthetic Vision) 汉字识别领域的相关文献
=========================================================
按主题分组，优先下载开放获取 (Open Access) 版本。
"""

import os
import urllib.request
import time

OUT = os.path.dirname(os.path.abspath(__file__))

# 避免被屏蔽
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
HEADERS = {"User-Agent": UA, "Accept": "application/pdf,*/*"}


def download(url: str, filename: str, desc: str = ""):
    """下载文件并带重试"""
    path = os.path.join(OUT, filename)
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        print(f"  [✓] 已存在: {filename}")
        return True
    print(f"  [→] 下载: {filename}  {desc}")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            if len(data) < 5000:
                # 可能是 HTML 错误页面
                if b"<!DOCTYPE" in data[:200] or b"<html" in data[:200]:
                    print(f"    ✗ 返回的是 HTML 页面 (可能需付费或跳转)")
                    return False
        with open(path, "wb") as f:
            f.write(data)
        print(f"    ✓ 完成 ({len(data)/1024:.0f} KB)")
        return True
    except Exception as e:
        print(f"    ✗ 失败: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# A组: SPV + 汉字识别核心文献 (直接相关)
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("A组: SPV + 汉字识别核心文献")
print("=" * 60)

# A1: Chai et al. 2007 — 像素化汉字识别 (奠基性工作)
download(
    "https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1525-1594.2007.00355.x",
    "A1_Chai_2007_Recognition_Pixelized_Chinese_Characters_SPV.pdf",
    "Artif Organs 2007, 被引>150次"
)

# A2: Zhao et al. 2011 — 像素化段落阅读
download(
    "https://iovs.arvojournals.org/article.aspx?articleid=2188609",
    "A2_Zhao_2011_Reading_Pixelized_Paragraphs_Chinese.pdf",
    "IOVS 2011"
)

# A3: Zhao et al. 2011 — 模拟光幻视汉字识别
download(
    "https://iovs.arvojournals.org/article.aspx?articleid=2166434",
    "A3_Zhao_2011_Chinese_Character_Recognition_Phosphene_Maps.pdf",
    "IOVS 2011"
)

# A4: Lu et al. 2013 — 优化汉字显示
download(
    "https://iovs.arvojournals.org/article.aspx?articleid=2189383",
    "A4_Lu_2013_Optimizing_Chinese_Character_Displays_SPV.pdf",
    "IOVS 2013"
)

# A5: Fu et al. 2006 — 有限像素阅读心理学
download(
    "https://europepmc.org/articles/pmc1810364?pdf=render",
    "A5_Fu_2006_Psychophysics_Reading_Limited_Pixels.pdf",
    "Vision Res 2006"
)

# A6: Kim & Park 2017 — 时空像素化提升字符识别
download(
    "https://www.mdpi.com/1424-8220/17/10/2439/pdf",
    "A6_Kim_2017_Spatiotemporal_Pixelization_Character_Recognition.pdf",
    "Sensors 2017"
)

# ═══════════════════════════════════════════════════════════
# B组: 近年 SPV 综述与前沿 (2020–2025)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("B组: 近年 SPV 综述与前沿")
print("=" * 60)

# B1: Wang et al. 2022 — 视觉假体信息处理综述 (被引>60)
download(
    "https://www.mdpi.com/1424-8220/22/17/6544/pdf",
    "B1_Wang_2022_Clinical_Progress_Optimization_Visual_Prostheses.pdf",
    "Sensors 2022 (综述, 涵盖汉字识别)"
)

# B2: de Ruyter van Steveninck et al. 2022 — SPV室内移动
download(
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8784248/pdf/",
    "B2_deRuyter_2022_Real_World_Indoor_Mobility_SPV.pdf",
    "J Vis 2022"
)

# B3: Elnabawy et al. 2022 — 混合现实SPV物体识别
download(
    "https://biomedical-engineering-online.biomedcentral.com/counter/pdf/10.1186/s12938-022-01059-7.pdf",
    "B3_Elnabawy_2022_Object_Recognition_Localization_Visual_Prostheses.pdf",
    "BioMed Eng OnLine 2022"
)

# B4: Xia et al. 2022 — 有限像素面部语义翻译
download(
    "https://pdf.sciencedirectassets.com/271625/1-s2.0-S0020025522X00165/1-s2.0-S0020025522008123/am.pdf",
    "B4_Xia_2022_Semantic_Translation_Face_Image_SPV.pdf",
    "Information Sciences 2022"
)

# B5: Wang et al. 2021 — 计算机视觉在视觉假体中的应用
download(
    "https://onlinelibrary.wiley.com/doi/pdf/10.1111/aor.13935",
    "B5_Wang_2021_Computer_Vision_Visual_Prosthesis.pdf",
    "Artif Organs 2021"
)

# ═══════════════════════════════════════════════════════════
# C组: 深度学习+SPV / 生物仿真 (2022–2024)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("C组: 深度学习+SPV / 生物仿真")
print("=" * 60)

# C1: Kasowski & Beyeler 2022 — VR-SPV 开源工具箱
download(
    "https://arxiv.org/pdf/2203.05675",
    "C1_Kasowski_2022_VR_SPV_Immersive_Bionic_Vision.pdf",
    "ACM AH 2022 / arXiv"
)

# C2: Granley et al. 2022 — 混合神经自编码器 (NeurIPS)
download(
    "https://arxiv.org/pdf/2211.05302",
    "C2_Granley_2022_Hybrid_Neural_Autoencoders_Stimulus_Encoding.pdf",
    "NeurIPS 2022"
)

# C3: Granley & Beyeler 2022 — 脑启发CNN建模视觉假体
download(
    "https://openreview.net/pdf?id=3zXj_tGcEEx",
    "C3_Granley_2022_Adapting_Brainlike_NN_Cortical_Prostheses.pdf",
    "NeurIPS SVRHM Workshop 2022"
)

# C4: van Steveninck et al. 2024 — 生物可信的光幻视仿真 (eLife)
download(
    "https://elifesciences.org/articles/85812.pdf",
    "C4_vanSteveninck_2024_Biologically_Plausible_Phosphene_Simulation.pdf",
    "eLife 2024"
)

# C5: Sanchez-Garcia et al. 2022 — Checkerboard 光栅模式
download(
    "https://openaccess.thecvf.com/content/CVPR2022W/NeuroVision/papers/Sanchez-Garcia_Simulated_Prosthetic_Vision_Confirms_Checkerboard_as_an_Effective_Raster_Pattern_for_CVPRW_2022_paper.pdf",
    "C5_SanchezGarcia_2022_Checkerboard_Raster_Pattern_Epiretinal.pdf",
    "CVPR NeuroVision Workshop 2022"
)

# C6: Sanchez-Garcia 2022 — 博士论文 (Egocentric CV + SPV)
download(
    "https://zaguan.unizar.es/record/112137/files/TESIS-2022-078.pdf",
    "C6_SanchezGarcia_2022_PhD_Egocentric_CV_ML_SPV.pdf",
    "PhD Thesis, Univ Zaragoza 2022"
)

# ═══════════════════════════════════════════════════════════
# D组: 硬件分辨率 & 电极阵列优化 (2022–2024)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("D组: 硬件分辨率 & 电极阵列优化")
print("=" * 60)

# D1: Wang et al. 2022 — 电子光感受器匹配自然分辨率 (Nature Comms, 被引>80)
download(
    "https://www.nature.com/articles/s41467-022-34387-y.pdf",
    "D1_Wang_2022_Electronic_Photoreceptors_Prosthetic_Visual_Acuity.pdf",
    "Nature Communications 2022"
)

# D2: Chen et al. 2022 — 光伏植入模拟器
download(
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9583482/pdf/",
    "D2_Chen_2022_Photovoltaic_Implant_Simulator_Resolution_Limits.pdf",
    "J Neural Eng 2022"
)

# D3: Wang et al. 2022 — PRIMA像素极限
download(
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9583173/pdf/",
    "D3_Wang_2022_Pixel_Size_Limit_PRIMA_Implants.pdf",
    "J Neural Eng 2022"
)

# D4: Beyeler et al. 2022 — 电极排列贪心优化
download(
    "https://arxiv.org/pdf/2204.13766",
    "D4_Bruce_2022_Greedy_Optimization_Electrode_Arrangement.pdf",
    "MICCAI 2022"
)

# D5: Bhuckory et al. 2023 — 蜂窝状植入物细胞迁移
download(
    "https://www.pnas.org/doi/pdf/10.1073/pnas.2307380120",
    "D5_Bhuckory_2023_Cellular_Migration_Honeycomb_Prosthesis.pdf",
    "PNAS 2023"
)

# ═══════════════════════════════════════════════════════════
# E组: 深度学习场景简化 + 端到端优化 (2021–2024)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("E组: 深度学习场景简化 + 端到端优化")
print("=" * 60)

# E1: Han et al. 2021 — 深度学习场景简化
download(
    "https://arxiv.org/pdf/2102.00297",
    "E1_Han_2021_Deep_Learning_Scene_Simplification_Bionic_Vision.pdf",
    "ACM CHI 2021 / arXiv"
)

# E2: de Ruyter van Steveninck et al. 2022 — 端到端SPV优化
download(
    "https://arxiv.org/pdf/2205.08412",
    "E2_deRuyter_2022_End_to_End_Optimization_SPV.pdf",
    "arXiv 2022"
)

# E3: Wu et al. 2023 — 深度学习in silico框架
download(
    "https://arxiv.org/pdf/2302.03570",
    "E3_Wu_2023_Deep_Learning_in_silico_Framework_Phosphene.pdf",
    "arXiv 2023"
)

# E4: Schoinas et al. 2025 — HILO人机交互优化
download(
    "https://arxiv.org/pdf/2502.00177",
    "E4_Schoinas_2025_Human_in_the_Loop_Optimization_SPV.pdf",
    "arXiv 2025"
)

# ═══════════════════════════════════════════════════════════
# F组: 视觉假体AI时代综述 (2024–2025)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("F组: 视觉假体AI时代综述")
print("=" * 60)

# F1: Review 2024 — 视网膜假体视觉场景预处理综述
download(
    "https://onlinelibrary.wiley.com/doi/pdf/10.1002/aisy.202400175",
    "F1_Review_2024_Preprocessing_Visual_Scenes_Retinal_Prosthesis.pdf",
    "Adv Intell Syst 2024"
)

# F2: Review 2025 — AI时代视觉假体
download(
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12405713/pdf/",
    "F2_Review_2025_Visual_Prostheses_AI_Era.pdf",
    "PMC 2025"
)

# F3: Palanker et al. 2025 — 电子视网膜假体综述
download(
    "https://web.stanford.edu/~palanker/publications/Prosthesis_annurev_vision_2025.pdf",
    "F3_Palanker_2025_Restoration_Sight_Electronic_Retinal_Prostheses.pdf",
    "Annu Rev Vis Sci 2025"
)

# F4: Lozano et al. 2024 — 神经义肢视觉神经形态学
download(
    "https://arxiv.org/pdf/2311.01192",
    "F4_Lozano_2023_Neuromorphic_Prosthetic_Vision.pdf",
    "arXiv 2023"
)


# ═══════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("下载完成! 文件目录:")
print("=" * 60)
for f in sorted(os.listdir(OUT)):
    if f.endswith(".pdf"):
        size = os.path.getsize(os.path.join(OUT, f))
        print(f"  {f}  ({size/1024:.0f} KB)")

print(f"\n目录: {OUT}")
