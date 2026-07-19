# SPV (Simulated Prosthetic Vision) 汉字识别 — 参考文献库

> 研究领域: 模拟假体视觉下的汉字可识别性 | 低分辨率像素化 | 高斯光斑仿真 | CNN识别 | 复杂度自适应

---

## 📚 已下载文献 (14篇, ~48MB)

### A. SPV + 汉字/字符识别
| # | 文件 | 作者/来源 | 年份 | 说明 |
|---|------|----------|------|------|
| A6 | `A6_Kim_2017_Spatiotemporal_Pixelization.pdf` | Kim & Park, *Sensors* | 2017 | 时空像素化提升字符识别率 |

### B. SPV 综述与评估
| # | 文件 | 作者/来源 | 年份 | 说明 |
|---|------|----------|------|------|
| B1 | `B1_Wang_2022_Clinical_Progress_Visual_Prostheses.pdf` | Wang et al., *Sensors* | 2022 | 视觉假体信息处理优化综述 (被引>60, 含汉字识别回顾) |
| B3 | `B3_Elnabawy_2022_Object_Recognition_MR_SPV.pdf` | Elnabawy et al., *BioMed Eng OnLine* | 2022 | 混合现实SPV物体识别与定位 |

### C. 深度学习 + SPV 仿真
| # | 文件 | 作者/来源 | 年份 | 说明 |
|---|------|----------|------|------|
| C1 | `C1_Kasowski_2022_VR_SPV_Immersive_Bionic_Vision.pdf` | Kasowski & Beyeler, *ACM AHs* | 2022 | VR-SPV 开源VR工具箱 (被引>25) |
| C2 | `C2_Granley_2022_Hybrid_Neural_Autoencoders_Stimulus_Encoding.pdf` | Granley et al., *NeurIPS* | 2022 | 混合神经自编码器用于感觉神经假体刺激编码 |
| C4 | `C4_vanSteveninck_2024_Biologically_Plausible_Phosphene_Simulation.pdf` | van Steveninck et al., *eLife* | 2024 | 生物可信光幻视仿真+可微分优化皮层视觉假体 |
| C6 | `C6_SanchezGarcia_2022_PhD_Egocentric_CV_ML_SPV.pdf` | Sánchez García, 博士论文, Univ. Zaragoza | 2022 | 以自我为中心的计算机视觉与机器学习用于SPV (含完整SPV综述) |

### D. 电极/硬件分辨率优化
| # | 文件 | 作者/来源 | 年份 | 说明 |
|---|------|----------|------|------|
| D4 | `D4_Bruce_2022_Greedy_Optimization_Electrode_Arrangement.pdf` | Bruce & Beyeler, *MICCAI* | 2022 | 贪心优化视网膜前植入电极排列 |

### E. 端到端深度学习优化
| # | 文件 | 作者/来源 | 年份 | 说明 |
|---|------|----------|------|------|
| E1 | `E1_Han_2021_Deep_Learning_Scene_Simplification_Bionic_Vision.pdf` | Han et al., arXiv (*ACM CHI*) | 2021 | 深度学习场景简化用于仿生视觉 (被引>70) |
| E2 | `E2_deRuyter_2022_End_to_End_Optimization_SPV.pdf` | de Ruyter van Steveninck et al., arXiv | 2022 | 端到端优化SPV图像处理 |
| E3 | `E3_Wu_2023_Deep_Learning_in_silico_Framework_Phosphene.pdf` | Wu et al., arXiv | 2023 | 深度学习in silico光幻视优化框架 (被引>10) |
| E4 | `E4_Schoinas_2025_Human_in_the_Loop_Optimization_SPV.pdf` | Schoinas et al., arXiv | 2025 | 人在回路优化视觉假体刺激策略 |

### F. 综述与前瞻
| # | 文件 | 作者/来源 | 年份 | 说明 |
|---|------|----------|------|------|
| F3 | `F3_Palanker_2025_Restoration_Sight_Electronic_Retinal_Prostheses.pdf` | Palanker et al., *Annu Rev Vis Sci* | 2025 | 电子视网膜假体恢复视力的权威综述 |
| F4 | `F4_Lozano_2023_Neuromorphic_Prosthetic_Vision.pdf` | Lozano et al., arXiv | 2023 | 神经形态视觉假体 |

---

## 📋 完整文献清单 (含付费论文，供检索参考)

### 🔴 核心必读: SPV汉字识别 (奠基性工作)

1. **Chai, X., Yu, W., Wang, J., Zhao, Y., Cai, C., & Ren, Q. (2007).**
   *Recognition of pixelized Chinese characters using simulated prosthetic vision.*
   Artificial Organs, 31(3), 175–182. DOI: [10.1111/j.1525-1594.2007.00355.x](https://doi.org/10.1111/j.1525-1594.2007.00355.x)
   - **被引 >150 | 奠基性工作**: 系统研究像素化分辨率、笔画数和字体对汉字识别准确率的影响

2. **Zhao, Y., Lu, Y., Zhao, J., Wang, K., Ren, Q., Wu, K., & Chai, X. (2011).**
   *Reading pixelized paragraphs of Chinese characters using simulated prosthetic vision.*
   Investigative Ophthalmology & Visual Science, 52(8), 5987–5994. DOI: [10.1167/iovs.10-5291](https://doi.org/10.1167/iovs.10-5291)
   - **被引 ~60**: 像素大小、分辨率、像素丢失率和灰度级对中文段落阅读的影响

3. **Zhao, Y., Lu, Y., Zhou, C., Chen, Y., Ren, Q., & Chai, X. (2011).**
   *Chinese character recognition using simulated phosphene maps.*
   Investigative Ophthalmology & Visual Science, 52(6), 3404–3412. DOI: [10.1167/iovs.10-5290](https://doi.org/10.1167/iovs.10-5290)
   - 光幻视地图下汉字识别，研究不同电极网格和光幻视模型

4. **Lu, Y., Kan, H., Liu, J., Wang, J., Tao, C., Chen, Y., Ren, Q., Hu, J., & Chai, X. (2013).**
   *Optimizing Chinese character displays improves recognition and reading performance of simulated irregular phosphene maps.*
   IOVS, 54(4), 2918–2926. DOI: [10.1167/iovs.12-11139](https://doi.org/10.1167/iovs.12-11139)
   - 优化不规则光幻视地图中汉字显示提升识别和阅读表现

5. **Jing, J., Zhao, Y., et al. (2023).**
   *EEG signals of Chinese character cognition under simulated prosthetic vision.*
   IEEE 3rd Int. Conf. on Electronic Technology, Communication and Information (ICETCI), 343–348.
   - ERP技术研究假体视觉下汉字认知的神经机制

### 🟡 近年SPV综述

6. **Wang, J., et al. (2022). ✅已下载**
   *Clinical Progress and Optimization of Information Processing in Artificial Visual Prostheses.*
   Sensors, 22(17), 6544. DOI: [10.3390/s22176544](https://doi.org/10.3390/s22176544)
   - 涵盖多种视觉假体信息处理优化方法，含汉字识别章节

7. **Pre-processing Visual Scenes for Retinal Prosthesis Systems: A Comprehensive Review. (2024)**
   Advanced Intelligent Systems. DOI: [10.1002/aisy.202400175](https://doi.org/10.1002/aisy.202400175)
   - 视网膜假体视觉场景预处理综述，含深度学习和图像处理方法

8. **Visual Prostheses in the Era of Artificial Intelligence Technology. (2025)**
   Nature Communications (Review). DOI: [PMC12405713](https://pmc.ncbi.nlm.nih.gov/articles/PMC12405713/)
   - AI驱动的视觉假体综述：刺激优化、场景理解、深度学习应用

### 🟢 深度学习 + SPV 前沿 (2020–2024)

9. **de Ruyter van Steveninck, J., van Gestel, T., Koenders, P., et al. (2022).**
   *Real-world indoor mobility with simulated prosthetic vision: the benefits and feasibility of contour-based scene simplification at different phosphene resolutions.*
   Journal of Vision, 22(2):1. DOI: [10.1167/jov.22.2.1](https://doi.org/10.1167/jov.22.2.1)

10. **Han, N., et al. (2021). ✅已下载**
    *Deep Learning–Based Scene Simplification for Bionic Vision.*
    ACM CHI 2021. arXiv: [2102.00297](https://arxiv.org/abs/2102.00297)
    - 深度学习场景简化用于仿生视觉 (被引>70)

11. **Granley, J., Relic, L., & Beyeler, M. (2022). ✅已下载**
    *Hybrid neural autoencoders for stimulus encoding in visual and other sensory neuroprostheses.*
    NeurIPS 2022. arXiv: [2211.05302](https://arxiv.org/abs/2211.05302)

12. **Kasowski, J., & Beyeler, M. (2022). ✅已下载**
    *Immersive virtual reality simulations of bionic vision.*
    ACM Augmented Humans 2022. arXiv: [2203.05675](https://arxiv.org/abs/2203.05675)
    - VR-SPV 开源工具包

13. **Granley, J., Riedel, A., & Beyeler, M. (2022).**
    *Adapting brain-like neural networks for modeling cortical visual prostheses.*
    NeurIPS SVRHM Workshop 2022.
    - CNN解码用于产生符合真实患者报告的逼真光幻视

14. **van Steveninck, J., et al. (2024). ✅已下载**
    *Towards biologically plausible phosphene simulation for the differentiable optimization of visual cortical prostheses.*
    eLife, 13, e85812. DOI: [10.7554/eLife.85812](https://doi.org/10.7554/eLife.85812)

15. **Xia, X., He, X., Feng, L., et al. (2022).**
    *Semantic translation of face image with limited pixels for simulated prosthetic vision.*
    Information Sciences, 609, 507–532.
    - 有限像素下面部图像语义翻译

16. **de Ruyter van Steveninck, J., et al. (2022). ✅已下载**
    *End-to-end optimization of image processing for simulated prosthetic vision.*
    arXiv: [2205.08412](https://arxiv.org/abs/2205.08412)

17. **Wu, Y., et al. (2023). ✅已下载**
    *A Deep Learning-based in silico Framework for Optimization of Phosphene Vision.*
    arXiv: [2302.03570](https://arxiv.org/abs/2302.03570)

18. **Schoinas, E., et al. (2025). ✅已下载**
    *Evaluating Deep Human-in-the-Loop Optimization for Retinal Prostheses.*
    arXiv: [2502.00177](https://arxiv.org/abs/2502.00177)

### 🔵 硬件 & 分辨率优化

19. **Wang, B.-Y., Chen, Z. C., Bhuckory, M., et al. (2022).**
    *Electronic photoreceptors enable prosthetic visual acuity matching the natural resolution in rats.*
    Nature Communications, 13(1), 6627. DOI: [10.1038/s41467-022-34387-y](https://doi.org/10.1038/s41467-022-34387-y)
    - **被引 >80**: 高分辨率光伏阵列实现自然分辨率匹配

20. **Bhuckory, M. B., Wang, B.-Y., et al. (2023).**
    *Cellular migration into a subretinal honeycomb-shaped prosthesis for high-resolution prosthetic vision.*
    PNAS, 120(42), e2307380120.

21. **Bruce, A., & Beyeler, M. (2022). ✅已下载**
    *Greedy optimization of electrode arrangement for epiretinal prostheses.*
    MICCAI 2022.

22. **Chen, Z. C., Wang, B.-Y., et al. (2022).**
    *Photovoltaic implant simulator reveals resolution limits in subretinal prosthesis.*
    Journal of Neural Engineering, 19(5), 055008.

23. **Palanker, D., et al. (2025). ✅已下载**
    *Restoration of Sight with Electronic Retinal Prostheses.*
    Annual Review of Vision Science.

### ⚪ 博士论文

24. **Sánchez García, M. (2022). ✅已下载**
    *Egocentric Computer Vision and Machine Learning for Simulated Prosthetic Vision.*
    PhD Thesis, Universidad de Zaragoza.
    - 系统性地研究SPV中的物体检测、分割和场景理解

---

## 🔗 关键研究机构与资源

- **Bionic Vision Lab (UCSB)** — Michael Beyeler 组: [bionicvisionlab.org](https://bionicvisionlab.org/publications)
- **Palanker Lab (Stanford)** — Daniel Palanker 组: [web.stanford.edu/~palanker](https://web.stanford.edu/~palanker/publications/)
- **Pulse2Percept** — 开源视网膜假体仿真框架: [pulse2percept.org](https://pulse2percept.org)
- **VR-SPV** — 开源VR SPV工具箱: [GitHub](https://github.com/bionicvisionlab/VR-SPV)

---

## 📝 获取付费论文的方法

以下关键论文在付费墙后，建议通过以下方式获取:

1. **Chai 2007** + **Zhao 2011 ×2** + **Lu 2013**
   - 通过机构订阅 Wiley / ARVO Journals
   - 或发邮件给通讯作者 (Xinyu Chai @ 上海交通大学) 请求预印本

2. **Nature Comms / PNAS** 论文
   - Nature Comms 为 OA 期刊，可通过 [nature.com](https://www.nature.com/articles/s41467-022-34387-y) 免费下载
   - PNAS 论文可通过 [PMC](https://www.ncbi.nlm.nih.gov/pmc/) 获取 (6个月后开放)

3. **Science (Chen et al. 2020)**
   - [science.org/doi/10.1126/science.abd7435](https://www.science.org/doi/10.1126/science.abd7435)

---

*文献检索日期: 2025年7月 | 覆盖时间: 2006–2025*
