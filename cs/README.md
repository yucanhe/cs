# 混沌压缩感知多媒体加密系统
# Chaos-based Compressed Sensing Multimedia Encryption

## 项目概述

本项目实现了一套基于**混沌系统**与**压缩感知**的多媒体加密方案，支持图像、音频和视频的安全加密与解密。该系统利用 n 维正弦-多项式复合模映射 (nD-SPCMM) 产生高质量混沌序列，结合压缩感知 (Compressed Sensing) 技术实现加密与压缩的一体化。

---

## 核心技术

### 1. 混沌系统 (nD-SPCMM)

**n维正弦-多项式复合模映射** (n-Dimensional Sine-Polynomial Compound Modular Map) 是本系统的核心混沌源：

```
x_{k+1}[i] = (a[i] * sin(b[i] * x_k[i]) + coupling + c) mod θ
```

**特性：**
- 支持任意维度扩展
- 级联结构保证雅可比矩阵为三角阵
- Lyapunov 指数解析可控
- 良好的均匀性和密钥敏感性

### 2. 压缩感知 (Compressed Sensing)

利用混沌序列构造测量矩阵，对多媒体数据进行压缩测量：

- **图像**：2D DCT → 分块 → CS 测量
- **音频**：STFT → DCT → CS 测量  
- **视频**：3D DCT → CS 测量

**重构算法**：FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)

### 3. 密码结构

- **密钥派生**：SHA-512(master_key || SHA512(plaintext))
- **扩散机制**：ARX-CBC + 动态 S-Box
- **可逆性**：密文以 PNG 图像格式存储，支持无损解密

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    输入 (明文)                          │
│              图像 / 音频 / 视频                         │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Stage 1: 密钥派生                          │
│         SHA-512 → nD-SPCMM 参数 seed                   │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Stage 2: 混沌序列生成                      │
│           nD-SPCMM → 统一混沌序列                       │
└─────────────────────┬───────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │  置乱   │ │ 测量    │ │ 扩散   │
    │Permutation│ │Measurement│ │Diffusion│
    └─────────┘ └─────────┘ └─────────┘
          │           │           │
          └───────────┼───────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    输出 (密文)                          │
│            cipher_rgba.png / .npz                      │
└─────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
cs/
├── chaos/                          # 混沌系统核心
│   ├── nD-SPCMM.py               # n维SPCMM定义与分析
│   └── haos_algebra_boolean_sbox_embed.py  # S-Box分析
│
├── image/                         # 图像加密模块
│   ├── cs_meas_crypto_full.py   # 核心加密算法
│   ├── exp_common.py            # 通用工具函数
│   ├── demo/                    # 演示脚本
│   └── experiments/             # 实验脚本
│
├── audio/                        # 音频加密模块
│   ├── audio_cs_spcmm_crypto.py
│   └── experiments_audio.py
│
├── video/                        # 视频加密模块
│   ├── video_3dcs_spcmm.py
│   └── video_experiments.py
│
├── resources/                    # 资源文件
│   ├── images/                   # 示例图像
│   ├── audio/                    # 示例音频
│   ├── video/                    # 示例视频
│   └── analysis/                 # 动力学分析图
│
├── results/                      # 实验结果
│   ├── image_exp/                # 图像实验
│   ├── audio_exp/                # 音频实验
│   └── video_exp/                # 视频实验
│
└── tests/                        # 测试数据
```

---

## 使用方法

### 图像加密/解密

```bash
# 加密
python image/demo/demo3.py \
    --img resources/images/lena.png \
    --cr 0.5 \
    --lam 0.01 \
    --iter 120 \
    --out results/image_exp/exp_base

# 参数说明:
# --cr     压缩率 m/n (0~1)
# --lam    FISTA 正则化参数
# --iter   FISTA 最大迭代次数
# --key    加密密钥
# --enc_ratio  选择加密比例
```

### 完整安全分析

```bash
# 运行完整的安全性分析实验
python image/experiments/image_security_analysis.py \
    --img resources/images/lena.png \
    --out results/image_exp/security_analysis
```

---

## 实验项目说明

### 🖼️ 1. 图像加密实验 (2D Signal)

**安全性分析：**
- 直方图分析 (Histogram Analysis)：展示加密前后像素分布，密文直方图应非常平坦（均匀分布）
- 相邻像素相关性 (Correlation)：水平、垂直、对角线方向的相关系数应接近 0
- 信息熵 (Information Entropy)：8位灰度图的理想信息熵为 8
- 密钥空间分析 (Key Space)：总有效密钥长度需 > 2^128 以抵御穷举攻击
- 密钥敏感性 (Key Sensitivity)：改变密钥中的一个比特位，解密结果应彻底失效
- NPCR/UACI 差分攻击：计算像素改变率和统一平均改变强度

**抗攻击分析：**
- 鲁棒性测试 (Robustness)：噪声攻击、剪切攻击

### 🔊 2. 音频加密实验 (1D Signal)

**安全性分析：**
- 时域波形图 (Waveform)：密文应表现为类白噪声
- 频谱分析 (Spectral)：密文频谱应杂乱无章
- 语谱图分析 (Spectrogram)：看不出原始语音特征

**质量与抗攻击：**
- SNR/PSNR：明文与还原音频的差异
- 相关系数：相邻采样点的相关性

### 🎬 3. 视频加密实验 (3D Signal)

**安全性与质量：**
- PSNR 与 SSIM：原始视频和解密视频的质量对比
- 关键帧独立性：分析关键帧被破坏后的不可读程度
- 视觉退化评估：展示加密视频帧，确认完全无法辨认

**抗攻击与压缩：**
- 压缩感知重构曲线：PSNR vs. Sampling Rate (m_rate)

### 🚀 4. 跨维度共同实验

- 加密/解密耗时：测试不同尺寸下的运行速度
- 吞吐量：计算单位时间内处理的数据量 (MB/s)

### 音频加密/解密

```bash
# 加密
python audio/audio_cs_spcmm_crypto.py encrypt \
    --in_wav resources/audio/1.wav \
    --out_npz encrypted.npz \
    --key "my-secret-key"

# 解密
python audio/audio_cs_spcmm_crypto.py decrypt \
    --in_npz encrypted.npz \
    --out_wav decrypted.wav \
    --key "my-secret-key"
```

### 视频加密/解密

```bash
# 加密
python video/video_3dcs_spcmm.py encrypt \
    --in_file resources/video/small.mp4 \
    --out_file video.npz \
    --key "my-secret-key"

# 解密
python video/video_3dcs_spcmm.py decrypt \
    --in_file video.npz \
    --out_file decrypted.mp4 \
    --key "my-secret-key"
```

---

## 实验结果

### 图像加密质量

| 指标 | 典型值 | 说明 |
|------|--------|------|
| PSNR | 30+ dB | 峰值信噪比 |
| SSIM | 0.87+ | 结构相似性 |
| NPCR | 99.6%+ | 像素变化率 |
| UACI | 33.4%+ | 平均变化强度 |
| 熵 | ~8.0 | 接近随机 |

### 密钥敏感性

- 密钥翻转 1 bit → NPCR > 99.5%
- 错误密钥解密 → PSNR < 7 dB (近似噪声)

### 不同图像尺寸

| 尺寸 | PSNR | SSIM |
|------|------|------|
| 256×256 | 27.0 dB | 0.86 |
| 512×512 | 30.4 dB | 0.87 |
| 1024×1024 | 35.4 dB | 0.94 |

---

## 安全特性

1. **密钥敏感性**：微小密钥变化导致密文巨大差异
2. **密文统计特性**：均匀分布，高熵，低相关性
3. **差分攻击抵抗**：NPCR/UACI 接近理论最大值
4. **可选择加密**：支持部分加密，平衡安全性与效率

---

## 完整学术实验框架

本项目包含完整的学术论文级别实验框架，覆盖以下方面：

### 🛡️ 一、统计安全性实验 (Statistical Security)

```bash
python image/experiments/statistical_security_analysis.py \
    --img resources/images/lena.png \
    --cipher results/image_exp/exp_base/cipher_rgba.png \
    --out results/statistical_security
```

**包含：**
- 直方图分析 + Chi-Square检验
- 相邻像素相关性 + 统计显著性检验
- 信息熵 + NIST标准
- KS检验 (Kolmogorov-Smirnov)
- 游程检验 (Runs Test)
- 频谱分析 (功率谱密度)
- 2D傅里叶频谱可视化

### ⚡ 二、密钥与鲁棒性实验 (Key & Robustness)

```bash
python image/experiments/key_robustness_experiments.py \
    --img resources/images/lena.png \
    --cipher results/image_exp/exp_base/cipher_rgba.png \
    --video resources/video/small.mp4 \
    --out results/key_robustness
```

**包含：**
- 密钥空间分析 (Key Space)
- 密钥敏感性曲线 (改变10^-16量级)
- NPCR/UACI差分攻击分析
- 噪声攻击 (Salt & Pepper, Gaussian)
- 抗剪切攻击 (Cropping)
- 视频丢包攻击 (Packet Loss)

### 📉 三、信号特有实验 (Domain Specific)

#### 视频实验
- PSNR/SSIM随采样率变化曲线
- 码率节省分析
- 时间一致性测试 (Temporal Stability)
- 关键帧独立性分析

#### 音频实验
- PESQ / STOI 测试
- LPC距离分析
- 语谱图分析

### ⚙️ 四、性能与效率实验 (Efficiency)

```bash
python image/experiments/performance_analysis.py \
    --img resources/images/lena.png \
    --out results/performance
```

**包含：**
- 计算复杂度分析 (Big O Notation)
- 运行耗时对比
- CPU/内存监控
- 吞吐量分析
- CS重构曲线 (PSNR vs 采样率)

### 🧪 五、高级实验 (Advanced)

```bash
python image/experiments/advanced_experiments.py \
    --plain resources/images/lena.png \
    --cipher results/image_exp/exp_base/cipher_rgba.png \
    --out results/advanced
```

**包含：**
- 分叉图 (Bifurcation Diagram)
- 李雅普诺夫指数 (Lyapunov Exponent)
- 相图分析 (Phase Portrait)
- 边缘保留度 (Edge Preservation)
- 视觉信息丢失 (VIF)
- 密文质量分析

---

## 依赖环境

- Python 3.8+
- NumPy
- SciPy
- Pillow (PIL)
- matplotlib
- scikit-image (可选)
- numba (可选，加速)

---

## 参考文献

1. nD-SPCMM: n-Dimensional Sine-Polynomial Compound Modular Map
2. CS: Compressed Sensing
3. FISTA: Fast Iterative Shrinkage-Thresholding Algorithm
4. ARX-CBC: Add-Rotate-XOR Cipher Block Chaining

---

## 许可证

MIT License

---

*本项目为学术研究成果，仅供研究学习使用。*
