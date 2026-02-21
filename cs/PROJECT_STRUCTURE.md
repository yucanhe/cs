## 项目结构与功能分类（概览）

> 本文件只做“分类说明”，不删除、不移动任何文件。

### 1. 混沌系统 / 数学与密码结构（核心理论）

- `chaos/nD-SPCMM.py`  
  - 定义 n 维 SPCMM 混沌映射 `nD_spcmm`，以及 Lyapunov 指数分析等动力学测试，生成相图、分叉图、PDF、LE 曲线、参数空间 MLE 热力图等。
- `chaos/haos_algebra_boolean_sbox_embed.py`  
  - 与 Boolean 代数 / S‑Box 嵌入相关的实验代码，用于密码结构分析。

---

### 2. 图像方向

#### 2.1 图像加密核心管线（算法主实现）

- `image/cs_meas_crypto_full.py`  
  - 图像端到端加密/解密主 pipeline：  
    - 由主密钥 + 明文图像字节派生 seed（SHA‑512）。  
    - 基于 nD‑SPCMM 生成统一混沌序列。  
    - 图像灰度归一化 → 分块 → DCT/IDCT。  
    - 块内 / 块间置乱（混沌驱动）。  
    - 构造 CS 测量矩阵并对块系数测量。  
    - 测量域 3D 循环移位 + 浮点扩散。  
    - 解密端使用 FISTA 重构、逆置乱、逆 DCT 得到重建图像，并计算 PSNR/SSIM。

#### 2.2 图像实验通用工具

- `image/exp_common.py`  
  - `ensure_dir`：确保目录存在。  
  - `parse_report`：从 `report.txt` 中解析 MAE、PSNR、SSIM、NPCR、UACI 等指标。  
  - `write_csv`：写入实验结果表。  
  - `plot_xy`：画简单折线图（如 PSNR vs cr）。

#### 2.3 图像 Demo / 单次实验脚本

- `image/demo/demo1.py`  
- `image/demo/demo2.py`  
- `image/demo/demo3.py`  
- `image/demo/demo_expiation.py`  
  - 一次性读取图像，调用图像加密管线，输出 demo 图像和 `report.txt` 等，用于展示单个案例或生成论文插图。

#### 2.4 图像批量实验 / 论文实验脚本

- `image/experiments/exp_1_tradeoff.py`  
  - 扫描不同压缩率 `cr`，多次调用 demo（默认 `demo3.py`），统计 PSNR/SSIM/NPCR/UACI 与 `cr` 的关系，并画对比图。
- `image/experiments/exp_2_dataset.py`  
  - 在图像数据集上批量跑实验，统计整体表现。
- `image/experiments/exp_3_repeat.py`  
  - 多次重复实验，考察稳定性和波动范围。
- `image/experiments/exp_4_keys.py`  
  - 多密钥对比实验，分析密钥敏感性与安全性。
- `image/experiments/exp_5_attacks.py`  
  - 攻击相关实验（如差分攻击、已知/选择明文等场景）。
- `image/experiments/exp_6_ablation.py`  
  - 消融实验：关闭/替换部分模块，验证各组件贡献。
- `image/experiments/exp_7_stats_plus.py`  
  - 扩展统计分析（信息熵、相关性、卡方等更细指标）。  
- `image/experiments/sweep_params.py`  
  - 参数扫掠工具，例如对迭代次数、阈值、压缩率等做 grid search。

---

### 3. 音频方向

#### 3.1 音频加密核心脚本

- `audio/audio_cs_spcmm_crypto.py`  
  - 使用 3D‑SPCMM 混沌序列驱动的音频加密方案：  
    - STFT（Hann 窗）将音频变换到时频域。  
    - 频带分组后，对每一频带的时间维做 DCT + CS 测量。  
    - 测量值经量化为 `uint32`，再经过 ARX‑CBC 扩散 + 动态 S‑Box 字节替换。  
    - 解密端：逆 S‑Box + 逆 ARX‑CBC + FISTA 重构 + iDCT + ISTFT 还原波形，并输出 SNR/PSNR/MSE 等质量指标。  
  - 提供命令行接口：  
    - `encrypt`：`--in_wav`, `--out_npz`, `--key`, `--m_rate`, `--lam`, `--iter`, `--nperseg`, `--noverlap`, `--f_max`, `--phi_group`。  
    - `decrypt`：`--in_npz`, `--out_wav`, `--key`, `--lam`, `--iter`, `--ref_wav` 等。

#### 3.2 音频实验总控脚本

- `audio/experiments_audio.py`  
  - 调用上面的 `audio_cs_spcmm_crypto.py`，整合 5 类实验：  
    1. 基本质量：加解密一次，计算 SNR/PSNR/MSE。  
    2. 差分攻击：对原音频做微小扰动（如 1 LSB），比较两次密文 RAW 字节的 NPCR/UACI。  
    3. 密钥敏感性：错误密钥解密的质量（SNR）。  
    4. 鲁棒性：对密文字节添加噪声或 dropout，并再解密观察退化。  
    5. 统计/感知：密文熵、相邻相关性、卡方检验，以及可选 STOI/PESQ。  
  - 输出 `summary.json` / `summary.txt` 和一系列中间结果/图像。

---

### 4. 视频方向

#### 4.1 视频加密核心脚本

- `video/video_3dcs_spcmm.py`  
  - 针对视频的 3D‑CS 混沌加密方案：  
    - 读取输入视频，按 `cube_t` 帧组成 3D cube。  
    - 转为 YCbCr，并分别处理 Y/Cb/Cr 通道。  
    - 对每 cube 的 Y 通道块做 3D‑DCT（空间 + 时间），选低频系数，用混沌高斯矩阵做 CS 测量。  
    - 关键 cube：直接对低频系数 CS + 量化 + ARX‑CBC + S‑Box。  
    - 非关键 cube：相对关键 cube 做残差编码（保持时序相关性）。  
    - 解密端配套使用 `PhiCache` + FISTA 快速重构，并恢复为 YCbCr → RGB 视频。  
  - 提供命令行接口：  
    - `encrypt`：`--in_file`, `--out_file`(manifest npz), `--key`, `--cube_t`, `--block`, `--key_stride`, `--m_rate_key`, `--keep_nonkey_y`, `--keep_chroma`, `--lf_xy`, `--lf_t`, `--q_step`, `--lam`, `--iters`, `--hot_start`, `--no_arx`, `--no_sbox`, `--cache_mb`。  
    - `decrypt`：`--in_file`(manifest), `--out_file`, `--key`, `--lam`, `--iters`, `--hot_start`, `--cache_mb`。

#### 4.2 视频实验总控脚本

- `video/video_experiments_video.py`  
  - 以 `video_3dcs_spcmm.py` 为“密码引擎”，组织一整套视频实验：  
    1. 基础加解密：记录 encrypt/decrypt 用时，输出解密视频。  
    2. 重建质量：对比原视频与解密视频，逐帧统计 PSNR/SSIM/MSE 并画曲线。  
    3. 统计安全性（解密视频视角）：直方图、信息熵、相邻相关性、卡方。  
    4. 时域结构：相邻帧差分的平均图（plain/dec 对比）。  
    5. NPCR/UACI（密文字节视角）：对原始视频做像素微扰，分别加密，对 manifest+chunks 里密文字节做 NPCR/UACI。  
    6. 密钥敏感性：错误密钥解密，并对比质量（PSNR/SSIM）。  
    7. 鲁棒性（丢包）：随机删减 manifest 中部分 cube 文件，再解密观察退化。  
    8. 参数扫描：  
       - `m_rate_key` 扫描：速率‑失真曲线（PSNR/SSIM vs m_rate_key）。  
       - FISTA `iters` 扫描：解密时间 vs 质量曲线。  

---

### 5. 批量运行与辅助脚本

- `run_all.sh`  
  - Shell 层面统一调用多个 `exp_*.py` / demo / `experiments_*.py` 的脚本，方便一键跑全部实验。

---

### 6. 结果 / 中间产物目录（实验输出，不是核心算法）

> 这些目录主要由上面的实验脚本自动生成，包含中间 npz、统计图、CSV 结果、报告文本等。

- 图像相关：
  - `exp_base/`, `exp_out/`, `exp_test/`  
  - `exp11_ablation3/`, `exp2_iter/`, `exp2_lam/`, `exp4_multi/`, `exp5_size/`, `exp6_keytest/`, `exp7_selenc/`, `exp8_robust/`, `exp9_kpa/`  
  - `outputs_demo/`, `outputs_demo_int_cipher_measure/`, `outputs_demo_int_measure/`, `outputs_demo3_full/`  
  - `outputs_final_int_measure/`, `outputs_fista_full/`, `outputs_opt_v2/`, `outputs_opt_v2_fixed/`  
  - `outputs_paper_all/`, `outputs_paper_all_fixed/`, `outputs_sbox_rgba/`  
  - `rd_cr0.125/`, `rd_cr0.25/`, `rd_cr0.375/`, `rd_cr0.5/`, `rd_cr0.625/`, `rd_cr0.75/`, `rd_cr0.875/`, `rd_cr1.0/`  
  - `outputs_cr_adjustable/`, `sweep_runs/`

- 视频相关：
  - `cipher_video_chunks/`：视频密文 cube 文件。  
  - `exp_video_out/`：视频实验输出（manifest、解密视频、图表等）。

- 其他块级 / 测试输出：
  - `test_chunks/`

---

### 7. 资源与示例数据

- 示例图像：
  - `1.png`  
  - `images/` 目录：包含更多图像样本或结果图。

- 示例音频：
  - `1.wav`, `rec.wav`, `rec_fast3.wav`

- 示例视频：
  - `small.mp4`, `1.mp4`, `decrypted_quality.mp4` 等。

- 混沌动力学分析输出图：
  - `1_phase_portrait_3d.png`  
  - `2_pdf_analysis.png`  
  - `3_precision_degradation.png`  
  - `4_bifurcation_diagram.png`  
  - `5_lyapunov_analysis.png`  
  - `6_parameter_space_mle.png`  
  - 以及其他单独统计图，如 `correlation_analysis.png`。

---

### 8. 汇总表与统计 CSV

- `all_results.csv`：跨实验的结果汇总表。  
- `summary.csv`：综合统计结果。  
- `exp8_robust_results_summary.csv`：鲁棒性相关汇总。  
- 各 `exp_*` / `experiments_*` 生成的 `summary.txt` / `*.json` / `*.csv`：对应实验的本地结果记录。


