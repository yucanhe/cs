# ============================================================
# Chaotic Compressed Sensing Multimedia Encryption (ChaoS-CS)
# 统一实验配置文件
# ============================================================

# ============================================================
# 1. 数据集配置
# ============================================================
DATASETS = {
    "image": {
        "standard": [
            "resources/images/lena.png",
            "resources/images/peppers.png",
            "resources/images/baboon.png",
            "resources/images/boat.png",
            "resources/images/barbara.png",
        ],
        "real_scenario": [
            "resources/images/camera.png",
            "resources/images/document.png",
        ],
        "sizes": ["256x256", "512x512", "1024x1024"],
    },
    "audio": {
        "speech": [
            "resources/audio/1.wav",  # TIMIT/LibriSpeech片段
        ],
        "music": [
            "resources/audio/music.wav",
        ],
        "noisy": [
            "resources/audio/noisy_speech.wav",
        ],
        "sample_rate": 44100,
        "duration_sec": 5,
    },
    "video": {
        "static": ["resources/video/static.mp4"],      # 静态场景
        "slow_motion": ["resources/video/slow.mp4"],  # 慢运动
        "fast_motion": ["resources/video/fast.mp4"],  # 快运动
        "face": ["resources/video/face.mp4"],          # 人脸
        "street": ["resources/video/street.mp4"],     # 街景
        "resolutions": {
            "360p": (640, 360),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
        },
        "fps": 30,
    },
}

# ============================================================
# 2. 加密参数表
# ============================================================
# 图像参数
IMAGE_PARAMS = {
    "B": 8,                    # 块大小
    "cr": 0.5,                 # 压缩率 (m/n)
    "m_meas": 32,              # 测量数 (B*B*cr)
    "fista_iter": 120,         # FISTA迭代次数
    "lam": 0.001,              # L1正则化参数
    "float_diffusion": True,   # 3D扩散开关
    "use_arx": True,           # ARX加密开关
    "use_sbox": True,          # S-Box加密开关
    "static_phi": False,       # 静态测量矩阵
}

# 音频参数
AUDIO_PARAMS = {
    "m_rate": 0.9,             # 压缩率
    "lam": 0.001,              # L1正则化
    "iters": 250,              # 迭代次数
    "nperseg": 1024,           # STFT窗口大小
    "noverlap": 512,           # STFT重叠
    "f_max": 512,              # 最大频率
    "phi_group": 8,            # 频率分组
    "use_arx": True,
    "use_sbox": True,
}

# 视频参数
VIDEO_PARAMS = {
    "block": 8,                # 空间块大小
    "cube_t": 8,              # 时间块大小 (帧数)
    "chunk_frames": 64,       # 每chunk帧数
    "key_stride": 2,          # 关键帧间隔
    "m_rate_key": 0.7,        # 关键帧压缩率
    "lf_xy": 10,              # 空间低频系数数
    "lf_t": 6,                # 时间低频系数数
    "q_step": 0.015,          # 量化步长
    "lam": 0.0015,            # L1正则化
    "iters": 140,             # FISTA迭代
    "hot_start": 1,           # 热启动
    "use_arx": 1,
    "use_sbox": 1,
    "keep_nonkey_y": 1.0,     # 非关键帧保留率
}

# ============================================================
# 3. 对比方法
# ============================================================
COMPARISON_METHODS = {
    "image": [
        "ChaoS-CS ( Ours)",   # 本方案
        "CS-Only",             # 仅CS无加密
        "ARX-Only",            # 仅ARX无CS
        "AES-CTR",             # AES-CTR模式
        "Chen-Chaos",          # 陈氏混沌加密
    ],
    "audio": [
        "ChaoS-CS (Ours)",
        "CS-Only",
        "STFT-AES",
        "DCT-S盒子",
    ],
    "video": [
        "ChaoS-CS (Ours)",
        "Selective-Encryption",
        "Full-Encryption",
        "HEVC-SEC",
    ],
}

# ============================================================
# 4. 硬件/软件环境
# ============================================================
ENVIRONMENT = {
    "cpu": "Intel Core i9 / AMD Ryzen 9",
    "gpu": "NVIDIA RTX 3080+ (可选)",
    "memory": "16GB+ RAM",
    "python": "3.8+",
    "numpy": "1.21+",
    "scipy": "1.7+",
    "scikit-image": "0.19+",
    "numba": "0.55+",
    "pywt": "0.5+",
    "opencv-python": "4.5+",
}

# ============================================================
# 5. 评价指标定义
# ============================================================
METRICS = {
    # 图像指标
    "image": {
        "psnr": {
            "formula": "PSNR = 10 * log10(255^2 / MSE)",
            "description": "Peak Signal-to-Noise Ratio (dB)",
            "threshold": ">30dB 为高质量",
        },
        "ssim": {
            "formula": "SSIM = (2*μx*μy+C1)*(2*σxy+C2) / ((μx^2+μy^2+C1)*(σx^2+σy^2+C2))",
            "description": "Structural Similarity Index",
            "threshold": ">0.9 为高质量",
        },
        "npcr": {
            "formula": "NPCR = Σ|D(i,j)| / (H*W) * 100%",
            "description": "Number of Pixels Change Rate",
            "threshold": ">99% 表示对明文变化敏感",
        },
        "uaci": {
            "formula": "UACI = Σ|I1(i,j)-I2(i,j)| / (255*H*W) * 100%",
            "description": "Unified Average Changing Intensity",
            "threshold": ">30% 表示良好的扩散",
        },
        "entropy": {
            "formula": "H = -Σp(i)log2(p(i))",
            "description": "Information Entropy (bits)",
            "threshold": "接近8为高随机性",
        },
        "correlation": {
            "formula": "r = Σ(xi-μx)(yi-μy) / √Σ(xi-μx)^2 * √Σ(yi-μy)^2",
            "description": "相邻像素相关性",
            "threshold": "接近0为高随机性",
        },
        "chi_square": {
            "formula": "χ² = Σ(oi-ei)²/ei",
            "description": "卡方检验",
            "threshold": "p-value > 0.05 为通过",
        },
    },
    # 音频指标
    "audio": {
        "snr": {
            "formula": "SNR = 10*log10(Σs²/Σn²)",
            "description": "Signal-to-Noise Ratio (dB)",
        },
        "psnr": {
            "formula": "PSNR = 10*log10(MAX²/MSE)",
            "description": "Perceptual Signal-to-Noise Ratio",
        },
        "stoi": {
            "formula": "短时客观可懂度",
            "description": "Short-Time Objective Intelligibility (0-1)",
            "threshold": ">0.8 为高可懂度",
        },
        "pesq": {
            "formula": " perceptual evaluation of speech quality",
            "description": "PESQ (-0.5 to 4.5)",
            "threshold": ">3.0 为高质量",
        },
    },
    # 视频指标
    "video": {
        "psnr_frame": "同图像PSNR，按帧平均",
        "ssim_frame": "同图像SSIM，按帧平均",
        "vmaf": "Video Multimethod Assessment Fusion (可选)",
        "temporal_stability": "时间一致性，相邻帧质量方差",
    },
}

# ============================================================
# 6. 密钥配置
# ============================================================
KEY_CONFIG = {
    "master_key": "my-secret-key-2026",
    "key_length_bits": 256,  # SHA-256
    "chaos_params": {
        "a": 3.99,
        "k": 0.13,
        "precision": 1e-16,  # 双精度
    },
    "key_space": ">2^256",  # 理论密钥空间
}

# ============================================================
# 7. 实验参数扫描范围
# ============================================================
PARAM_SWEEPS = {
    "image": {
        "cr": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "iters": [50, 80, 100, 120, 150, 200],
        "lam": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    },
    "audio": {
        "m_rate": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        "iters": [100, 150, 200, 250, 300],
        "nperseg": [512, 1024, 2048],
    },
    "video": {
        "m_rate_key": [0.3, 0.5, 0.7, 0.9],
        "key_stride": [1, 2, 3, 4, 5],
        "keep_nonkey": [0.3, 0.5, 0.7, 1.0],
        "q_step": [0.005, 0.01, 0.015, 0.02, 0.03],
    },
}

# ============================================================
# 8. 鲁棒性测试参数
# ============================================================
ROBUSTNESS_TESTS = {
    "noise": {
        "gaussian": [0.01, 0.05, 0.1, 0.2],      # 方差
        "salt_pepper": [0.01, 0.05, 0.1, 0.2],   # 噪声比例
        "speckle": [0.01, 0.05, 0.1],            # 噪声强度
    },
    "cropping": {
        "image": [0.1, 0.25, 0.5],               # 裁剪比例
    },
    "packet_loss": {
        "video": [0.05, 0.1, 0.2, 0.3],          # 丢包率
        "audio": [0.05, 0.1, 0.2],               # 丢包率
    },
    "dropout": {
        "block": [0.1, 0.2, 0.3, 0.5],            # 块丢失率
    },
}

# ============================================================
# 9. 输出配置
# ============================================================
OUTPUT_CONFIG = {
    "results_dir": "results/experiments",
    "save_cipher": True,
    "save_decrypted": True,
    "save_plots": True,
    "save_report": True,
    "plot_format": "png",
    "dpi": 300,
}
