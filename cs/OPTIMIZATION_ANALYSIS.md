# 音频和视频加密优化分析

## 🎵 音频优化参数

### 可调参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| **m_rate** | 0.9 | 0.5-1.0 | CS压缩率，越高质量越好 |
| **iters** | 250 | 50-500 | FISTA迭代次数 |
| **nperseg** | 1024 | 256-4096 | STFT窗口大小 |
| **noverlap** | 512 | nperseg/2 | STFT重叠 |
| **f_max** | 512 | 256-2048 | 最大频率 |
| **lam** | 0.001 | 0.0001-0.01 | 正则化参数 |

### 预期效果

```
m_rate 影响:
m_rate=0.7: 低质量, 高压缩
m_rate=0.9: 平衡 (当前)
m_rate=0.99: 高质量, 低压缩

iters 影响:
iters=100: 速度快, 质量低
iters=250: 平衡 (当前)
iters=500: 速度慢, 质量高

nperseg 影响:
nperseg=512:  高时间分辨率
nperseg=1024: 平衡 (当前)
nperseg=2048: 高频率分辨率
```

---

## 🎬 视频优化参数

### 可调参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| **m_rate_key** | 0.7 | 0.5-1.0 | 关键帧压缩率 |
| **key_stride** | 2 | 1-5 | 关键帧间隔 |
| **chunk_frames** | 30 | 10-60 | 每批处理帧数 |
| **cube_t** | 8 | 4-16 | 时间块大小 |
| **block** | 16 | 8-32 | 空间块大小 |
| **iters** | 100 | 50-200 | FISTA迭代 |
| **q_step** | 0.5 | 0.1-1.0 | 量化步长 |

### 预期效果

```
m_rate_key 影响:
m_rate_key=0.5: 高压缩, 质量略低
m_rate_key=0.7: 平衡 (当前)
m_rate_key=0.9: 低压缩, 质量高

key_stride 影响:
key_stride=1:  每帧是关键帧, 质量最高, 文件最大
key_stride=2:  每2帧关键帧 (当前)
key_stride=5:  每5帧关键帧, 质量较低, 文件最小
```

---

## 🚀 推荐优化配置

### 音频

```bash
# 高质量配置
python audio/audio_cs_spcmm_crypto.py encrypt \
    --in_wav resources/audio/1.wav \
    --out_npz results/audio_high.npz \
    --key "my-secret-key-2026" \
    --m_rate 0.95 \
    --iter 300 \
    --nperseg 2048

# 高速配置
python audio/audio_cs_spcmm_crypto.py encrypt \
    --in_wav resources/audio/1.wav \
    --out_npz results/audio_fast.npz \
    --key "my-secret-key-2026" \
    --m_rate 0.8 \
    --iter 150 \
    --nperseg 512
```

### 视频

```bash
# 高质量配置
python video/video_3dcs_spcmm.py encrypt \
    --in_file resources/video/small.mp4 \
    --out_file results/video_high.npz \
    --key "my-secret-key-2026" \
    --m_rate_key 0.9 \
    --key_stride 1 \
    --iter 150

# 高压缩配置
python video/video_3dcs_spcmm.py encrypt \
    --in_file resources/video/small.mp4 \
    --out_file results/video_compact.npz \
    --key "my-secret-key-2026" \
    --m_rate_key 0.5 \
    --key_stride 3 \
    --iter 50
```

---

## 📊 优化对比表

### 音频

| 配置 | m_rate | iters | 预期 SNR | 预期质量 |
|------|--------|-------|----------|----------|
| 原配置 | 0.9 | 250 | ~4dB | 基准 |
| 高质量 | 0.95 | 300 | ~6dB | +50% |
| 高速 | 0.8 | 150 | ~3dB | -25% |

### 视频

| 配置 | m_rate_key | key_stride | 预期 PSNR | 预期质量 |
|------|------------|------------|-----------|----------|
| 原配置 | 0.7 | 2 | ~30dB | 基准 |
| 高质量 | 0.9 | 1 | ~35dB | +60% |
| 高压缩 | 0.5 | 3 | ~25dB | -20% |
