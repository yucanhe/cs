# 加密方案架构优化分析

## 🔍 当前方案架构

### 图像加密流程
```
原始图像 → DCT变换 → 分块 → CS测量 → 3D扩散 → ARX-CBC → S-Box → 密文
                                                        ↓
解密: 逆S-Box → 逆ARX → 逆扩散 → CS重建(IDCT) → 逆DCT → 原始图像
```

### 音频加密流程
```
原始音频 → STFT → DCT → CS测量 → ARX-CBC → S-Box → 密文
                                           ↓
解密: 逆ARX → CS重建 → 逆DCT → 逆STFT → 原始音频
```

### 视频加密流程
```
视频帧 → 3D分块 → 3D-DCT → 3D-CS测量 → 关键帧/非关键帧处理 → 密文
                                                    ↓
解密: 逆处理 → 3D-CS重建 → 逆3D-DCT → 原始帧
```

---

## 🎯 可优化的改进点

### 1. 图像加密方案优化

#### 1.1 当前架构问题

| 问题 | 现状 | 优化建议 |
|------|------|----------|
| **测量矩阵** | 随机高斯 | 改用确定性矩阵 (Bernoulli, Scrambled FFT) |
| **扩散机制** | 3D扩散 + ARX双层 | 简化为单一高效层 |
| **S-Box** | 每块动态生成 | 预计算静态S-Box |
| **重建算法** | FISTA | 改用 AMP/CoSaMP/IST |

#### 1.2 优化方案A: 高效测量矩阵
```python
# 当前: 随机高斯 (效率低)
Phi = np.random.randn(m, n)

# 优化1: Bernoulli (±1) 矩阵 (效率高4倍)
Phi = np.sign(np.random.randn(m, n))

# 优化2: 确定性Scrambled FFT (更安全)
def scrambled_fft_matrix(m, n):
    # 快速且满足RIP
    P = fft_matrix(n)[:, :m]
    S = np.diag(np.exp(1j * np.random.rand(n) * 2 * np.pi))
    return np.real(P @ S)[:m, :]
```

#### 1.3 优化方案B: 简化的扩散机制
```python
# 当前: 3D扩散 + ARX (复杂)
D = diffusion3d_forward(Yn, mask)  # O(HWM)
Cu32 = u32_diffuse_forward(D, key)  # O(N)

# 优化: 仅用ARX (更快)
Cu32 = u32_diffuse_forward(Yn, mask)  # 合并到一步
```

#### 1.4 优化方案C: 改用AMP算法
```python
# 当前: FISTA (逐块迭代)
def fista_l1(Phi, y, lam, max_iter):
    # 收敛慢
    
# 优化: AMP (近似消息传递) - 收敛更快
def amp_l1(Phi, y, lam, max_iter):
    x = Phi.T @ y
    z = y - Phi @ x
    for i in range(max_iter):
        x = x + Phi.T @ z / sigma_z
        x = soft_threshold(x, lam)
        z = y - Phi @ x + z * sigma_z / n
    return x
```

---

### 2. 音频加密方案优化

#### 2.1 当前架构问题

| 问题 | 现状 | 优化建议 |
|------|------|----------|
| **时频变换** | STFT+DCT | 改用UWP (小波包) |
| **分组处理** | 固定分组 | 自适应分组 |
| **CS重建** | 逐组独立 | 联合稀疏重建 |

#### 2.2 优化方案A: 小波包变换
```python
# 当前: STFT + DCT (双变换)
f, tt, Z = stft(x, nperseg=1024)
s = dct(mag[fi, :])  # 再次DCT

# 优化: 直接用UWP (单变换, 更适合音频)
import pywt
wpt = pywt.WaveletPacket2D(data, 'db8', mode='sym')
# UWP提供更好的时频局部化
```

#### 2.3 优化方案B: 自适应分组
```python
# 当前: 固定8个频率一组
phi_group = 8

# 优化: 根据能量自适应分组
def adaptive_grouping(mag, target_group_size=16):
    energy = np.sum(mag**2, axis=1)
    cumsum = np.cumsum(energy)
    # 按能量均分为若干组
```

---

### 3. 视频加密方案优化

#### 3.1 当前架构问题

| 问题 | 现状 | 优化建议 |
|------|------|----------|
| **分块方式** | 固定立方体 | 自适应ROI |
| **关键帧** | 固定间隔 | 内容自适应 |
| **运动补偿** | 未利用 | 利用时间相关性 |

#### 3.2 优化方案A: 内容自适应关键帧
```python
# 当前: 固定key_stride=2
key_stride = 2

# 优化: 基于场景变化检测
def adaptive_keyframe_detection(frames, threshold=0.3):
    keyframes = [0]
    for i in range(1, len(frames)):
        diff = np.mean(np.abs(frames[i] - frames[i-1]))
        if diff > threshold * 255:
            keyframes.append(i)
    return keyframes
```

#### 3.3 优化方案B: 3D注意力机制
```python
# 当前: 独立处理每个cube
for cube in cubes:
    Y = Phi_3d @ cube  # 独立

# 优化: 考虑相邻cube相关性
def attention_3d_measurement(cubes, Phi):
    # 对相邻cube使用加权测量
    result = []
    for i, cube in enumerate(cubes):
        w = 0.8 * cube + 0.1 * (cubes[i-1] if i>0 else 0) + 0.1 * (cubes[i+1] if i<len-1 else 0)
        result.append(Phi @ w)
    return np.array(result)
```

---

## 📊 优化效果预估

### 图像优化

| 优化项 | 原时间 | 优化后 | 加速比 | 质量影响 |
|--------|--------|--------|--------|----------|
| Bernoulli矩阵 | 2.2s | 1.5s | **1.5x** | 略降 |
| 简化扩散 | 2.2s | 1.8s | **1.2x** | 无 |
| AMP重建 | 12s | 6s | **2x** | 略降 |
| **综合** | 12s | 4s | **3x** | 待测 |

### 音频优化

| 优化项 | 原时间 | 优化后 | 加速比 |
|--------|--------|--------|--------|
| UWP变换 | 14s | 10s | **1.4x** |
| AMP重建 | 50s | 25s | **2x** |
| **综合** | 64s | 20s | **3x** |

### 视频优化

| 优化项 | 原时间 | 优化后 | 加速比 |
|--------|--------|--------|--------|
| 自适应关键帧 | 60s | 45s | **1.3x** |
| 运动补偿 | 60s | 40s | **1.5x** |
| **综合** | 60s | 30s | **2x** |

---

## 🚀 推荐实施优先级

### 第一阶段: 快速优化 (1-2天)

1. **图像**: 移除Float Diffusion (已完成 ✅)
2. **图像**: 使用Bernoulli测量矩阵
3. **音频**: 改用静态S-Box

### 第二阶段: 核心优化 (3-5天)

1. **图像**: 改用AMP算法
2. **音频**: 改用小波包变换
3. **视频**: 实现自适应关键帧

### 第三阶段: 高级优化 (1周+)

1. 多重混沌级联
2. 硬件加速 (GPU/CUDA)
3. 端到端神经网络优化

---

## ⚠️ 优化风险评估

| 优化项 | 安全性 | 实现难度 | 风险 |
|--------|--------|----------|------|
| Bernoulli矩阵 | 需验证 | 低 | ⚠️ 需测试 |
| 简化扩散 | 保持 | 低 | ✅ 安全 |
| AMP算法 | 略降 | 中 | ⚠️ 质量权衡 |
| UWP变换 | 保持 | 高 | 🔄 需实现 |
