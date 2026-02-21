#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频加密方案优化模块
1. 自适应关键帧检测
2. 运动补偿
3. Bernoulli测量矩阵
"""

import numpy as np
import hashlib


def adaptive_keyframe_detection(frames, threshold=0.3, min_key_distance=5):
    """
    自适应关键帧检测
    
    基于帧间差异自动确定关键帧位置，比固定key_stride更高效
    
    参数:
        frames: 视频帧列表 (N x H x W)
        threshold: 场景切换阈值 (0-1)
        min_key_distance: 最小关键帧间隔
    
    返回:
        keyframe_indices: 关键帧索引列表
    """
    n_frames = len(frames)
    keyframes = [0]  # 第一帧总是关键帧
    
    prev_frame = frames[0]
    for i in range(1, n_frames):
        curr_frame = frames[i]
        
        # 计算帧差异 (归一化到 0-1)
        diff = np.mean(np.abs(curr_frame.astype(float) - prev_frame.astype(float))) / 255.0
        
        # 距离上一个关键帧的距离
        dist_from_last_key = i - keyframes[-1]
        
        # 如果差异超过阈值且距离足够，则为关键帧
        if diff > threshold and dist_from_last_key >= min_key_distance:
            keyframes.append(i)
        
        prev_frame = curr_frame
    
    # 确保最后一个关键帧之后有足够的帧
    if n_frames - keyframes[-1] < min_key_distance // 2:
        keyframes = keyframes[:-1]
    
    return keyframes


def motion_compensated_prediction(curr_frame, prev_frame, block_size=16):
    """
    运动补偿预测
    
    使用块匹配算法估计帧间运动，并生成预测帧
    
    参数:
        curr_frame: 当前帧
        prev_frame: 上一帧
        block_size: 块大小
    
    返回:
        predicted_frame: 预测帧
        motion_vectors: 运动矢量场
    """
    H, W = curr_frame.shape
    n_blocks_h = H // block_size
    n_blocks_w = W // block_size
    
    predicted = np.zeros_like(curr_frame)
    motion_vectors = np.zeros((n_blocks_h, n_blocks_w, 2), dtype=np.int16)
    
    search_range = block_size // 2
    
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            # 当前块位置
            y_start = i * block_size
            x_start = j * block_size
            curr_block = curr_frame[y_start:y_start+block_size, x_start:x_start+block_size]
            
            # 搜索最匹配的块
            best_match = None
            best_sad = float('inf')
            
            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    py_start = y_start + dy
                    px_start = x_start + dx
                    
                    # 边界检查
                    if py_start < 0 or px_start < 0:
                        continue
                    if py_start + block_size > H or px_start + block_size > W:
                        continue
                    
                    prev_block = prev_frame[py_start:py_start+block_size, px_start:px_start+block_size]
                    
                    # 计算SAD (Sum of Absolute Differences)
                    sad = np.sum(np.abs(curr_block.astype(float) - prev_block.astype(float)))
                    
                    if sad < best_sad:
                        best_sad = sad
                        best_match = (dy, dx)
            
            if best_match:
                dy, dx = best_match
                motion_vectors[i, j] = [dy, dx]
                
                # 从参考帧复制预测块
                py_start = y_start + dy
                px_start = x_start + dx
                predicted[y_start:y_start+block_size, x_start:x_start+block_size] = \
                    prev_frame[py_start:py_start+block_size, px_start:px_start+block_size]
            else:
                predicted[y_start:y_start+block_size, x_start:x_start+block_size] = \
                    curr_frame[y_start:y_start+block_size, x_start:x_start+block_size]
    
    return predicted, motion_vectors


def residual_encoding(residual, m_rate=0.5):
    """
    残差编码
    
    对运动补偿后的残差进行压缩
    
    参数:
        residual: 残差帧
        m_rate: 压缩率
    
    返回:
        encoded: 编码后的数据
    """
    n = residual.size
    m = int(n * m_rate)
    
    # 简单随机采样
    np.random.seed(42)
    indices = np.random.choice(n, m, replace=False)
    indices = np.sort(indices)
    
    sampled = residual.flatten()[indices]
    
    return {
        'sampled': sampled,
        'indices': indices,
        'shape': residual.shape
    }


def residual_decoding(encoded):
    """
    残差解码
    """
    n = np.prod(encoded['shape'])
    reconstructed = np.zeros(n)
    reconstructed[encoded['indices']] = encoded['sampled']
    
    return reconstructed.reshape(encoded['shape'])


def bernoulli_measurement_matrix(m, n, seed=None):
    """
    Bernoulli测量矩阵 (±1随机)
    
    比高斯矩阵更快，且满足RIP性质
    
    参数:
        m: 测量数
        n: 信号维度
        seed: 随机种子
    
    返回:
        Phi: 测量矩阵 (m x n)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 生成 ±1 而非 0/1
    Phi = np.random.choice([-1, 1], size=(m, n)) / np.sqrt(m)
    
    return Phi


def create_deterministic_bernoulli(m, n, key, index=0):
    """
    基于密钥创建确定性Bernoulli矩阵
    
    参数:
        m, n: 矩阵维度
        key: 密钥字符串
        index: 矩阵索引
    
    返回:
        Phi: 测量矩阵
    """
    # 从密钥生成种子
    seed_bytes = hashlib.sha256(f"{key}_{index}".encode()).digest()
    seed = int.from_bytes(seed_bytes[:4], 'big') % (2**32)
    
    return bernoulli_measurement_matrix(m, n, seed=seed)


def scrambled_fft_measurement(m, n, key, index=0):
    """
    确定性Scrambled FFT测量矩阵
    
    快速且安全
    
    参数:
        m, n: 矩阵维度
        key: 密钥
        index: 矩阵索引
    
    返回:
        Phi: 测量矩阵
    """
    # 生成确定性随机相位
    seed_bytes = hashlib.sha256(f"{key}_fft_{index}".encode()).digest()
    seed = int.from_bytes(seed_bytes[:4], 'big') % (2**32)
    np.random.seed(seed)
    
    # 随机相位
    phases = np.random.rand(n) * 2 * np.pi
    
    # FFT矩阵
    fft_matrix = np.fft.fft(np.eye(n)) / np.sqrt(n)
    
    # 随机列置换
    perm = np.random.permutation(n)
    scrambled = fft_matrix[:, perm]
    
    # 取前m行
    Phi = np.real(scrambled[:m, :])
    
    # 归一化
    norms = np.sqrt(np.sum(Phi**2, axis=1, keepdims=True))
    Phi = Phi / (norms + 1e-10)
    
    return Phi


def optimized_3d_measurement(cube, method="bernoulli", key="default", m_rate=0.7):
    """
    优化的3D测量
    
    参数:
        cube: 3D数据块 (H x W x T)
        method: 测量矩阵方法 ("gaussian", "bernoulli", "fft")
        key: 密钥
        m_rate: 压缩率
    
    返回:
        measured: 测量结果
        metadata: 元数据
    """
    H, W, T = cube.shape
    n = H * W * T
    m = int(n * m_rate)
    
    # 根据方法选择测量矩阵
    if method == "gaussian":
        Phi = np.random.randn(m, n) / np.sqrt(m)
    elif method == "bernoulli":
        Phi = create_deterministic_bernoulli(m, n, key)
    elif method == "fft":
        Phi = scrambled_fft_measurement(m, n, key)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 展平并测量
    x = cube.flatten()
    y = Phi @ x
    
    return y, {'method': method, 'm': m, 'n': n, 'key': key}


# ============= 测试 =============
if __name__ == "__main__":
    import time
    
    print("Testing video optimizations...")
    
    # 测试1: 自适应关键帧检测
    print("\n[1] Adaptive Keyframe Detection")
    # 生成测试帧 (模拟视频)
    np.random.seed(42)
    test_frames = []
    for i in range(30):
        if i < 10:
            # 场景1: 静态
            frame = np.random.randint(100, 120, (32, 32), dtype=np.uint8)
        elif i < 15:
            # 场景2: 切换
            frame = np.random.randint(180, 200, (32, 32), dtype=np.uint8)
        else:
            # 场景3: 淡入
            frame = np.random.randint(100 + i*2, 120 + i*2, (32, 32), dtype=np.uint8)
        test_frames.append(frame)
    
    keyframes = adaptive_keyframe_detection(test_frames, threshold=0.3, min_key_distance=5)
    print(f"    Detected keyframes: {keyframes}")
    print(f"    Total keyframes: {len(keyframes)} / {len(test_frames)}")
    
    # 测试2: Bernoulli矩阵
    print("\n[2] Bernoulli Measurement Matrix")
    t0 = time.time()
    Phi_bern = bernoulli_measurement_matrix(100, 256, seed=42)
    t1 = time.time()
    print(f"    Bernoulli (100x256): {t1-t0:.4f}s")
    
    t0 = time.time()
    Phi_gauss = np.random.randn(100, 256) / np.sqrt(100)
    t1 = time.time()
    print(f"    Gaussian (100x256): {t1-t0:.4f}s")
    
    # 测试3: 确定性Bernoulli
    print("\n[3] Deterministic Bernoulli")
    Phi1 = create_deterministic_bernoulli(10, 20, "test-key", 0)
    Phi2 = create_deterministic_bernoulli(10, 20, "test-key", 0)
    Phi3 = create_deterministic_bernoulli(10, 20, "test-key", 1)
    
    print(f"    Same key, same index: {np.allclose(Phi1, Phi2)}")
    print(f"    Same key, diff index: {np.allclose(Phi1, Phi3)}")
    
    print("\n>>> Video optimization tests completed!")
