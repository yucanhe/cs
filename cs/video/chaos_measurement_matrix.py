#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于混沌序列的测量矩阵生成器
用于视频/图像压缩感知加密
"""

import math
import numpy as np
import hashlib


def spcmm3_stream(master_key: str, cube_id: int, block_id: int, purpose: str):
    """
    3D-SPCMM 混沌流生成器
    
    参数:
        master_key: 主密钥
        cube_id: 立方体ID
        block_id: 块ID
        purpose: 用途标签
    
    生成:
        混沌序列值 (0,1)
    """
    # 从密钥派生初始条件
    seed = chaos_seed(master_key, f"cube={cube_id}", f"blk={block_id}", purpose)
    
    # 3D-SPCMM参数
    a = 3.99
    k = 0.13
    
    x, y, z = seed[0], seed[1], seed[2]
    
    # 预热
    for _ in range(64):
        x = a * x * (1.0 - x)
        y = a * y * (1.0 - y)
        z = a * z * (1.0 - z)
        x = (x + k * (y - z)) % 1.0
        y = (y + k * (z - x)) % 1.0
        z = (z + k * (x - y)) % 1.0
    
    # 生成序列
    while True:
        x = a * x * (1.0 - x)
        y = a * y * (1.0 - y)
        z = a * z * (1.0 - z)
        x = (x + k * (y - z)) % 1.0
        y = (y + k * (z - x)) % 1.0
        z = (z + k * (x - y)) % 1.0
        yield x
        yield y
        yield z


def chaos_seed(master_key: str, *tags: str) -> tuple:
    """
    从密钥派生混沌初始条件
    """
    b = master_key.encode("utf-8") + b"|" + "|".join(tags).encode("utf-8")
    h = hashlib.sha256(b).digest()
    
    seeds = []
    for i in range(6):
        v = int.from_bytes(h[i*4:(i+1)*4], "big") & ((1 << 53) - 1)
        x = (v + 1) / (2**53 + 2)
        seeds.append(float(x))
    
    return tuple(seeds[:3])


def chaos_gaussian_measurement(master_key: str, cube_id: int, block_id: int, m: int, n: int):
    """
    混沌高斯测量矩阵 (Box-Muller变换)
    
    参数:
        master_key: 主密钥
        cube_id: 立方体ID
        block_id: 块ID
        m: 测量数
        n: 信号维度
    
    返回:
        Phi: 测量矩阵 (m x n)
    """
    g = spcmm3_stream(master_key, cube_id, block_id, f"gauss_m={m}_n={n}")
    vals = np.empty(m * n, dtype=np.float64)
    i = 0
    while i < m * n:
        u1 = max(next(g), 1e-12)
        u2 = next(g)
        r = math.sqrt(-2.0 * math.log(u1))
        z0 = r * math.cos(2 * math.pi * u2)
        vals[i] = z0
        i += 1
    
    Phi = vals.reshape(m, n)
    # 行归一化
    denom = np.sqrt(np.sum(Phi * Phi, axis=1, keepdims=True)) + 1e-8
    Phi = Phi / denom
    return Phi.astype(np.float32)


def chaos_bernoulli_measurement(master_key: str, cube_id: int, block_id: int, m: int, n: int):
    """
    混沌Bernoulli测量矩阵 (±1)
    
    比高斯更快，适合硬件实现
    
    参数:
        master_key: 主密钥
        cube_id: 立方体ID
        block_id: 块ID
        m: 测量数
        n: 信号维度
    
    返回:
        Phi: 测量矩阵 (m x n)
    """
    g = spcmm3_stream(master_key, cube_id, block_id, f"bern_m={m}_n={n}")
    vals = np.empty(m * n, dtype=np.float32)
    i = 0
    while i < m * n:
        v = next(g)
        # 0.5概率为+1，0.5概率为-1
        vals[i] = 1.0 if v >= 0.5 else -1.0
        i += 1
    
    Phi = vals.reshape(m, n)
    # 行归一化
    denom = np.sqrt(np.sum(Phi * Phi, axis=1, keepdims=True)) + 1e-8
    Phi = Phi / denom
    return Phi


def chaos_uniform_measurement(master_key: str, cube_id: int, block_id: int, m: int, n: int):
    """
    混沌均匀测量矩阵 (0,1)均匀分布
    
    参数:
        master_key: 主密钥
        cube_id: 立方体ID
        block_id: 块ID
        m: 测量数
        n: 信号维度
    
    返回:
        Phi: 测量矩阵 (m x n)
    """
    g = spcmm3_stream(master_key, cube_id, block_id, f"unif_m={m}_n={n}")
    vals = np.empty(m * n, dtype=np.float32)
    i = 0
    while i < m * n:
        vals[i] = next(g)
        i += 1
    
    Phi = vals.reshape(m, n)
    # 行归一化
    denom = np.sqrt(np.sum(Phi * Phi, axis=1, keepdims=True)) + 1e-8
    Phi = Phi / denom
    return Phi


def chaos_scrambled_hadamard(master_key: str, cube_id: int, block_id: int, m: int, n: int):
    """
    混沌置换Hadamard矩阵
    
    确定性结构化矩阵，快速且满足RIP
    
    参数:
        master_key: 主密钥
        cube_id: 立方体ID
        block_id: 块ID
        m: 测量数 (必须为2的幂次)
        n: 信号维度
    
    返回:
        Phi: 测量矩阵 (m x n)
    """
    # 找到最近的2的幂次
    n_fft = 1
    while n_fft < n:
        n_fft *= 2
    
    # 生成Hadamard矩阵 (Sylvester方法)
    H = np.ones((n_fft, n_fft), dtype=np.float32)
    for i in range(int(np.log2(n_fft))):
        step = 2 ** (i + 1)
        for j in range(0, n_fft, step):
            for k in range(0, step // 2):
                H[j:j+step, k] = np.concatenate([H[j:j+step//2, k], -H[j:j+step//2, k]])
                H[j:j+step, k + step // 2] = np.concatenate([H[j:j+step//2, k + step // 2], -H[j:j+step//2, k + step // 2]])
    
    # 混沌置换
    g = spcmm3_stream(master_key, cube_id, block_id, f"hadamard_perm_{n_fft}")
    perm = []
    remaining = list(range(n_fft))
    for _ in range(n_fft):
        idx = int(next(g) * len(remaining))
        perm.append(remaining.pop(idx))
    
    # 应用置换
    H_perm = H[perm, :n]
    
    # 取前m行
    Phi = H_perm[:m, :] / np.sqrt(m)
    
    return Phi.astype(np.float32)


def chaos_scrambled_fft(master_key: str, cube_id: int, block_id: int, m: int, n: int):
    """
    混沌置换FFT矩阵
    
    确定性结构化，快速
    
    参数:
        master_key: 主密钥
        cube_id: 立方体ID
        block_id: 块ID
        m: 测量数
        n: 信号维度
    
    返回:
        Phi: 测量矩阵 (m x n)
    """
    # 找到最近的2的幂次
    n_fft = 1
    while n_fft < n:
        n_fft *= 2
    
    # 生成FFT矩阵
    F = np.fft.fft(np.eye(n_fft), norm='ortho')
    
    # 混沌行置换 + 列置换
    g = spcmm3_stream(master_key, cube_id, block_id, f"fft_perm_{n_fft}")
    
    # 行置换
    row_perm = []
    remaining = list(range(n_fft))
    for _ in range(n_fft):
        idx = int(next(g) * len(remaining))
        row_perm.append(remaining.pop(idx))
    
    # 列置换
    col_perm = []
    remaining = list(range(n_fft))
    for _ in range(n_fft):
        idx = int(next(g) * len(remaining))
        col_perm.append(remaining.pop(idx))
    
    # 应用置换
    F_perm = F[row_perm, :][:, col_perm]
    
    # 取前m行，前n列
    Phi = np.real(F_perm[:m, :n])
    
    # 行归一化
    denom = np.sqrt(np.sum(Phi * Phi, axis=1, keepdims=True)) + 1e-8
    Phi = Phi / denom
    
    return Phi.astype(np.float32)


def chaos_partial_circulant(master_key: str, cube_id: int, block_id: int, m: int, n: int):
    """
    混沌部分循环矩阵
    
    结构化矩阵，适合时序信号
    
    参数:
        master_key: 主密钥
        cube_id: 立方体ID
        block_id: 块ID
        m: 测量数
        n: 信号维度
    
    返回:
        Phi: 测量矩阵 (m x n)
    """
    # 首先生成n个混沌值作为种子
    g = spcmm3_stream(master_key, cube_id, block_id, f"circ_seed_{n}")
    seed = np.array([next(g) for _ in range(n)], dtype=np.float32)
    
    # 构建循环矩阵
    rows = []
    for i in range(m):
        row = np.roll(seed, i)[:n]
        rows.append(row)
    
    Phi = np.array(rows, dtype=np.float32)
    
    # 行归一化
    denom = np.sqrt(np.sum(Phi * Phi, axis=1, keepdims=True)) + 1e-8
    Phi = Phi / denom
    
    return Phi


# 测量矩阵类型映射
MEASUREMENT_MATRIX_TYPES = {
    "gaussian": chaos_gaussian_measurement,
    "bernoulli": chaos_bernoulli_measurement,
    "uniform": chaos_uniform_measurement,
    "hadamard": chaos_scrambled_hadamard,
    "fft": chaos_scrambled_fft,
    "circulant": chaos_partial_circulant,
}


def get_chaos_measurement_matrix(method: str, master_key: str, cube_id: int, block_id: int, m: int, n: int):
    """
    获取混沌测量矩阵的工厂函数
    
    参数:
        method: 矩阵类型 ("gaussian", "bernoulli", "uniform", "hadamard", "fft", "circulant")
        master_key: 主密钥
        cube_id: 立方体ID
        block_id: 块ID
        m: 测量数
        n: 信号维度
    
    返回:
        Phi: 测量矩阵
    """
    if method not in MEASUREMENT_MATRIX_TYPES:
        raise ValueError(f"Unknown method: {method}. Available: {list(MEASUREMENT_MATRIX_TYPES.keys())}")
    
    return MEASUREMENT_MATRIX_TYPES[method](master_key, cube_id, block_id, m, n)


# ============= 测试 =============
if __name__ == "__main__":
    import time
    
    print("="*60)
    print("Chaos-based Measurement Matrix Generator Test")
    print("="*60)
    
    master_key = "test-key-2026"
    cube_id = 0
    block_id = 0
    m, n = 64, 256
    
    methods = ["gaussian", "bernoulli", "uniform", "hadamard", "fft", "circulant"]
    
    print(f"\nMatrix size: {m} x {n}")
    print("-"*60)
    
    for method in methods:
        t0 = time.time()
        try:
            Phi = get_chaos_measurement_matrix(method, master_key, cube_id, block_id, m, n)
            t1 = time.time()
            
            # 验证矩阵性质
            norms = np.sqrt(np.sum(Phi**2, axis=1))
            print(f"{method:12s}: shape={Phi.shape}, time={t1-t0:.4f}s, row_norm_mean={norms.mean():.4f}")
        except Exception as e:
            print(f"{method:12s}: ERROR - {e}")
    
    print("\n>>> Test completed!")
