# -*- coding: utf-8 -*-
"""
AMP (Approximate Message Passing) 重建算法
用于压缩感知图像重建，比FISTA收敛更快
"""

import numpy as np
from typing import Optional

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not available, using pure numpy")


def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """软阈值函数"""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


def omp(Phi: np.ndarray, y: np.ndarray, k: int, tol: float = 1e-6) -> np.ndarray:
    """
    OMP (Orthogonal Matching Pursuit) 算法
    适合稀疏信号重建
    """
    n = Phi.shape[1]
    x = np.zeros(n)
    residual = y.copy()
    indices = []
    
    for _ in range(k):
        # 找与残差最相关的列
        correlations = np.abs(Phi.T @ residual)
        correlations[indices] = -1  # 已选列排除
        idx = np.argmax(correlations)
        
        if correlations[idx] < tol:
            break
            
        indices.append(idx)
        
        # 最小二乘更新
        Phi_k = Phi[:, indices]
        x_k = np.linalg.lstsq(Phi_k, y, rcond=None)[0]
        
        # 更新残差
        residual = y - Phi_k @ x_k
    
    x[indices] = x_k
    return x


def amp_l1(Phi: np.ndarray, y: np.ndarray, 
           lam: float = 0.01, 
           max_iter: int = 100, 
           tol: float = 1e-5,
           beta: float = 1.0) -> np.ndarray:
    """
    AMP (Approximate Message Passing) 算法
    
    参数:
        Phi: 测量矩阵 (m x n)
        y: 测量向量 (m)
        lam: L1正则化参数
        max_iter: 最大迭代次数
        tol: 收敛阈值
        beta: 步长因子 (0-1)
    
    返回:
        x: 重建的稀疏向量 (n)
    
    参考:
        Donoho, D. L., Maleki, A., & Montanari, A. (2009)
        "Message-passing algorithms for compressed sensing"
    """
    m, n = Phi.shape
    
    # 初始化
    x = np.zeros(n, dtype=np.float64)
    z = y.copy()  # 残差
    sigma_sq = np.mean(z**2)  # 噪声方差估计
    
    # 计算Phi的列范数归一化
    col_norms = np.sqrt(np.sum(Phi**2, axis=0))
    col_norms[col_norms == 0] = 1
    Phi_normalized = Phi / col_norms
    
    prev_obj = 1e30
    
    for it in range(max_iter):
        # AMP更新公式: x = soft_threshold(Phi^T z + x, lam)
        # 引入Onsager校正项
        sigma_z = np.sqrt(np.mean(z**2))
        if sigma_z < 1e-10:
            sigma_z = 1e-10
            
        # 消息传递
        message = Phi_normalized.T @ z + x
        
        # 软阈值
        x_new = soft_threshold(message, lam * sigma_z * beta)
        
        # 更新残差 (AMP关键公式)
        # z = y - Phi x + (1/m) * z * sum(soft_threshold'(message))
        # soft_threshold的导数近似为指示函数
        shrinkage = np.abs(message) > lam * sigma_z * beta
        eta = np.sum(shrinkage) / n
        
        z_new = y - Phi_normalized @ x_new + z * eta
        
        # 收敛检查
        obj = 0.5 * np.linalg.norm(y - Phi_normalized @ x_new)**2 + lam * np.sum(np.abs(x_new))
        
        if abs(prev_obj - obj) / (abs(prev_obj) + 1e-12) < tol:
            x = x_new * col_norms  # 反归一化
            break
            
        prev_obj = obj
        x = x_new
        z = z_new
    
    return x * col_norms  # 反归一化


def amp_l1_numba(Phi: np.ndarray, y: np.ndarray, 
                 lam: float = 0.01, 
                 max_iter: int = 100, 
                 tol: float = 1e-5) -> np.ndarray:
    """
    Numba加速版AMP算法
    """
    if not NUMBA_AVAILABLE:
        return amp_l1(Phi, y, lam, max_iter, tol)
    
    m, n = Phi.shape
    x = np.zeros(n, dtype=np.float64)
    z = y.copy().astype(np.float64)
    
    col_norms = np.sqrt(np.sum(Phi**2, axis=0))
    col_norms[col_norms == 0] = 1
    Phi_n = Phi / col_norms
    
    prev_obj = 1e30
    
    for _ in range(max_iter):
        sigma_z = np.sqrt(np.mean(z**2))
        if sigma_z < 1e-10:
            sigma_z = 1e-10
        
        message = Phi_n.T @ z + x
        
        # 软阈值
        x_new = np.sign(message) * np.maximum(np.abs(message) - lam * sigma_z, 0)
        
        # AMP残差更新
        shrinkage = np.abs(message) > lam * sigma_z
        eta = np.sum(shrinkage) / n
        z = y - Phi_n @ x_new + z * eta
        
        obj = 0.5 * np.linalg.norm(y - Phi_n @ x_new)**2 + lam * np.sum(np.abs(x_new))
        
        if abs(prev_obj - obj) / (abs(prev_obj) + 1e-12) < tol:
            break
        prev_obj = obj
        x = x_new
    
    return x * col_norms


def damsm(Phi: np.ndarray, y: np.ndarray,
          lam: float = 0.01,
          max_iter: int = 100,
          tol: float = 1e-5) -> np.ndarray:
    """
AMP (Denoising AMP)    D- 算法
    使用BM3D等高级去噪器
    
    这是一个简化版本，使用软阈值作为去噪器
    """
    m, n = Phi.shape
    
    x = np.zeros(n)
    z = y.copy()
    
    for it in range(max_iter):
        # 消息传递
        x_hat = x + Phi.T @ z
        
        # 去噪 (这里用软阈值，可替换为BM3D/TV等)
        x_new = soft_threshold(x_hat, lam * np.sqrt(m))
        
        # 残差更新
        z = y - Phi @ x_new + z * np.sum(x_new != x) / m
        
        # 收敛检查
        if np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-10) < tol:
            x = x_new
            break
        x = x_new
    
    return x


def cosamp(Phi: np.ndarray, y: np.ndarray, 
          k: int, 
          max_iter: int = 100,
          tol: float = 1e-5) -> np.ndarray:
    """
    CoSaMP (Compressive Sampling Matching Pursuit) 算法
    适合已知稀疏度的场景
    """
    m, n = Phi.shape
    x = np.zeros(n)
    t = 0
    r = y.copy()
    support = set()
    
    while t < max_iter:
        # 找相关列
        c = np.abs(Phi.T @ r)
        c[list(support)] = -1  # 已选列排除
        
        # 合并支持集
        Gamma = support | set(np.argsort(c)[-2*k:])
        
        # 最小二乘
        Phi_G = Phi[:, list(Gamma)]
        a = np.linalg.lstsq(Phi_G, y, rcond=None)[0]
        
        # 保留最大的k个
        new_support = set(Gamma)
        x_new = np.zeros(n)
        x_new[list(Gamma)] = a
        
        # 裁剪到k个最大元素
        if np.sum(np.abs(a) > 0) > k:
            indices = np.argsort(np.abs(a))[-k:]
            x_new = np.zeros(n)
            x_new[Gamma[indices]] = a[indices]
            support = set(Gamma[indices])
        else:
            support = new_support
        
        # 更新残差
        r = y - Phi @ x_new
        
        # 收敛检查
        if np.linalg.norm(r) < tol:
            break
        t += 1
    
    return x_new


def select_reconstructor(method: str = "amp"):
    """
    工厂函数：选择重建算法
    """
    methods = {
        "fista": None,  # 保持原有
        "amp": amp_l1,
        "amp_numba": amp_l1_numba,
        "omp": lambda Phi, y, lam, max_iter, tol: omp(Phi, y, k=int(max(lam * Phi.shape[1], 5)), tol=tol),
        "cosamp": lambda Phi, y, lam, max_iter, tol: cosamp(Phi, y, k=int(lam * Phi.shape[1]), max_iter=max_iter, tol=tol),
        "damsm": damsm,
    }
    
    if method not in methods:
        print(f"Warning: Unknown method {method}, using AMP")
        method = "amp"
    
    return methods.get(method)


# ============= 测试代码 =============
if __name__ == "__main__":
    # 简单测试
    np.random.seed(42)
    
    # 创建测试信号
    n = 64
    m = 32
    k = 5  # 稀疏度
    
    # 随机稀疏信号
    x_true = np.zeros(n)
    indices = np.random.choice(n, k, replace=False)
    x_true[indices] = np.random.randn(k)
    
    # 测量矩阵
    Phi = np.random.randn(m, n) / np.sqrt(m)
    
    # 测量
    y = Phi @ x_true + 0.01 * np.random.randn(m)
    
    # 测试AMP
    x_amp = amp_l1(Phi, y, lam=0.1, max_iter=50)
    
    # 计算误差
    mse_amp = np.linalg.norm(x_amp - x_true) / np.linalg.norm(x_true)
    
    print(f"AMP重建相对误差: {mse_amp:.4f}")
    print(f"原始稀疏度: {k}, AMP非零元素: {np.sum(np.abs(x_amp) > 0.01)}")
