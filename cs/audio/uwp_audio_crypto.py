#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频小波包变换加密模块
使用小波包变换(UWP)替代STFT+DCT，提供更好的时频局部化
"""

import numpy as np
import pywt
from scipy.signal import stft, istft
from scipy.fftpack import dct, idct


def uwp_decompose(data, wavelet='db8', level=3, mode='symmetric'):
    """
    小波包分解
    
    参数:
        data: 输入信号 (1D)
        wavelet: 小波基 ('db8', 'haar', 'sym4', etc.)
        level: 分解层数
        mode: 边界延拓模式
    
    返回:
        小波包树结构
    """
    wp = pywt.WaveletPacket(data, wavelet, mode=mode, maxlevel=level)
    return wp


def uwp_reconstruct(wp):
    """
    小波包重构
    """
    return wp.reconstruct()


def uwp_encrypt_audio(x, fs, key, 
                      m_rate=0.9, 
                      wavelet='db8', 
                      level=3,
                      cs_iters=100,
                      lam=0.01):
    """
    基于小波包变换的音频加密
    
    参数:
        x: 输入音频 (float, -1 to 1)
        fs: 采样率
        key: 加密密钥
        m_rate: 压缩率
        wavelet: 小波基
        level: 分解层数
        cs_iters: CS重建迭代次数
        lam: L1正则化参数
    
    返回:
        encrypted_data: 加密后的数据
        metadata: 元数据
    """
    n = len(x)
    
    # 小波包分解
    wp = pywt.WaveletPacket(x, wavelet, mode='symmetric', maxlevel=level)
    
    # 获取所有小波包节点
    nodes = wp.get_level(level, decompose=False)
    
    # 对每个节点系数进行CS压缩
    encrypted_coefs = {}
    for path, node in nodes:
        coef = node.data
        m = int(len(coef) * m_rate)
        
        if m >= len(coef):
            # 不压缩
            encrypted_coefs[path] = coef
        else:
            # 创建测量矩阵
            np.random.seed(hash(key + path) % (2**32))
            Phi = np.random.randn(m, len(coef)) / np.sqrt(m)
            
            # 测量
            y = Phi @ coef
            
            # 加密 (简单的xor混淆)
            np.random.seed(hash(key + path + "enc") % (2**32))
            key_stream = np.random.randint(0, 256, size=m, dtype=np.uint8)
            y_enc = np.packbits((y > 0).astype(np.uint8)) ^ key_stream
            
            encrypted_coefs[path] = {
                'y': y,
                'key_stream': key_stream,
                'm': m,
                'n': len(coef)
            }
    
    return encrypted_coefs, {
        'n': n,
        'fs': fs,
        'wavelet': wavelet,
        'level': level,
        'm_rate': m_rate
    }


def uwp_decrypt_audio(encrypted_coefs, metadata, key,
                      cs_iters=100,
                      lam=0.01):
    """
    基于小波包变换的音频解密
    """
    wavelet = metadata['wavelet']
    level = metadata['level']
    m_rate = metadata['m_rate']
    
    # 重建小波包树
    wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=level)
    
    # 解密每个节点
    for path, node in wp.get_level(level, decompose=False):
        if isinstance(encrypted_coefs[path], dict):
            # CS重建
            y = encrypted_coefs[path]['y']
            m = encrypted_coefs[path]['m']
            n = encrypted_coefs[path]['n']
            
            # 创建测量矩阵
            np.random.seed(hash(key + path) % (2**32))
            Phi = np.random.randn(m, n) / np.sqrt(m)
            
            # OMP重建 (简化版)
            coef = omp_reconstruct(Phi, y, k=int(m/4))
        else:
            coef = encrypted_coefs[path]
        
        # 设置节点系数
        wp[path].data = coef
    
    # 重构信号
    x_rec = wp.reconstruct()
    
    return x_rec[:metadata['n']]


def omp_reconstruct(Phi, y, k=10, tol=1e-6):
    """
    OMP (正交匹配追踪) 重建算法
    """
    n = Phi.shape[1]
    x = np.zeros(n)
    residual = y.copy()
    indices = []
    
    for _ in range(k):
        # 找最相关列
        correlations = np.abs(Phi.T @ residual)
        correlations[indices] = -1
        idx = np.argmax(correlations)
        
        if correlations[idx] < tol:
            break
        
        indices.append(idx)
        
        # 最小二乘
        Phi_k = Phi[:, indices]
        x_k = np.linalg.lstsq(Phi_k, y, rcond=None)[0]
        
        # 更新残差
        residual = y - Phi_k @ x_k
    
    x[indices] = x_k
    return x


def hybrid_stft_uwp_encrypt(x, fs, key,
                            m_rate=0.9,
                            nperseg=1024,
                            noverlap=512,
                            wavelet='db8',
                            level=3):
    """
    混合STFT+小波包加密
    使用STFT获取时频表示，然后用小波包进行进一步分解
    """
    # STFT
    f, t, Z = stft(x, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, 
                   boundary='zeros', padded=True)
    
    mag = np.abs(Z)
    phase = np.angle(Z)
    
    # 对幅度进行小波包分解
    F, T = mag.shape
    encrypted_mag = np.zeros_like(mag)
    
    for fi in range(F):
        # 小波包分解
        wp = pywt.WaveletPacket(mag[fi, :], wavelet, mode='symmetric', maxlevel=level)
        
        # 简单加密: 对低频系数进行更强的保护
        nodes = wp.get_level(level, decompose=False)
        for path, node in nodes:
            # 混沌置乱
            np.random.seed(hash(key + f"{fi}_{path}") % (2**32))
            perm = np.random.permutation(len(node.data))
            node.data = node.data[perm]
        
        # 重构
        encrypted_mag[fi, :] = wp.reconstruct()
    
    # 重建加密音频
    Z_enc = encrypted_mag * np.exp(1j * phase)
    x_enc, t = istft(Z_enc, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    
    return x_enc[:len(x)], {'method': 'hybrid_stft_uwp', 'nperseg': nperseg, 'wavelet': wavelet, 'level': level}


# ============================================================
# 简化的UWP-CS加密 (推荐使用)
# ============================================================

def uwp_cs_encrypt_simple(x, key, m_rate=0.8, wavelet='db4', level=4):
    """
    简化的UWP-CS加密
    1) 对信号进行小波包分解
    2) 对小波系数进行CS压缩测量
    3) 混沌置乱系数位置
    """
    n = len(x)
    
    # 小波包分解
    wp = pywt.WaveletPacket(x, wavelet, mode='per', maxlevel=level)
    coeffs = wp.data
    
    # CS测量
    m = int(n * m_rate)
    np.random.seed(sum(ord(c) for c in key))
    Phi = np.random.randn(m, n) / np.sqrt(m)
    y = Phi @ coeffs
    
    # 混沌置乱
    np.random.seed(sum(ord(c) for c in key) + 1)
    perm = np.random.permutation(m)
    y_perm = y[perm]
    
    # 加密测量值
    np.random.seed(sum(ord(c) for c in key) + 2)
    mask = np.random.randint(0, 2, size=m)
    y_enc = y_perm * (-1)**mask
    
    return {
        'y_enc': y_enc,
        'perm': perm,
        'mask': mask,
        'n': n,
        'm': m,
        'wavelet': wavelet,
        'level': level
    }


def uwp_cs_decrypt_simple(encrypted, key):
    """
    简化的UWP-CS解密
    """
    y_enc = encrypted['y_enc']
    perm = encrypted['perm']
    mask = encrypted['mask']
    n = encrypted['n']
    m = encrypted['m']
    wavelet = encrypted['wavelet']
    level = encrypted['level']
    
    # 逆加密
    y_perm = y_enc * (-1)**mask
    
    # 逆置乱
    inv_perm = np.argsort(perm)
    y = y_perm[inv_perm]
    
    # CS重建 (使用简单的线性重建作为近似)
    np.random.seed(sum(ord(c) for c in key))
    Phi = np.random.randn(m, n) / np.sqrt(m)
    
    # 伪逆重建
    coeffs = np.linalg.pinv(Phi) @ y
    
    # 小波包重构
    wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='per', maxlevel=level)
    wp.data = coeffs[:n]
    x_rec = wp.reconstruct()
    
    return x_rec[:n]


# ============= 测试 =============
if __name__ == "__main__":
    import time
    from scipy.io import wavfile
    
    # 测试
    print("Testing UWP audio encryption...")
    
    # 加载测试音频
    fs, x = wavfile.read("../resources/audio/1.wav")
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32) / 32768.0
    
    print(f"Audio length: {len(x)}, Sample rate: {fs}")
    
    # 测试混合加密
    print("\n[1] Testing hybrid STFT+UWP encryption...")
    t0 = time.time()
    x_enc, meta = hybrid_stft_uwp_encrypt(x, fs, "test-key", m_rate=0.9, wavelet='db8', level=3)
    t1 = time.time()
    print(f"    Time: {t1-t0:.2f}s")
    print(f"    Encrypted audio range: [{x_enc.min():.3f}, {x_enc.max():.3f}]")
    
    # 比较频谱
    from scipy.signal import spectrogram
    f1, t1, S1 = spectrogram(x, fs=fs, nperseg=256)
    f2, t2, S2 = spectrogram(x_enc, fs=fs, nperseg=256)
    
    diff = np.mean(np.abs(S1 - S2))
    print(f"    Spectrogram difference: {diff:.2f}")
    
    print("\n>>> UWP encryption test completed!")
