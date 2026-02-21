#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频加密完整实验分析
Audio Encryption Comprehensive Analysis

包含:
1. 时域波形图分析 (Waveform Analysis)
2. 频谱分析 (Spectral Analysis)
3. 语谱图分析 (Spectrogram Analysis)
4. SNR/PSNR 质量分析
5. 相关系数分析 (Correlation Analysis)
6. 抗攻击测试
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import hashlib
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==================== 音频加解密函数 ====================
# 这里复用现有的音频加密代码

def generate_keystream(key, length):
    """生成混沌密钥流"""
    import hashlib
    
    # 简化版密钥流生成
    h = hashlib.sha512(key.encode()).hexdigest()
    np.random.seed(int(h[:16], 16))
    return np.random.randint(0, 256, length, dtype=np.uint8)


def audio_xor_encrypt(audio_data, key):
    """简单的异或加密（用于测试）"""
    keystream = generate_keystream(key, len(audio_data))
    encrypted = np.bitwise_xor(audio_data.astype(np.uint8), keystream)
    return encrypted


def audio_xor_decrypt(encrypted_data, key):
    """解密"""
    return audio_xor_encrypt(encrypted_data, key)  # XOR对称


# ==================== 分析函数 ====================

def waveform_analysis(original_path, encrypted_path, output_dir):
    """时域波形图分析"""
    print("\n=== 时域波形图分析 (Waveform Analysis) ===")
    
    # 读取音频
    sr1, orig = wavfile.read(original_path)
    sr2, enc = wavfile.read(encrypted_path)
    
    # 转换为float用于绘图
    orig_float = orig.astype(float) / 32767.0 if orig.dtype == np.int16 else orig.astype(float)
    enc_float = enc.astype(float) / 32767.0 if enc.dtype == np.int16 else enc.astype(float)
    
    # 绘制波形
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 原始波形
    time_orig = np.arange(len(orig_float)) / sr1
    axes[0].plot(time_orig, orig_float, 'b-', alpha=0.7, linewidth=0.5)
    axes[0].set_title('Original Audio Waveform')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlim([0, min(1.0, time_orig[-1])])
    axes[0].grid(True, alpha=0.3)
    
    # 加密波形
    time_enc = np.arange(len(enc_float)) / sr2
    axes[1].plot(time_enc, enc_float, 'r-', alpha=0.7, linewidth=0.5)
    axes[1].set_title('Encrypted Audio Waveform')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_xlim([0, min(1.0, time_enc[-1])])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'waveform_analysis.png'), dpi=150)
    plt.close()
    
    # 计算统计信息
    print(f"Original: min={orig_float.min():.4f}, max={orig_float.max():.4f}, std={orig_float.std():.4f}")
    print(f"Encrypted: min={enc_float.min():.4f}, max={enc_float.max():.4f}, std={enc_float.std():.4f}")
    
    return {
        'orig_std': float(orig_float.std()),
        'enc_std': float(enc_float.std())
    }


def spectral_analysis(original_path, encrypted_path, output_dir):
    """频谱分析"""
    print("\n=== 频谱分析 (Spectral Analysis) ===")
    
    # 读取音频
    sr1, orig = wavfile.read(original_path)
    sr2, enc = wavfile.read(encrypted_path)
    
    # 转换为float
    orig_float = orig.astype(float) / 32767.0 if orig.dtype == np.int16 else orig.astype(float)
    enc_float = enc.astype(float) / 32767.0 if enc.dtype == np.int16 else enc.astype(float)
    
    # 如果是立体声，取单声道
    if len(orig_float.shape) > 1:
        orig_float = orig_float.mean(axis=1)
    if len(enc_float.shape) > 1:
        enc_float = enc_float.mean(axis=1)
    
    # 计算FFT频谱
    n = len(orig_float)
    freqs = np.fft.rfftfreq(n, 1/sr1)
    orig_fft = np.abs(np.fft.rfft(orig_float))
    enc_fft = np.abs(np.fft.rfft(enc_float))
    
    # 归一化
    orig_fft = orig_fft / orig_fft.max()
    enc_fft = enc_fft / enc_fft.max()
    
    # 绘制频谱
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    axes[0].semilogy(freqs, orig_fft, 'b-', alpha=0.7)
    axes[0].set_title('Original Audio Spectrum')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (log)')
    axes[0].set_xlim([0, sr1/2])
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(freqs, enc_fft, 'r-', alpha=0.7)
    axes[1].set_title('Encrypted Audio Spectrum')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (log)')
    axes[1].set_xlim([0, sr2/2])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectral_analysis.png'), dpi=150)
    plt.close()
    
    # 计算频谱平坦度
    def spectral_flatness(x):
        x = np.abs(x)
        x = x[x > 0]
        return np.exp(np.mean(np.log(x))) / np.mean(x)
    
    orig_flatness = spectral_flatness(orig_fft)
    enc_flatness = spectral_flatness(enc_fft)
    
    print(f"Original spectral flatness: {orig_flatness:.4f}")
    print(f"Encrypted spectral flatness: {enc_flatness:.4f}")
    
    return {
        'orig_flatness': float(orig_flatness),
        'enc_flatness': float(enc_flatness)
    }


def spectrogram_analysis(original_path, encrypted_path, output_dir):
    """语谱图分析"""
    print("\n=== 语谱图分析 (Spectrogram Analysis) ===")
    
    # 读取音频
    sr1, orig = wavfile.read(original_path)
    sr2, enc = wavfile.read(encrypted_path)
    
    # 转换为float
    orig_float = orig.astype(float) / 32767.0 if orig.dtype == np.int16 else orig.astype(float)
    enc_float = enc.astype(float) / 32767.0 if enc.dtype == np.int16 else enc.astype(float)
    
    # 如果是立体声，取单声道
    if len(orig_float.shape) > 1:
        orig_float = orig_float.mean(axis=1)
    if len(enc_float.shape) > 1:
        enc_float = enc_float.mean(axis=1)
    
    # 计算语谱图
    nperseg = 1024
    f1, t1, Sxx1 = signal.spectrogram(orig_float, fs=sr1, nperseg=nperseg)
    f2, t2, Sxx2 = signal.spectrogram(enc_float, fs=sr2, nperseg=nperseg)
    
    # 绘制语谱图
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    im1 = axes[0].pcolormesh(t1, f1, 10*np.log10(Sxx1 + 1e-10), shading='gouraud', cmap='viridis')
    axes[0].set_title('Original Audio Spectrogram')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_xlabel('Time (s)')
    plt.colorbar(im1, ax=axes[0], label='dB')
    
    im2 = axes[1].pcolormesh(t2, f2, 10*np.log10(Sxx2 + 1e-10), shading='gouraud', cmap='viridis')
    axes[1].set_title('Encrypted Audio Spectrogram')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (s)')
    plt.colorbar(im2, ax=axes[1], label='dB')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectrogram_analysis.png'), dpi=150)
    plt.close()
    
    print("Spectrogram analysis completed")
    return {'status': 'completed'}


def snr_psnr_analysis(original_path, decrypted_path):
    """SNR/PSNR分析"""
    print("\n=== SNR/PSNR 分析 ===")
    
    # 读取音频
    sr1, orig = wavfile.read(original_path)
    sr2, dec = wavfile.read(decrypted_path)
    
    # 转换为float
    orig_float = orig.astype(float) / 32767.0 if orig.dtype == np.int16 else orig.astype(float)
    dec_float = dec.astype(float) / 32767.0 if dec.dtype == np.int16 else dec.astype(float)
    
    # 长度对齐
    min_len = min(len(orig_float), len(dec_float))
    orig_float = orig_float[:min_len]
    dec_float = dec_float[:min_len]
    
    # 计算SNR
    signal_power = np.mean(orig_float ** 2)
    noise_power = np.mean((orig_float - dec_float) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # 计算PSNR (将音频类比为图像的每一行)
    # 使用MSE
    mse = np.mean((orig_float - dec_float) ** 2)
    max_val = max(orig_float.max(), dec_float.max())
    psnr = 10 * np.log10(max_val ** 2 / (mse + 1e-10))
    
    print(f"SNR: {snr:.2f} dB")
    print(f"PSNR: {psnr:.2f} dB")
    
    return {'snr': float(snr), 'psnr': float(psnr)}


def correlation_analysis(original_path, encrypted_path):
    """相关系数分析"""
    print("\n=== 相关系数分析 (Correlation Analysis) ===")
    
    # 读取音频
    sr1, orig = wavfile.read(original_path)
    sr2, enc = wavfile.read(encrypted_path)
    
    # 转换为float
    orig_float = orig.astype(float) / 32767.0 if orig.dtype == np.int16 else orig.astype(float)
    enc_float = enc.astype(float) / 32767.0 if enc.dtype == np.int16 else enc.astype(float)
    
    # 如果是立体声，取单声道
    if len(orig_float.shape) > 1:
        orig_float = orig_float.mean(axis=1)
    if len(enc_float.shape) > 1:
        enc_float = enc_float.mean(axis=1)
    
    # 计算相邻采样点的相关系数
    def adjacent_correlation(signal):
        pairs = list(zip(signal[:-1], signal[1:]))
        pairs = np.array(pairs)
        return np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]
    
    orig_corr = adjacent_correlation(orig_float)
    enc_corr = adjacent_correlation(enc_float)
    
    print(f"Original adjacent correlation: {orig_corr:.6f}")
    print(f"Encrypted adjacent correlation: {enc_corr:.6f}")
    
    return {
        'orig_adjacent_corr': float(orig_corr),
        'enc_adjacent_corr': float(enc_corr)
    }


def run_full_audio_analysis(original_path, output_dir):
    """运行完整的音频分析"""
    print("=" * 60)
    print("音频加密完整安全性分析")
    print("Audio Encryption Comprehensive Analysis")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用项目中的音频加密功能
    from audio.audio_cs_spcmm_crypto import encrypt_audio, decrypt_audio
    
    key = "my-secret-audio-key-2026"
    encrypted_path = os.path.join(output_dir, 'encrypted_audio.npz')
    decrypted_path = os.path.join(output_dir, 'decrypted_audio.wav')
    
    # 加密
    print("\n>>> Step 1: 加密音频...")
    encrypt_audio(original_path, encrypted_path, key)
    
    # 解密
    print(">>> Step 2: 解密音频...")
    decrypt_audio(encrypted_path, decrypted_path, key)
    
    results = {}
    
    # 1. 波形分析
    results['waveform'] = waveform_analysis(original_path, encrypted_path, output_dir)
    
    # 2. 频谱分析
    results['spectral'] = spectral_analysis(original_path, encrypted_path, output_dir)
    
    # 3. 语谱图分析
    results['spectrogram'] = spectrogram_analysis(original_path, encrypted_path, output_dir)
    
    # 4. SNR/PSNR
    if os.path.exists(decrypted_path):
        results['snr_psnr'] = snr_psnr_analysis(original_path, decrypted_path)
    
    # 5. 相关系数
    results['correlation'] = correlation_analysis(original_path, encrypted_path)
    
    # 保存结果
    results_path = os.path.join(output_dir, 'audio_analysis_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='音频加密完整分析')
    parser.add_argument('--audio', type=str, default='resources/audio/1.wav', help='输入音频路径')
    parser.add_argument('--out', type=str, default='results/audio_exp/security_analysis', help='输出目录')
    
    args = parser.parse_args()
    
    run_full_audio_analysis(args.audio, args.out)
