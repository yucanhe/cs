#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Statistical Security Analysis for Chaos-Based Encryption
统计安全性实验 - 学术论文级别

包含:
1. 直方图分析 (Histogram Analysis) + Chi-Square检验
2. 相邻像素相关性 (Correlation Analysis) + 统计显著性检验
3. 信息熵 (Information Entropy) + NIST标准
4. 像素分布检验 (Distribution Tests) - KS检验
5. 频谱分析 (Spectral Analysis) - 功率谱密度
6. 游程检验 (Runs Test) - 随机性检验
7. 图像质量指标 (Image Quality Metrics)
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, signal
from scipy.fft import fft2, fftshift
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


def chi_square_test(data, bins=256):
    """卡方检验 - 评估密文均匀性"""
    hist, _ = np.histogram(data.flatten(), bins=bins, range=(0, 256))
    expected = np.ones(bins) * (data.size / bins)
    chi2, p_value = stats.chisquare(hist, expected)
    return chi2, p_value


def kolmogorov_smirnov_test(data):
    """KS检验 - 评估密文分布"""
    # 转换为0-1范围
    data_norm = data.flatten() / 255.0
    # 与均匀分布比较
    ks_stat, p_value = stats.kstest(data_norm, 'uniform')
    return ks_stat, p_value


def runs_test(data):
    """游程检验 - 评估随机性"""
    # 转换为二进制序列
    median = np.median(data)
    binary = (data > median).astype(int).flatten()
    
    runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i-1]:
            runs += 1
    
    n0 = np.sum(binary == 0)
    n1 = np.sum(binary == 1)
    n = n0 + n1
    
    # 计算期望值和方差
    expected_runs = (2 * n0 * n1) / n + 1
    var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n ** 2 * (n - 1))
    
    if var_runs > 0:
        z = (runs - expected_runs) / np.sqrt(var_runs)
    else:
        z = 0
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return runs, z, p_value


def spectral_analysis(data):
    """频谱分析 - 功率谱密度"""
    # 2D FFT
    f = fft2(data.astype(float))
    fshift = fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    
    # 计算径向平均功率谱
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2
    
    # 创建径向坐标
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r = r.astype(int)
    
    # 计算每个半径的平均功率
    r_max = int(np.max(r)) + 1
    radial_profile = np.zeros(r_max)
    counts = np.zeros(r_max)
    
    for i in range(h):
        for j in range(w):
            radial_profile[r[i, j]] += magnitude_spectrum[i, j]
            counts[r[i, j]] += 1
    
    # 避免除零
    counts[counts > 0] = radial_profile[counts > 0] / counts[counts > 0]
    
    return radial_profile, magnitude_spectrum


def correlation_significance_test(data, n_samples=10000):
    """相关性统计显著性检验"""
    h, w = data.shape
    
    # 水平方向
    h_pairs = []
    for _ in range(min(n_samples, h * (w-1))):
        i = np.random.randint(0, h)
        j = np.random.randint(0, w-1)
        h_pairs.append((data[i, j], data[i, j+1]))
    
    h_pairs = np.array(h_pairs)
    corr_h = np.corrcoef(h_pairs[:, 0], h_pairs[:, 1])[0, 1]
    
    # 垂直方向
    v_pairs = []
    for _ in range(min(n_samples, (h-1) * w)):
        i = np.random.randint(0, h-1)
        j = np.random.randint(0, w)
        v_pairs.append((data[i, j], data[i+1, j]))
    
    v_pairs = np.array(v_pairs)
    corr_v = np.corrcoef(v_pairs[:, 0], v_pairs[:, 1])[0, 1]
    
    # 计算置信区间 (95%)
    n = len(h_pairs)
    se_h = 1 / np.sqrt(n)
    se_v = 1 / np.sqrt(n)
    
    ci_h = (corr_h - 1.96*se_h, corr_h + 1.96*se_h)
    ci_v = (corr_v - 1.96*se_v, corr_v + 1.96*se_v)
    
    return {
        'horizontal': {'correlation': corr_h, 'ci_95': ci_h},
        'vertical': {'correlation': corr_v, 'ci_95': ci_v}
    }


def nist_entropy_test(data):
    """NIST风格熵测试"""
    # 计算字节频率
    byte_counts = np.bincount(data.flatten().astype(np.uint8), minlength=256)
    probabilities = byte_counts / data.size
    
    # 计算熵
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    # 计算理想熵
    ideal_entropy = 8.0
    
    # 计算偏离度
    entropy_deviation = abs(entropy - ideal_entropy) / ideal_entropy * 100
    
    # NIST建议: 熵应该 > 7.95 (99.4% of ideal)
    nist_pass = entropy >= 7.95
    
    return {
        'entropy': entropy,
        'ideal': ideal_entropy,
        'deviation_percent': entropy_deviation,
        'nist_pass': nist_pass
    }


def histogram_uniformity_test(data):
    """直方图均匀性测试"""
    hist, _ = np.histogram(data.flatten(), bins=256, range=(0, 256))
    hist_norm = hist / hist.sum()
    
    # 计算均匀性指标
    # 1. 标准差
    std_dev = np.std(hist_norm)
    
    # 2. 峰度 (应该接近0对于均匀分布)
    kurtosis = stats.kurtosis(hist_norm)
    
    # 3. 偏度 (应该接近0对于均匀分布)
    skewness = stats.skew(hist_norm)
    
    # 4. 卡方检验
    chi2, p_value = chi_square_test(data)
    
    # 5. KS检验
    ks_stat, ks_p = kolmogorov_smirnov_test(data)
    
    return {
        'std_dev': float(std_dev),
        'kurtosis': float(kurtosis),
        'skewness': float(skewness),
        'chi_square': float(chi2),
        'chi_p_value': float(p_value),
        'ks_statistic': float(ks_stat),
        'ks_p_value': float(ks_p),
        'uniform': p_value > 0.01  # 假设检验
    }


def image_quality_metrics(original, decrypted):
    """图像质量指标"""
    orig = original.astype(float)
    dec = decrypted.astype(float)
    
    # MSE
    mse = np.mean((orig - dec) ** 2)
    
    # PSNR
    if mse > 0:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # SSIM (简化版)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = np.mean(orig)
    mu2 = np.mean(dec)
    sigma1 = np.var(orig)
    sigma2 = np.var(dec)
    sigma12 = np.mean((orig - mu1) * (dec - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
    
    # MAE
    mae = np.mean(np.abs(orig - dec))
    
    return {
        'mse': float(mse),
        'psnr': float(psnr),
        'ssim': float(ssim),
        'mae': float(mae)
    }


def run_comprehensive_statistical_analysis(img_path, cipher_path, output_dir):
    """运行完整的统计安全性分析"""
    print("=" * 70)
    print("Comprehensive Statistical Security Analysis")
    print("统计安全性实验 - 学术论文级别")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    orig_img = Image.open(img_path).convert('L')
    cipher_img = Image.open(cipher_path).convert('L')
    
    orig_arr = np.array(orig_img)
    cipher_arr = np.array(cipher_img)
    
    results = {}
    
    # 1. 直方图分析
    print("\n>>> 1. Histogram Analysis...")
    hist_result = histogram_uniformity_test(cipher_arr)
    results['histogram'] = hist_result
    
    # 绘制直方图对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 原始直方图
    orig_hist, _ = np.histogram(orig_arr.flatten(), bins=256, range=(0, 256))
    axes[0, 0].bar(range(256), orig_hist, color='blue', alpha=0.7, width=1)
    axes[0, 0].set_title('Original Image Histogram')
    axes[0, 0].set_xlabel('Pixel Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # 密文直方图
    cipher_hist, _ = np.histogram(cipher_arr.flatten(), bins=256, range=(0, 256))
    axes[0, 1].bar(range(256), cipher_hist, color='red', alpha=0.7, width=1)
    axes[0, 1].set_title(f'Encrypted Image Histogram\n(Chi²={hist_result["chi_square"]:.2f}, p={hist_result["chi_p_value"]:.4f})')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # 归一化对比
    orig_hist_n = orig_hist / orig_hist.sum()
    cipher_hist_n = cipher_hist / cipher_hist.sum()
    axes[1, 0].plot(range(256), orig_hist_n, 'b-', label='Original', alpha=0.7)
    axes[1, 0].plot(range(256), cipher_hist_n, 'r-', label='Encrypted', alpha=0.7)
    axes[1, 0].set_title('Normalized Histogram Comparison')
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Normalized Frequency')
    axes[1, 0].legend()
    
    # 理想均匀分布
    ideal = np.ones(256) / 256
    axes[1, 1].bar(range(256), cipher_hist_n, color='red', alpha=0.5, width=1, label='Encrypted')
    axes[1, 1].plot(range(256), ideal, 'k--', linewidth=2, label='Ideal Uniform')
    axes[1, 1].set_title(f'Encrypted vs Ideal Uniform\n(Kurtosis={hist_result["kurtosis"]:.4f}, Skew={hist_result["skewness"]:.4f})')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'histogram_analysis.png'), dpi=150)
    plt.close()
    
    print(f"   Chi-Square: {hist_result['chi_square']:.2f}")
    print(f"   Kurtosis: {hist_result['kurtosis']:.4f} (ideal: 0)")
    print(f"   Skewness: {hist_result['skewness']:.4f} (ideal: 0)")
    
    # 2. 相关性分析
    print("\n>>> 2. Correlation Analysis...")
    corr_result = correlation_significance_test(cipher_arr)
    results['correlation'] = corr_result
    
    # 绘制相关性散点图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (direction, pairs_func) in enumerate([
        ('Horizontal', lambda: [(orig_arr[i,j], orig_arr[i,j+1]) for i in range(orig_arr.shape[0]) for j in range(orig_arr.shape[1]-1)],
        ('Vertical', lambda: [(orig_arr[i,j], orig_arr[i+1,j]) for i in range(orig_arr.shape[0]-1) for j in range(orig_arr.shape[1])],
        ('Diagonal', lambda: [(orig_arr[i,j], orig_arr[i+1,j+1]) for i in range(orig_arr.shape[0]-1) for j in range(orig_arr.shape[1]-1)])
    ]):
        orig_pairs = np.array(pairs_func()[:2000])
        cipher_pairs = np.array([(cipher_arr[i,j], cipher_arr[i,j+1]) for i in range(cipher_arr.shape[0]) for j in range(cipher_arr.shape[1]-1)][:2000])
        
        axes[0, idx].scatter(orig_pairs[:,0], orig_pairs[:,1], alpha=0.3, s=2, c='blue')
        axes[0, idx].set_title(f'Original {direction}\nr={corr_result.get(direction.lower()[:3], {"correlation":0})["correlation"]:.4f}')
        axes[0, idx].set_xlabel('Pixel(i)')
        axes[0, idx].set_ylabel('Pixel(i+1)')
        
        axes[1, idx].scatter(cipher_pairs[:,0], cipher_pairs[:,1], alpha=0.3, s=2, c='red')
        dir_key = {'Horizontal': 'horizontal', 'Vertical': 'vertical', 'Diagonal': 'diagonal'}[direction]
        axes[1, idx].set_title(f'Encrypted {direction}\nr={corr_result[dir_key]["correlation"]:.4f}')
        axes[1, idx].set_xlabel('Pixel(i)')
        axes[1, idx].set_ylabel('Pixel(i+1)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'), dpi=150)
    plt.close()
    
    print(f"   Horizontal: {corr_result['horizontal']['correlation']:.6f} (CI: {corr_result['horizontal']['ci_95']})")
    print(f"   Vertical: {corr_result['vertical']['correlation']:.6f} (CI: {corr_result['vertical']['ci_95']})")
    
    # 3. 信息熵分析
    print("\n>>> 3. Information Entropy Analysis...")
    entropy_result = nist_entropy_test(cipher_arr)
    results['entropy'] = entropy_result
    
    print(f"   Entropy: {entropy_result['entropy']:.6f} (ideal: 8.0)")
    print(f"   Deviation: {entropy_result['deviation_percent']:.2f}%")
    print(f"   NIST Pass: {entropy_result['nist_pass']}")
    
    # 4. 游程检验
    print("\n>>> 4. Runs Test...")
    runs, z, p_value = runs_test(cipher_arr)
    results['runs_test'] = {'runs': runs, 'z_score': z, 'p_value': p_value}
    
    print(f"   Runs: {runs}")
    print(f"   Z-Score: {z:.4f}")
    print(f"   P-Value: {p_value:.4f}")
    print(f"   Random: {'Yes' if p_value > 0.05 else 'No'}")
    
    # 5. 频谱分析
    print("\n>>> 5. Spectral Analysis...")
    radial_profile, magnitude_spectrum = spectral_analysis(cipher_arr)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 功率谱密度
    axes[0].semilogy(range(len(radial_profile)), radial_profile + 1e-10, 'r-', linewidth=2)
    axes[0].set_title('Radial Power Spectrum')
    axes[0].set_xlabel('Radial Frequency')
    axes[0].set_ylabel('Power Spectral Density (log)')
    axes[0].grid(True, alpha=0.3)
    
    # 2D频谱
    im = axes[1].imshow(np.log10(magnitude_spectrum + 1e-10), cmap='hot', aspect='equal')
    axes[1].set_title('2D Fourier Spectrum (log scale)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectral_analysis.png'), dpi=150)
    plt.close()
    
    # 6. KS检验
    print("\n>>> 6. Kolmogorov-Smirnov Test...")
    ks_stat, ks_p = kolmogorov_smirnov_test(cipher_arr)
    results['ks_test'] = {'statistic': ks_stat, 'p_value': ks_p}
    
    print(f"   KS Statistic: {ks_stat:.6f}")
    print(f"   P-Value: {ks_p:.6f}")
    print(f"   Uniform: {'Yes' if ks_p > 0.05 else 'No'}")
    
    # 保存结果
    results_path = os.path.join(output_dir, 'statistical_security_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        # 转换numpy类型
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert, ensure_ascii=False)
    
    # 打印最终摘要
    print("\n" + "=" * 70)
    print("Statistical Security Summary")
    print("=" * 70)
    print(f"  Information Entropy: {entropy_result['entropy']:.4f}/8.0 ({entropy_result['entropy']/8*100:.1f}%)")
    print(f"  Horizontal Correlation: {corr_result['horizontal']['correlation']:.6f}")
    print(f"  Vertical Correlation: {corr_result['vertical']['correlation']:.6f}")
    print(f"  Chi-Square Test: {hist_result['chi_square']:.2f} (p={hist_result['chi_p_value']:.4f})")
    print(f"  Runs Test: Z={z:.4f}, Random={'Yes' if p_value > 0.05 else 'No'}")
    print(f"  KS Test: {ks_stat:.4f}, Uniform={'Yes' if ks_p > 0.05 else 'No'}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive Statistical Security Analysis')
    parser.add_argument('--img', type=str, required=True, help='Original image path')
    parser.add_argument('--cipher', type=str, required=True, help='Encrypted image path')
    parser.add_argument('--out', type=str, default='results/statistical_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    run_comprehensive_statistical_analysis(args.img, args.cipher, args.out)
