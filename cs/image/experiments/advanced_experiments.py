#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Experiments - Chaos Analysis & Visual Security
高级实验 - 混沌分析与视觉安全

包含:
1. 相空间分析 (Phase Space Analysis)
   - 分叉图 (Bifurcation Diagram)
   - 李雅普诺夫指数 (Lyapunov Exponent)
   - 庞加莱截面 (Poincaré Section)
2. 视觉安全评价 (Visual Security Evaluation)
   - 边缘保留度 (Edge Preservation)
   - 视觉信息丢失 (Visual Information Loss)
3. 密钥流分析 (Keystream Analysis)
4. 密文图像质量评估 (Cipher Image Quality)
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# ==================== 混沌分析 ====================

def spcmm_iterate(x, a, b, c, theta=1.0):
    """nD-SPCMM 迭代"""
    x_new = (a * np.sin(b * x) + c) % theta
    return x_new


def bifurcation_diagram(output_dir):
    """分叉图分析"""
    print("\n>>> Bifurcation Diagram Analysis...")
    
    # 固定参数
    b = np.pi
    c = 0.1
    theta = 1.0
    
    # 变化参数 a
    a_values = np.linspace(0.5, 2.0, 500)
    
    # 迭代
    iterations = 100
    transient = 50
    
    x = 0.5
    results = {a: [] for a in a_values}
    
    for a in a_values:
        x = 0.5
        for i in range(transient):
            x = spcmm_iterate(x, a, b, c, theta)
        for i in range(iterations):
            x = spcmm_iterate(x, a, b, c, theta)
            results[a].append(x)
    
    # 绘制分叉图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for a in a_values:
        ax.plot([a] * len(results[a]), results[a], 'b.', markersize=0.1, alpha=0.5)
    
    ax.set_xlabel('Parameter a', fontsize=12)
    ax.set_ylabel('x', fontsize=12)
    ax.set_title('Bifurcation Diagram of SPCMM', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bifurcation_diagram.png'), dpi=150)
    plt.close()
    
    print("   Bifurcation diagram saved")
    
    return {'status': 'completed'}


def lyapunov_exponent(output_dir):
    """李雅普诺夫指数分析"""
    print("\n>>> Lyapunov Exponent Analysis...")
    
    # 参数
    b = np.pi
    c = 0.1
    theta = 1.0
    a_values = np.linspace(0.1, 2.0, 200)
    
    lyapunov_exponents = []
    
    for a in a_values:
        x = 0.5
        # 计算Lyapunov指数
        le = 0
        iterations = 500
        
        for i in range(iterations):
            # 导数: dx_new/dx = a*b*cos(b*x)
            derivative = abs(a * b * np.cos(b * x))
            if derivative > 0:
                le += np.log(derivative)
            x = spcmm_iterate(x, a, b, c, theta)
        
        le /= iterations
        lyapunov_exponents.append(le)
    
    # 绘制Lyapunov指数
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(a_values, lyapunov_exponents, 'b-', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.fill_between(a_values, lyapunov_exponents, 0, 
                    where=np.array(lyapunov_exponents) > 0, 
                    color='red', alpha=0.3, label='Chaotic (LE > 0)')
    ax.fill_between(a_values, lyapunov_exponents, 0, 
                    where=np.array(lyapunov_exponents) <= 0, 
                    color='green', alpha=0.3, label='Stable (LE < 0)')
    
    ax.set_xlabel('Parameter a', fontsize=12)
    ax.set_ylabel('Lyapunov Exponent', fontsize=12)
    ax.set_title('Lyapunov Exponent of SPCMM\n(Positive = Chaotic)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lyapunov_exponent.png'), dpi=150)
    plt.close()
    
    # 统计
    chaotic_ratio = sum(1 for le in lyapunov_exponents if le > 0) / len(lyapunov_exponents) * 100
    
    print(f"   Chaotic region ratio: {chaotic_ratio:.1f}%")
    print(f"   Max LE: {max(lyapunov_exponents):.4f}")
    print(f"   Min LE: {min(lyapunov_exponents):.4f}")
    
    return {
        'chaotic_ratio': chaotic_ratio,
        'max_le': float(max(lyapunov_exponents)),
        'min_le': float(min(lyapunov_exponents))
    }


def phase_portrait(output_dir):
    """相图分析"""
    print("\n>>> Phase Portrait Analysis...")
    
    # 3D-SPCMM 示例
    def spcmm_3d(x, y, z, a=1.2, b=1.5, c=0.1):
        x_new = (a * np.sin(b * x) + c) % 1.0
        y_new = (a * np.sin(b * y) + c) % 1.0
        z_new = (a * np.sin(b * z) + c) % 1.0
        return x_new, y_new, z_new
    
    # 生成轨迹
    n_points = 5000
    trajectory = np.zeros((n_points, 3))
    x, y, z = 0.1, 0.2, 0.3
    
    for i in range(n_points):
        x, y, z = spcmm_3d(x, y, z)
        trajectory[i] = [x, y, z]
    
    # 绘制3D相图
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    
    # XY平面
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b.', markersize=0.5, alpha=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Phase Portrait (XY)')
    ax1.grid(True, alpha=0.3)
    
    # XZ平面
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(trajectory[:, 0], trajectory[:, 2], 'r.', markersize=0.5, alpha=0.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')
    ax2.set_title('Phase Portrait (XZ)')
    ax2.grid(True, alpha=0.3)
    
    # YZ平面
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(trajectory[:, 1], trajectory[:, 2], 'g.', markersize=0.5, alpha=0.5)
    ax3.set_xlabel('y')
    ax3.set_ylabel('z')
    ax3.set_title('Phase Portrait (YZ)')
    ax3.grid(True, alpha=0.3)
    
    # 3D
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
             'b-', linewidth=0.1, alpha=0.5)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')
    ax4.set_title('3D Phase Portrait')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase_portrait.png'), dpi=150)
    plt.close()
    
    print("   Phase portrait saved")
    
    return {'status': 'completed'}


# ==================== 视觉安全分析 ====================

def edge_preservation(plain_path, cipher_path):
    """边缘保留度分析"""
    print("\n>>> Edge Preservation Analysis...")
    
    # 加载图像
    plain = np.array(Image.open(plain_path).convert('L'))
    cipher = np.array(Image.open(cipher_path).convert('L'))
    
    # Sobel边缘检测
    def sobel_edges(img):
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = sobel_x.T
        
        edges_x = signal.convolve2d(img, sobel_x, mode='same', boundary='symm')
        edges_y = signal.convolve2d(img, sobel_y, mode='same', boundary='symm')
        
        return np.sqrt(edges_x**2 + edges_y**2)
    
    plain_edges = sobel_edges(plain)
    cipher_edges = sobel_edges(cipher)
    
    # 计算边缘保留度 (EPR)
    plain_edge_strength = np.sum(plain_edges)
    cipher_edge_strength = np.sum(cipher_edges)
    
    epr = (cipher_edge_strength / plain_edge_strength) * 100
    
    # 计算边缘方向保留
    plain_angle = np.arctan2(signal.convolve2d(plain, np.array([[-1, 0, 1]]), mode='same'),
                             signal.convolve2d(plain, np.array([[-1], [0], [1]]), mode='same'))
    cipher_angle = np.arctan2(signal.convolve2d(cipher, np.array([[-1, 0, 1]]), mode='same'),
                               signal.convolve2d(cipher, np.array([[-1], [0], [1]]), mode='same'))
    
    # 计算角度差异
    angle_diff = np.abs(plain_angle - cipher_angle)
    angle_preservation = (1 - np.mean(angle_diff) / np.pi) * 100
    
    print(f"   Edge Preservation Ratio (EPR): {epr:.2f}%")
    print(f"   Edge Direction Preservation: {angle_preservation:.2f}%")
    
    return {
        'epr': float(epr),
        'angle_preservation': float(angle_preservation)
    }


def visual_information_loss(plain_path, cipher_path):
    """视觉信息丢失分析"""
    print("\n>>> Visual Information Loss (VIF) Analysis...")
    
    # 加载图像
    plain = np.array(Image.open(plain_path).convert('L')).astype(float)
    cipher = np.array(Image.open(cipher_path).convert('L')).astype(float)
    
    # 简化版VIF计算
    # 使用小波分解
    import pywt
    
    # 小波分解
    coeffs_plain = pywt.wavedec2(plain, 'haar', level=2)
    coeffs_cipher = pywt.wavedec2(cipher, 'haar', level=2)
    
    # 计算各层能量
    def wavelet_energy(coeffs):
        energy = 0
        for c in coeffs:
            if isinstance(c, tuple):
                for cc in c:
                    energy += np.sum(cc**2)
            else:
                energy += np.sum(c**2)
        return energy
    
    plain_energy = wavelet_energy(coeffs_plain)
    cipher_energy = wavelet_energy(coeffs_cipher)
    
    # 视觉信息丢失
    vif_loss = (1 - cipher_energy / (plain_energy + 1e-10)) * 100
    
    print(f"   Visual Information Loss: {vif_loss:.2f}%")
    
    return {'vif_loss': float(vif_loss)}


def ciphertext_quality_analysis(cipher_path, output_dir):
    """密文图像质量分析"""
    print("\n>>> Ciphertext Quality Analysis...")
    
    cipher = np.array(Image.open(cipher_path).convert('L'))
    
    results = {}
    
    # 1. 亮度分析
    brightness = np.mean(cipher)
    results['brightness'] = float(brightness)
    print(f"   Brightness: {brightness:.2f}")
    
    # 2. 对比度分析
    contrast = np.std(cipher)
    results['contrast'] = float(contrast)
    print(f"   Contrast (std): {contrast:.2f}")
    
    # 3. 均匀性分析
    hist, _ = np.histogram(cipher.flatten(), bins=256, range=(0, 256))
    hist_norm = hist / hist.sum()
    uniformity = 1 - np.std(hist_norm) / np.mean(hist_norm)
    results['uniformity'] = float(uniformity)
    print(f"   Uniformity: {uniformity:.4f}")
    
    # 绘制分析图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始密文
    axes[0, 0].imshow(cipher, cmap='gray')
    axes[0, 0].set_title('Encrypted Image')
    axes[0, 0].axis('off')
    
    # 亮度分布
    axes[0, 1].hist(cipher.flatten(), bins=64, color='gray', alpha=0.7)
    axes[0, 1].axvline(brightness, color='r', linestyle='--', label=f'Mean: {brightness:.1f}')
    axes[0, 1].set_title('Brightness Distribution')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # 梯度图
    gradient = np.gradient(cipher.astype(float))
    grad_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)
    axes[1, 0].imshow(grad_magnitude, cmap='hot')
    axes[1, 0].set_title('Gradient Magnitude')
    axes[1, 0].axis('off')
    
    # 局部方差图
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(cipher.astype(float), size=8)
    local_var = uniform_filter(cipher.astype(float)**2, size=8) - local_mean**2
    axes[1, 1].imshow(local_var, cmap='viridis')
    axes[1, 1].set_title('Local Variance')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cipher_quality.png'), dpi=150)
    plt.close()
    
    return results


def run_advanced_experiments(plain_path, cipher_path, output_dir):
    """运行高级实验"""
    print("=" * 70)
    print("Advanced Experiments")
    print("高级实验 - 混沌分析与视觉安全")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 1. 分叉图
    results['bifurcation'] = bifurcation_diagram(output_dir)
    
    # 2. Lyapunov指数
    results['lyapunov'] = lyapunov_exponent(output_dir)
    
    # 3. 相图
    results['phase_portrait'] = phase_portrait(output_dir)
    
    # 4. 边缘保留度
    results['edge_preservation'] = edge_preservation(plain_path, cipher_path)
    
    # 5. 视觉信息丢失
    try:
        results['visual_info_loss'] = visual_information_loss(plain_path, cipher_path)
    except:
        print("   (pywt not available, skipping VIF)")
    
    # 6. 密文质量
    results['cipher_quality'] = ciphertext_quality_analysis(cipher_path, output_dir)
    
    # 保存结果
    results_path = os.path.join(output_dir, 'advanced_experiments_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("Advanced Experiments Summary")
    print("=" * 70)
    
    if 'lyapunov' in results:
        print(f"  Chaos Analysis:")
        print(f"    Chaotic region: {results['lyapunov']['chaotic_ratio']:.1f}%")
        print(f"    Max LE: {results['lyapunov']['max_le']:.4f}")
    
    if 'edge_preservation' in results:
        print(f"  Visual Security:")
        print(f"    EPR: {results['edge_preservation']['epr']:.2f}%")
    
    if 'cipher_quality' in results:
        print(f"  Cipher Quality:")
        print(f"    Brightness: {results['cipher_quality']['brightness']:.2f}")
        print(f"    Uniformity: {results['cipher_quality']['uniformity']:.4f}")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced Experiments')
    parser.add_argument('--plain', type=str, required=True, help='Original image path')
    parser.add_argument('--cipher', type=str, required=True, help='Encrypted image path')
    parser.add_argument('--out', type=str, default='results/advanced', help='Output directory')
    
    args = parser.parse_args()
    
    run_advanced_experiments(args.plain, args.cipher, args.out)
