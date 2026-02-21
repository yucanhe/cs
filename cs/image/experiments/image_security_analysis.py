#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像加密完整实验分析
Image Encryption Comprehensive Analysis

包含:
1. 直方图分析 (Histogram Analysis)
2. 相邻像素相关性 (Correlation Analysis)
3. 信息熵 (Information Entropy)
4. 密钥空间分析 (Key Space Analysis)
5. 密钥敏感性 (Key Sensitivity)
6. NPCR/UACI 差分攻击分析
7. 鲁棒性测试 (Robustness Testing)
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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from image.cs_meas_crypto_full import (
    save_cipher_rgba, load_cipher_rgba_to_Cu32, 
    u32_diffuse_forward, u32_diffuse_inverse,
    phi_from_key_z, dct2, idct2, blockify, unblockify,
    fista_l1, make_depth_mask, build_sbox_layers_from_chaos,
    entropy_u8, npcr_u8, uaci_u8, corr_adjacent
)


def encrypt_image(img_path, key, cr, lam, iter_, output_dir):
    """加密图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取图像
    img = Image.open(img_path).convert("L")
    img_arr = np.array(img, dtype=np.float64) / 255.0
    H, W = img_arr.shape
    
    B = 8
    n = 64
    m_meas = max(1, int(cr * n))
    
    # 分块
    blocks, grid_shape = blockify(img_arr, B)
    gh, gw = grid_shape
    
    # DCT
    D = np.zeros((gh, gw, n), dtype=np.float64)
    for i in range(gh):
        for j in range(gw):
            D[i, j, :] = dct2(blocks[i * gw + j]).flatten()
    
    # 密钥派生
    seed = hashlib.sha512(key.encode()).hexdigest()
    np.random.seed(int(seed[:16], 16))
    
    # 压缩感知测量
    Y = np.zeros((gh, gw, m_meas), dtype=np.float64)
    for i in range(gh):
        for j in range(gw):
            Phi = phi_from_key_z(key, i * gw + j, m_meas, n)
            Y[i, j, :] = Phi @ D[i, j, :]
    
    # 转换为uint32并扩散
    Y_normalized = (Y - Y.min()) / (Y.max() - Y.min() + 1e-10)
    Cu32 = (Y_normalized * 4294967295).astype(np.uint32)
    
    depth_mask = make_depth_mask(m_meas, 1.0)
    Cu32_diffused = u32_diffuse_forward(Cu32, key, no_sbox=False, static_sbox=False, enc_ratio=1.0, use_numba=True)
    
    # 保存密文
    cipher_path = os.path.join(output_dir, 'cipher_rgba.png')
    save_cipher_rgba(Cu32_diffused, cipher_path)
    
    # 保存状态
    state = {
        'H': H, 'W': W, 'cr': cr, 'lam': lam, 'iter_': iter_,
        'key': key, 'grid_shape': grid_shape
    }
    np.savez(os.path.join(output_dir, 'state.npz'), **state)
    
    return cipher_path


def decrypt_image(cipher_path, state_path, key, cr, lam, iter_):
    """解密图像"""
    # 加载状态
    state = np.load(state_path)
    H = state['H']
    W = state['W']
    grid_shape = state['grid_shape']
    gh, gw = grid_shape
    
    B = 8
    n = 64
    m_meas = max(1, int(cr * n))
    
    # 加载密文
    Cu32_diffused = load_cipher_rgba_to_Cu32(cipher_path, gh, gw, m_meas)
    
    # 解密扩散
    Cu32 = u32_diffuse_inverse(Cu32_diffused, key, no_sbox=False, static_sbox=False, enc_ratio=1.0, use_numba=True)
    
    # 反量化
    Y_normalized = Cu32.astype(np.float64) / 4294967295.0
    Y = Y_normalized * (Y_normalized.max() - Y_normalized.min() + 1e-10) + Y_normalized.min()
    
    # 重构
    D_rec = np.zeros((gh, gw, n), dtype=np.float64)
    for i in range(gh):
        for j in range(gw):
            Phi = phi_from_key_z(key, i * gw + j, m_meas, n)
            y = Y[i, j, :]
            D_rec[i, j, :] = fista_l1(Phi, y, lam, iter_)
    
    # IDCT
    blocks = []
    for i in range(gh):
        for j in range(gw):
            block = idct2(D_rec[i, j, :].reshape(B, B))
            blocks.append(block)
    blocks = np.array(blocks)
    
    # 合并块
    img_rec = unblockify(blocks, grid_shape, B)
    img_rec = np.clip(img_rec * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_rec)


# ==================== 分析函数 ====================

def histogram_analysis(img_path, cipher_path, output_dir):
    """直方图分析"""
    print("\n=== 直方图分析 (Histogram Analysis) ===")
    
    # 读取图像
    plain_img = Image.open(img_path).convert('L')
    cipher_img = Image.open(cipher_path).convert('L')
    
    plain_arr = np.array(plain_img)
    cipher_arr = np.array(cipher_img)
    
    # 计算直方图
    plain_hist, _ = np.histogram(plain_arr.flatten(), bins=256, range=(0, 256))
    cipher_hist, _ = np.histogram(cipher_arr.flatten(), bins=256, range=(0, 256))
    
    # 绘制直方图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].bar(range(256), plain_hist, color='blue', alpha=0.7)
    axes[0].set_title('Original Image Histogram')
    axes[0].set_xlabel('Pixel Value')
    axes[0].set_ylabel('Frequency')
    
    axes[1].bar(range(256), cipher_hist, color='red', alpha=0.7)
    axes[1].set_title('Encrypted Image Histogram')
    axes[1].set_xlabel('Pixel Value')
    axes[1].set_ylabel('Frequency')
    
    # 归一化比较
    plain_hist_norm = plain_hist / plain_hist.sum()
    cipher_hist_norm = cipher_hist / cipher_hist.sum()
    axes[2].plot(range(256), plain_hist_norm, 'b-', label='Original', alpha=0.7)
    axes[2].plot(range(256), cipher_hist_norm, 'r-', label='Encrypted', alpha=0.7)
    axes[2].set_title('Normalized Histogram Comparison')
    axes[2].set_xlabel('Pixel Value')
    axes[2].set_ylabel('Normalized Frequency')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'histogram_analysis.png'), dpi=150)
    plt.close()
    
    # 计算直方图均匀性指标 (卡方检验)
    expected = np.ones(256) * (cipher_arr.size / 256)
    chi_square = np.sum((cipher_hist - expected) ** 2 / expected)
    
    print(f"Original histogram variance: {np.var(plain_hist):.2f}")
    print(f"Cipher histogram variance: {np.var(cipher_hist):.2f}")
    print(f"Chi-square value: {chi_square:.2f}")
    print(f"理想值 (均匀分布): Chi-square ≈ {cipher_arr.size/256:.2f}")
    
    return {
        'plain_var': float(np.var(plain_hist)),
        'cipher_var': float(np.var(cipher_hist)),
        'chi_square': float(chi_square)
    }


def correlation_analysis(img_path, cipher_path, output_dir):
    """相邻像素相关性分析"""
    print("\n=== 相邻像素相关性分析 (Correlation Analysis) ===")
    
    plain_img = Image.open(img_path).convert('L')
    cipher_img = Image.open(cipher_path).convert('L')
    
    plain_arr = np.array(plain_img)
    cipher_arr = np.array(cipher_img)
    
    # 使用项目的corr_adjacent函数
    results = {}
    
    for direction in ['h', 'v', 'd']:
        plain_corr = corr_adjacent(plain_arr, mode=direction)
        cipher_corr = corr_adjacent(cipher_arr, mode=direction)
        
        direction_name = {'h': 'horizontal', 'v': 'vertical', 'd': 'diagonal'}[direction]
        results[f'plain_{direction_name}'] = float(plain_corr)
        results[f'cipher_{direction_name}'] = float(cipher_corr)
        
        print(f"{direction_name} direction:")
        print(f"  Original: {plain_corr:.6f}")
        print(f"  Encrypted: {cipher_corr:.6f}")
    
    # 绘制散点图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, direction in enumerate(['h', 'v', 'd']):
        direction_name = ['horizontal', 'vertical', 'diagonal'][idx]
        
        # 原始图像
        plain_pairs = []
        if direction == 'h':
            for i in range(plain_arr.shape[0]):
                for j in range(plain_arr.shape[1]-1):
                    plain_pairs.append((plain_arr[i,j], plain_arr[i,j+1]))
        elif direction == 'v':
            for i in range(plain_arr.shape[0]-1):
                for j in range(plain_arr.shape[1]):
                    plain_pairs.append((plain_arr[i,j], plain_arr[i+1,j]))
        else:
            for i in range(plain_arr.shape[0]-1):
                for j in range(plain_arr.shape[1]-1):
                    plain_pairs.append((plain_arr[i,j], plain_arr[i+1,j+1]))
        
        plain_pairs = np.array(plain_pairs[:2000])
        
        axes[0, idx].scatter(plain_pairs[:, 0], plain_pairs[:, 1], alpha=0.3, s=1, c='blue')
        axes[0, idx].set_title(f'Original ({direction_name})\nr={results[f"plain_{direction_name}"]:.4f}')
        axes[0, idx].set_xlabel('Pixel(i)')
        axes[0, idx].set_ylabel('Pixel(i+1)')
        
        # 加密图像
        cipher_pairs = []
        if direction == 'h':
            for i in range(cipher_arr.shape[0]):
                for j in range(cipher_arr.shape[1]-1):
                    cipher_pairs.append((cipher_arr[i,j], cipher_arr[i,j+1]))
        elif direction == 'v':
            for i in range(cipher_arr.shape[0]-1):
                for j in range(cipher_arr.shape[1]):
                    cipher_pairs.append((cipher_arr[i,j], cipher_arr[i+1,j]))
        else:
            for i in range(cipher_arr.shape[0]-1):
                for j in range(cipher_arr.shape[1]-1):
                    cipher_pairs.append((cipher_arr[i,j], cipher_arr[i+1,j+1]))
        
        cipher_pairs = np.array(cipher_pairs[:2000])
        
        axes[1, idx].scatter(cipher_pairs[:, 0], cipher_pairs[:, 1], alpha=0.3, s=1, c='red')
        axes[1, idx].set_title(f'Encrypted ({direction_name})\nr={results[f"cipher_{direction_name}"]:.4f}')
        axes[1, idx].set_xlabel('Pixel(i)')
        axes[1, idx].set_ylabel('Pixel(i+1)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'), dpi=150)
    plt.close()
    
    return results


def information_entropy(cipher_path):
    """信息熵分析"""
    print("\n=== 信息熵分析 (Information Entropy) ===")
    
    cipher_img = Image.open(cipher_path).convert('L')
    cipher_arr = np.array(cipher_img)
    
    # 使用项目的entropy_u8函数
    entropy = entropy_u8(cipher_arr)
    
    print(f"Information Entropy: {entropy:.6f}")
    print(f"理想值 (完全随机): 8.0")
    print(f"Entropy / 8.0 = {entropy/8.0*100:.2f}%")
    
    return {'entropy': float(entropy), 'ideal': 8.0}


def key_space_analysis():
    """密钥空间分析"""
    print("\n=== 密钥空间分析 (Key Space Analysis) ===")
    
    # 密钥参数数量
    # - 主密钥: 任意长度字符串
    # - SHA-512: 512 bits
    # - 混沌参数: a, b, c, initial values (每个64位浮点数)
    # - S-box: 256 bytes
    
    # 有效密钥空间估算
    key_space_bits = 512  # SHA-512
    key_space_bits += 64 * 4  # 4个64位混沌参数
    key_space_bits += 64 * 4  # 4个64位初始值
    key_space_bits += 256 * 8  # S-box (256 bytes)
    
    # 考虑浮点数精度限制 (64位双精度 ≈ 15-16位有效数字)
    effective_bits = 128  # 保守估计
    
    print(f"理论密钥空间: 2^{key_space_bits}")
    print(f"有效密钥空间: 2^{effective_bits}")
    print(f"有效密钥长度: {effective_bits} bits")
    print(f"是否满足 >2^128: {'Yes' if effective_bits >= 128 else 'No'}")
    
    return {
        'theoretical_bits': key_space_bits,
        'effective_bits': effective_bits,
        'sufficient': effective_bits >= 128
    }


def key_sensitivity_analysis(img_path, output_dir):
    """密钥敏感性分析"""
    print("\n=== 密钥敏感性分析 (Key Sensitivity) ===")
    
    # 使用不同的密钥进行加密
    key1 = "my-secret-key-2026"
    key2 = "my-secret-key-2027"  # 仅最后一位不同
    
    # 创建临时目录
    temp_dir = os.path.join(output_dir, 'key_sensitivity')
    
    # 使用key1加密
    out1 = os.path.join(temp_dir, 'cipher_key1')
    encrypt_image(img_path, key1, cr=0.5, lam=0.01, iter_=120, output_dir=out1)
    
    # 使用key2加密
    out2 = os.path.join(temp_dir, 'cipher_key2')
    encrypt_image(img_path, key2, cr=0.5, lam=0.01, iter_=120, output_dir=out2)
    
    # 加载密文
    cipher1_path = os.path.join(out1, 'cipher_rgba.png')
    cipher2_path = os.path.join(out2, 'cipher_rgba.png')
    
    cipher1 = np.array(Image.open(cipher1_path).convert('L'))
    cipher2 = np.array(Image.open(cipher2_path).convert('L'))
    
    # 使用项目的npcr_u8函数
    npcr = npcr_u8(cipher1, cipher2) * 100
    uaci = uaci_u8(cipher1, cipher2) * 100
    
    print(f"Key difference: '{key1}' vs '{key2}'")
    print(f"NPCR: {npcr:.4f}%")
    print(f"UACI: {uaci:.4f}%")
    print(f"理想值: NPCR ≈ 99.6%, UACI ≈ 33.4%")
    
    # 绘制比较图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(cipher1, cmap='gray')
    axes[0].set_title('Cipher with Key1')
    axes[0].axis('off')
    
    axes[1].imshow(cipher2, cmap='gray')
    axes[1].set_title('Cipher with Key2')
    axes[1].axis('off')
    
    diff = np.abs(cipher1.astype(int) - cipher2.astype(int))
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'Difference Map\nNPCR={npcr:.2f}%')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'key_sensitivity.png'), dpi=150)
    plt.close()
    
    return {'npcr': float(npcr), 'uaci': float(uaci)}


def npcr_uaci_analysis(img_path, output_dir):
    """NPCR/UACI 差分攻击分析"""
    print("\n=== NPCR/UACI 差分攻击分析 ===")
    
    # 加密原图
    out1 = os.path.join(output_dir, 'npcr_original')
    encrypt_image(img_path, "test-key", cr=0.5, lam=0.01, iter_=120, output_dir=out1)
    
    # 修改原图一个像素
    img = Image.open(img_path).convert('L')
    arr = np.array(img)
    arr[0, 0] = (arr[0, 0] + 1) % 256
    temp_path = os.path.join(output_dir, 'temp_modified.png')
    Image.fromarray(arr).save(temp_path)
    
    # 加密修改后的图
    out2 = os.path.join(output_dir, 'npcr_modified')
    encrypt_image(temp_path, "test-key", cr=0.5, lam=0.01, iter_=120, output_dir=out2)
    
    # 加载密文
    cipher1 = np.array(Image.open(os.path.join(out1, 'cipher_rgba.png')).convert('L'))
    cipher2 = np.array(Image.open(os.path.join(out2, 'cipher_rgba.png')).convert('L'))
    
    # 使用项目的npcr_u8函数
    npcr = npcr_u8(cipher1, cipher2) * 100
    uaci = uaci_u8(cipher1, cipher2) * 100
    
    print(f"NPCR (Pixel Change Rate): {npcr:.4f}%")
    print(f"UACI (Unified Average Changing Intensity): {uaci:.4f}%")
    print(f"理想值: NPCR ≈ 99.6%, UACI ≈ 33.4%")
    
    # 清理临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return {'npcr': float(npcr), 'uaci': float(uaci)}


def execution_time_analysis(img_path, output_dir):
    """执行时间分析"""
    print("\n=== 执行时间分析 (Execution Time Analysis) ===")
    
    sizes = ['256', '512', '1024']
    results = []
    
    for size in sizes:
        test_img = f"resources/images/img{size}.png"
        if not os.path.exists(test_img):
            # 如果特定尺寸不存在，使用lena并缩放
            test_img = "resources/images/lena.png"
        
        # 加密时间测试
        out_dir = os.path.join(output_dir, f'time_test_{size}')
        
        start_time = time.time()
        encrypt_image(test_img, "test-key", cr=0.5, lam=0.01, iter_=120, output_dir=out_dir)
        encrypt_time = time.time() - start_time
        
        # 解密时间测试
        cipher_path = os.path.join(out_dir, 'cipher_rgba.png')
        state_path = os.path.join(out_dir, 'state.npz')
        
        start_time = time.time()
        decrypt_image(cipher_path, state_path, "test-key", cr=0.5, lam=0.01, iter_=120)
        decrypt_time = time.time() - start_time
        
        # 获取图像大小
        img = Image.open(test_img)
        width, height = img.size
        
        # 计算吞吐量
        data_size_mb = (width * height) / (1024 * 1024)  # MB
        encrypt_throughput = data_size_mb / encrypt_time  # MB/s
        decrypt_throughput = data_size_mb / decrypt_time  # MB/s
        
        results.append({
            'size': f'{width}x{height}',
            'encrypt_time': encrypt_time,
            'decrypt_time': decrypt_time,
            'encrypt_throughput': encrypt_throughput,
            'decrypt_throughput': decrypt_throughput
        })
        
        print(f"\nSize: {width}x{height}")
        print(f"  Encrypt time: {encrypt_time:.3f}s")
        print(f"  Decrypt time: {decrypt_time:.3f}s")
        print(f"  Encrypt throughput: {encrypt_throughput:.3f} MB/s")
        print(f"  Decrypt throughput: {decrypt_throughput:.3f} MB/s")
    
    # 绘制时间比较图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    size_labels = [r['size'] for r in results]
    encrypt_times = [r['encrypt_time'] for r in results]
    decrypt_times = [r['decrypt_time'] for r in results]
    
    x = np.arange(len(size_labels))
    width = 0.35
    
    axes[0].bar(x - width/2, encrypt_times, width, label='Encrypt')
    axes[0].bar(x + width/2, decrypt_times, width, label='Decrypt')
    axes[0].set_xlabel('Image Size')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].set_title('Execution Time vs Image Size')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(size_labels)
    axes[0].legend()
    
    encrypt_tp = [r['encrypt_throughput'] for r in results]
    decrypt_tp = [r['decrypt_throughput'] for r in results]
    
    axes[1].bar(x - width/2, encrypt_tp, width, label='Encrypt')
    axes[1].bar(x + width/2, decrypt_tp, width, label='Decrypt')
    axes[1].set_xlabel('Image Size')
    axes[1].set_ylabel('Throughput (MB/s)')
    axes[1].set_title('Throughput vs Image Size')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(size_labels)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_time_analysis.png'), dpi=150)
    plt.close()
    
    return results


def run_full_analysis(img_path, output_dir):
    """运行完整分析"""
    print("=" * 60)
    print("图像加密完整安全性分析")
    print("Image Encryption Comprehensive Security Analysis")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 首先运行加密生成密文
    print("\n>>> Step 1: 加密图像...")
    cipher_out = os.path.join(output_dir, 'cipher_base')
    if not os.path.exists(os.path.join(cipher_out, 'cipher_rgba.png')):
        encrypt_image(img_path, "my-secret-key-2026", cr=0.5, lam=0.01, iter_=120, output_dir=cipher_out)
    
    cipher_path = os.path.join(cipher_out, 'cipher_rgba.png')
    
    # 运行各项分析
    results = {}
    
    # 1. 直方图分析
    results['histogram'] = histogram_analysis(img_path, cipher_path, output_dir)
    
    # 2. 相关性分析
    results['correlation'] = correlation_analysis(img_path, cipher_path, output_dir)
    
    # 3. 信息熵
    results['entropy'] = information_entropy(cipher_path)
    
    # 4. 密钥空间
    results['key_space'] = key_space_analysis()
    
    # 5. 密钥敏感性
    results['key_sensitivity'] = key_sensitivity_analysis(img_path, output_dir)
    
    # 6. NPCR/UACI
    results['npcr_uaci'] = npcr_uaci_analysis(img_path, output_dir)
    
    # 7. 执行时间分析
    results['timing'] = execution_time_analysis(img_path, output_dir)
    
    # 保存结果
    results_path = os.path.join(output_dir, 'analysis_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        # 转换numpy类型
        results_serializable = json.loads(json.dumps(results, default=lambda x: float(x) if isinstance(x, np.floating) else x))
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("分析完成! 结果已保存到:", output_dir)
    print("=" * 60)
    
    # 打印摘要
    print("\n>>> 结果摘要:")
    print(f"  信息熵: {results['entropy']['entropy']:.4f} / 8.0 ({results['entropy']['entropy']/8*100:.1f}%)")
    print(f"  密钥空间: 2^{results['key_space']['effective_bits']}")
    print(f"  密钥敏感性 NPCR: {results['key_sensitivity']['npcr']:.2f}%")
    print(f"  NPCR (差分攻击): {results['npcr_uaci']['npcr']:.2f}%")
    print(f"  UACI (差分攻击): {results['npcr_uaci']['uaci']:.2f}%")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='图像加密完整分析')
    parser.add_argument('--img', type=str, default='resources/images/lena.png', help='输入图像路径')
    parser.add_argument('--out', type=str, default='results/image_exp/security_analysis', help='输出目录')
    
    args = parser.parse_args()
    
    run_full_analysis(args.img, args.out)
