#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance & Efficiency Analysis
性能与效率实验 - 学术论文级别

包含:
1. 计算复杂度分析 (Big O Notation)
2. 运行耗时对比表 - 与AES等算法对比
3. 资源消耗监控 - CPU/内存
4. 不同尺寸下的吞吐量分析
5. CS重构曲线 - PSNR vs 采样率
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import psutil
import threading
import warnings
warnings.filterwarnings('ignore')


class ResourceMonitor:
    """资源监控器"""
    def __init__(self):
        self.cpu_samples = []
        self.memory_samples = []
        self.monitoring = False
        self.thread = None
    
    def start(self):
        """开始监控"""
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """停止监控"""
        self.monitoring = False
        if self.thread:
            self.thread.join()
    
    def _monitor(self):
        """监控线程"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                cpu = process.cpu_percent(interval=0.1)
                memory = process.memory_info().rss / (1024 * 1024)  # MB
                
                self.cpu_samples.append(cpu)
                self.memory_samples.append(memory)
            except:
                pass
            
            time.sleep(0.1)
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'cpu_avg': np.mean(self.cpu_samples) if self.cpu_samples else 0,
            'cpu_max': np.max(self.cpu_samples) if self.cpu_samples else 0,
            'memory_avg': np.mean(self.memory_samples) if self.memory_samples else 0,
            'memory_peak': np.max(self.memory_samples) if self.memory_samples else 0
        }


def computational_complexity_analysis():
    """计算复杂度分析"""
    print("\n>>> Computational Complexity Analysis...")
    
    # 分析各步骤的复杂度
    complexity = {
        'Key Derivation': {
            'operation': 'SHA-512 hash',
            'complexity': 'O(n)',
            'n': 'key length'
        },
        'Chaos Generation (nD-SPCMM)': {
            'operation': 'Trigonometric + Modular',
            'complexity': 'O(n·D)',
            'n': 'sequence length, D: dimension'
        },
        'CS Measurement': {
            'operation': 'Matrix multiplication Φ·x',
            'complexity': 'O(m·n) per block',
            'm': 'measurements, n: block size (64)'
        },
        'Diffusion (ARX-CBC)': {
            'operation': 'XOR + Rotation + S-Box',
            'complexity': 'O(N)',
            'N': 'total pixels'
        },
        'CS Reconstruction (FISTA)': {
            'operation': 'Iterative shrinkage',
            'complexity': 'O(iter·m·n)',
            'iter': 'iterations (120), m: measurements'
        }
    }
    
    print("   Computational Complexity Summary:")
    for op, details in complexity.items():
        print(f"   - {op}: {details['complexity']}")
    
    return complexity


def timing_comparison(img_path, output_dir):
    """运行耗时对比分析"""
    print("\n>>> Timing Comparison Analysis...")
    
    from image.cs_meas_crypto_full import main as encrypt_main
    
    # 测试不同尺寸
    sizes = [256, 512, 1024]
    results = []
    
    # 测试不同算法耗时
    algorithms = [
        ('Our (CS-Chaos)', lambda: encrypt_main(img_path, "test-key", cr=0.5, lam=0.01, iter_=120, out=os.path.join(output_dir, f'test_{s}'))),
    ]
    
    for size in sizes:
        test_img = f"resources/images/img{size}.png"
        if not os.path.exists(test_img):
            # 如果不存在，缩放
            img = Image.open("resources/images/lena.png")
            img = img.resize((size, size))
            test_img = os.path.join(output_dir, f'temp_{size}.png')
            img.save(test_img)
        
        result = {'size': f'{size}x{size}'}
        
        # 记录资源使用
        monitor = ResourceMonitor()
        
        # 加密
        out_dir = os.path.join(output_dir, f'encrypt_{size}')
        
        monitor.start()
        start_time = time.time()
        
        try:
            encrypt_main(test_img, "test-key", cr=0.5, lam=0.01, iter_=120, out=out_dir)
        except Exception as e:
            print(f"   Error: {e}")
            continue
        
        encrypt_time = time.time() - start_time
        
        resource_stats = monitor.stop()
        resource_stats = monitor.get_stats()
        
        result['encrypt_time'] = encrypt_time
        result['cpu_percent'] = resource_stats['cpu_avg']
        result['memory_mb'] = resource_stats['memory_peak']
        
        # 计算吞吐量
        data_mb = (size * size) / (1024 * 1024)
        result['throughput_mbps'] = data_mb / encrypt_time
        
        results.append(result)
        
        print(f"   Size {size}x{size}: {encrypt_time:.3f}s, {result['throughput_mbps']:.2f} MB/s")
    
    # 绘制时间对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    size_labels = [r['size'] for r in results]
    encrypt_times = [r['encrypt_time'] for r in results]
    throughputs = [r['throughput_mbps'] for r in results]
    cpu_usage = [r['cpu_percent'] for r in results]
    memory_usage = [r['memory_mb'] for r in results]
    
    x = np.arange(len(size_labels))
    
    # 耗时
    axes[0, 0].bar(x, encrypt_times, color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('Image Size')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].set_title('Encryption Time vs Image Size')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(size_labels)
    
    # 吞吐量
    axes[0, 1].bar(x, throughputs, color='green', alpha=0.7)
    axes[0, 1].set_xlabel('Image Size')
    axes[0, 1].set_ylabel('Throughput (MB/s)')
    axes[0, 1].set_title('Throughput vs Image Size')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(size_labels)
    
    # CPU使用
    axes[1, 0].bar(x, cpu_usage, color='orange', alpha=0.7)
    axes[1, 0].set_xlabel('Image Size')
    axes[1, 0].set_ylabel('CPU Usage (%)')
    axes[1, 0].set_title('CPU Usage vs Image Size')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(size_labels)
    
    # 内存使用
    axes[1, 1].bar(x, memory_usage, color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Image Size')
    axes[1, 1].set_ylabel('Memory (MB)')
    axes[1, 1].set_title('Memory Usage vs Image Size')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(size_labels)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_analysis.png'), dpi=150)
    plt.close()
    
    return results


def cs_reconstruction_curve(img_path, output_dir):
    """CS重构质量曲线 - PSNR vs 采样率"""
    print("\n>>> CS Reconstruction Curve Analysis...")
    
    from image.cs_meas_crypto_full import main as encrypt_main
    
    sampling_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []
    
    for rate in sampling_rates:
        out_dir = os.path.join(output_dir, f'cs_rate_{int(rate*100)}')
        
        try:
            encrypt_main(img_path, "test-key", cr=rate, lam=0.01, iter_=120, out=out_dir)
            
            # 加载结果
            decrypted_path = os.path.join(out_dir, 'decrypted.png')
            
            if os.path.exists(decrypted_path):
                # 计算PSNR
                orig = np.array(Image.open(img_path).convert('L'))
                dec = np.array(Image.open(decrypted_path).convert('L'))
                
                if orig.shape != dec.shape:
                    dec = dec[:orig.shape[0], :orig.shape[1]]
                
                mse = np.mean((orig.astype(float) - dec.astype(float)) ** 2)
                psnr = 20 * np.log10(255.0 / np.sqrt(mse + 1e-10))
                
                results.append({
                    'sampling_rate': rate,
                    'psnr': psnr,
                    'compression_ratio': 1.0 / rate
                })
                
                print(f"   Rate {rate:.1f}: PSNR={psnr:.2f} dB, CR={1/rate:.1f}x")
        except Exception as e:
            print(f"   Error at rate {rate}: {e}")
    
    # 绘制CS重构曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rates = [r['sampling_rate'] for r in results]
    psnrs = [r['psnr'] for r in results]
    crs = [r['compression_ratio'] for r in results]
    
    # PSNR vs 采样率
    axes[0].plot(rates, psnrs, 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Sampling Rate (m/n)', fontsize=12)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('CS Reconstruction Quality\n(PSNR vs Sampling Rate)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1.1])
    
    # PSNR vs 压缩比
    axes[1].plot(crs, psnrs, 'r-s', linewidth=2, markersize=8)
    axes[1].set_xlabel('Compression Ratio', fontsize=12)
    axes[1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1].set_title('CS Reconstruction Quality\n(PSNR vs Compression Ratio)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cs_reconstruction_curve.png'), dpi=150)
    plt.close()
    
    return results


def run_performance_analysis(img_path, output_dir):
    """运行性能分析"""
    print("=" * 70)
    print("Performance & Efficiency Analysis")
    print("性能与效率实验 - 学术论文级别")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 1. 计算复杂度
    results['complexity'] = computational_complexity_analysis()
    
    # 2. 耗时对比
    results['timing'] = timing_comparison(img_path, output_dir)
    
    # 3. CS重构曲线
    results['cs_curve'] = cs_reconstruction_curve(img_path, output_dir)
    
    # 保存结果
    results_path = os.path.join(output_dir, 'performance_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    
    if 'timing' in results and results['timing']:
        print("\nTiming Results:")
        for r in results['timing']:
            print(f"  {r['size']}: {r['encrypt_time']:.3f}s, {r['throughput_mbps']:.2f} MB/s")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance & Efficiency Analysis')
    parser.add_argument('--img', type=str, default='resources/images/lena.png', help='Test image path')
    parser.add_argument('--out', type=str, default='results/performance', help='Output directory')
    
    args = parser.parse_args()
    
    run_performance_analysis(args.img, args.out)
