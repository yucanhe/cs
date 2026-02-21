#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像加密参数优化实验
测试不同 CR 和 iter 组合
"""

import os
import sys
import re

def run_experiment(cr, iter_val, out_dir):
    """运行单个实验"""
    cmd = f'python image/demo/demo3.py --img resources/images/lena.png --key "my-secret-key-2026" --cr {cr} --lam 0.005 --iter {iter_val} --ablate no_float_diff --out {out_dir}'
    return os.system(cmd)

def parse_result(report_path):
    """解析结果"""
    if not os.path.exists(report_path):
        return {}
    
    with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    metrics = {}
    
    patterns = {
        'psnr': r'PSNR:\s*([\d.]+)',
        'ssim': r'SSIM:\s*([\d.]+)',
        'npcr': r'NPCR:\s*([\d.]+)%',
        'uaci': r'UACI:\s*([\d.]+)%',
        'entropy': r'Entropy:\s*([\d.]+)\s*bits',
        'encrypt_time': r'Encrypt time:\s*([\d.]+)s',
        'decrypt_time': r'Decrypt time:\s*([\d.]+)s',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except:
                pass
    
    return metrics

def main():
    print("=" * 70)
    print("图像加密参数优化实验")
    print("CR vs ITER 组合测试")
    print("=" * 70)
    
    # 测试组合
    configs = [
        # (cr, iter, name)
        (0.5, 150, "cr05_iter150"),
        (0.5, 200, "cr05_iter200"),
        (0.6, 150, "cr06_iter150"),
        (0.6, 180, "cr06_iter180"),
        (0.6, 200, "cr06_iter200"),
        (0.7, 150, "cr07_iter150"),
        (0.7, 180, "cr07_iter180"),
        (0.7, 200, "cr07_iter200"),
        (0.8, 150, "cr08_iter150"),
        (0.8, 200, "cr08_iter200"),
    ]
    
    results = []
    
    for i, (cr, iter_val, name) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing: CR={cr}, ITER={iter_val}")
        
        out_dir = f"results/optimize/{name}"
        os.makedirs(out_dir, exist_ok=True)
        
        run_experiment(cr, iter_val, out_dir)
        
        metrics = parse_result(os.path.join(out_dir, "report.txt"))
        metrics['cr'] = cr
        metrics['iter'] = iter_val
        metrics['name'] = name
        
        results.append(metrics)
        
        print(f"   PSNR: {metrics.get('psnr', 'N/A'):.2f} dB")
        print(f"   SSIM: {metrics.get('ssim', 'N/A'):.4f}")
        print(f"   Time: {metrics.get('encrypt_time', 'N/A'):.2f}s")
    
    # 保存结果
    import json
    with open("results/optimize/optimization_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印对比表
    print("\n" + "=" * 70)
    print("优化结果对比")
    print("=" * 70)
    print(f"\n{'配置':<18} {'CR':<6} {'ITER':<6} {'PSNR':<10} {'SSIM':<10} {'NPCR':<10} {'时间':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<18} {r['cr']:<6} {r.get('iter', 'N/A'):<6} {r.get('psnr', 'N/A'):<10.2f} {r.get('ssim', 'N/A'):<10.4f} {r.get('npcr', 'N/A'):<10.2f}% {r.get('encrypt_time', 'N/A'):<10.2f}s")
    
    # 找最优
    print("\n" + "=" * 70)
    print("最优配置 (按SSIM排序)")
    print("=" * 70)
    
    sorted_results = sorted(results, key=lambda x: x.get('ssim', 0), reverse=True)
    
    for r in sorted_results[:3]:
        print(f"  {r['name']}: SSIM={r.get('ssim', 'N/A'):.4f}, PSNR={r.get('psnr', 'N/A'):.2f}dB")

if __name__ == "__main__":
    main()
