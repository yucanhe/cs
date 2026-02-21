#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Experiments
消融实验 - 测试每个组件的贡献

组件:
1. baseline_cs: 只用CS测量 (无扩散)
2. no_float_diff: 无3D浮点扩散
3. no_u32_arx: 无U32 ARX-CBC扩散
4. no_sbox: 无S-Box
5. static_sbox: 静态S-Box
6. full: 完整系统 (基准)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd


def run_ablation_experiment(img_path, key, cr, lam, iter_, out_dir, ablation_mode):
    """运行单组消融实验"""
    import subprocess
    
    cmd = [
        sys.executable, "image/demo/demo3.py",
        "--img", img_path,
        "--key", key,
        "--cr", str(cr),
        "--lam", str(lam),
        "--iter", str(iter_),
        "--out", out_dir,
        "--ablate", ablation_mode
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return result.stdout + result.stderr


def parse_report(filepath):
    """从report.txt解析结果"""
    if not os.path.exists(filepath):
        return {}
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    import re
    metrics = {}
    
    patterns = {
        'psnr': r'PSNR[\s:]+([\d.]+)\s*dB',
        'ssim': r'SSIM[\s:]+([\d.]+)',
        'npcr': r'NPCR[\s:]+([\d.]+)%',
        'uaci': r'UACI[\s:]+([\d.]+)%',
        'entropy': r'Entropy[\s:]+([\d.]+)',
        'encrypt_time': r'Encrypt time:[\s]+([\d.]+)',
        'decrypt_time': r'Decrypt time:[\s]+([\d.]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except:
                pass
    
    # 解析相关性
    corr_patterns = {
        'corr_h': r'AdjCorr-H:[\s]+([-\d.]+)',
        'corr_v': r'AdjCorr-V:[\s]+([-\d.]+)',
        'corr_d': r'AdjCorr-D:[\s]+([-\d.]+)',
    }
    
    for key, pattern in corr_patterns.items():
        match = re.search(pattern, content)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except:
                pass
    
    return metrics


def run_ablation_study():
    """运行消融研究"""
    print("=" * 70)
    print("Ablation Study")
    print("消融实验 - 测试各组件贡献")
    print("=" * 70)
    
    img_path = "resources/images/lena.png"
    key = "my-secret-key-2026"
    cr = 0.5
    lam = 0.005
    iter_ = 200
    
    # 消融模式
    ablation_modes = [
        ('full', '完整系统 (基准)'),
        ('baseline_cs', '仅CS测量 (无扩散)'),
        ('no_float_diff', '无3D浮点扩散'),
        ('no_u32_arx', '无U32 ARX-CBC'),
        ('no_sbox', '无S-Box'),
        ('static_sbox', '静态S-Box'),
    ]
    
    results = []
    
    for mode, desc in ablation_modes:
        print(f"\n{'='*70}")
        print(f"Testing: {mode} - {desc}")
        print("=" * 70)
        
        out_dir = f"results/ablation/{mode}"
        os.makedirs(out_dir, exist_ok=True)
        
        # 运行实验
        start_time = time.time()
        output = run_ablation_experiment(img_path, key, cr, lam, iter_, out_dir, mode)
        total_time = time.time() - start_time
        
        # 解析结果
        report_path = os.path.join(out_dir, 'report.txt')
        metrics = parse_report(report_path)
        
        metrics.update({
            'mode': mode,
            'description': desc,
            'total_time': total_time
        })
        
        results.append(metrics)
        
        print(f"\nResults:")
        print(f"  PSNR: {metrics.get('psnr', 'N/A')} dB")
        print(f"  SSIM: {metrics.get('ssim', 'N/A')}")
        print(f"  NPCR: {metrics.get('npcr', 'N/A')}%")
        print(f"  UACI: {metrics.get('uaci', 'N/A')}%")
        print(f"  Entropy: {metrics.get('entropy', 'N/A')}")
        print(f"  Encrypt Time: {metrics.get('encrypt_time', 'N/A')}s")
        print(f"  Total Time: {total_time:.2f}s")
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv('results/ablation/ablation_results.csv', index=False)
    
    # 打印对比表
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    
    print(f"\n{'Mode':<20} {'PSNR':>8} {'SSIM':>8} {'NPCR':>8} {'UACI':>8} {'Entropy':>8} {'Time':>8}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        print(f"{row['mode']:<20} {row.get('psnr', 0):>8.2f} {row.get('ssim', 0):>8.4f} "
              f"{row.get('npcr', 0):>8.2f} {row.get('uaci', 0):>8.2f} "
              f"{row.get('entropy', 0):>8.4f} {row.get('total_time', 0):>8.2f}")
    
    # 分析各组件贡献
    print("\n" + "=" * 70)
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 70)
    
    full_row = df[df['mode'] == 'full'].iloc[0]
    
    print(f"\n基准 (full) 指标:")
    print(f"  PSNR: {full_row.get('psnr', 0):.2f} dB")
    print(f"  SSIM: {full_row.get('ssim', 0):.4f}")
    print(f"  NPCR: {full_row.get('npcr', 0):.2f}%")
    print(f"  Entropy: {full_row.get('entropy', 0):.4f}")
    
    for mode, desc in ablation_modes[1:]:  # 跳过full
        row = df[df['mode'] == mode].iloc[0]
        
        print(f"\n{desc} ({mode}):")
        
        # PSNR变化
        psnr_diff = row.get('psnr', 0) - full_row.get('psnr', 0)
        print(f"  PSNR变化: {psnr_diff:+.2f} dB")
        
        # SSIM变化
        ssim_diff = row.get('ssim', 0) - full_row.get('ssim', 0)
        print(f"  SSIM变化: {ssim_diff:+.4f}")
        
        # NPCR变化 (安全性)
        npcr_diff = row.get('npcr', 0) - full_row.get('npcr', 0)
        print(f"  NPCR变化: {npcr_diff:+.2f}%")
        
        # 熵变化
        ent_diff = row.get('entropy', 0) - full_row.get('entropy', 0)
        print(f"  熵变化: {ent_diff:+.4f}")
    
    # 保存JSON报告
    report = {
        'baseline': {
            'psnr': float(full_row.get('psnr', 0)),
            'ssim': float(full_row.get('ssim', 0)),
            'npcr': float(full_row.get('npcr', 0)),
            'uaci': float(full_row.get('uaci', 0)),
            'entropy': float(full_row.get('entropy', 0)),
        },
        'ablations': []
    }
    
    for _, row in df.iterrows():
        if row['mode'] != 'full':
            report['ablations'].append({
                'mode': row['mode'],
                'description': row['description'],
                'psnr_change': float(row.get('psnr', 0) - full_row.get('psnr', 0)),
                'ssim_change': float(row.get('ssim', 0) - full_row.get('ssim', 0)),
                'npcr_change': float(row.get('npcr', 0) - full_row.get('npcr', 0)),
                'entropy_change': float(row.get('entropy', 0) - full_row.get('entropy', 0)),
            })
    
    with open('results/ablation/ablation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Results saved to: results/ablation/")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    os.makedirs("results/ablation", exist_ok=True)
    run_ablation_study()
