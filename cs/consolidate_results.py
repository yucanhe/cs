#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Results Organizer
整合所有实验结果 - 生成统一报告

功能:
1. 收集所有实验数据
2. 生成综合报告 (JSON + HTML Dashboard)
3. 统计指标汇总表
4. 可视化对比图
"""

import os
import json
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def parse_report_file(filepath):
    """解析report.txt文件"""
    results = {}
    
    if not os.path.exists(filepath):
        return results
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 提取关键指标
    patterns = {
        'psnr': r'PSNR[\s:]+([\d.]+)\s*dB',
        'ssim': r'SSIM[\s:]+([\d.]+)',
        'npcr': r'NPCR[\s:]+([\d.]+)%',
        'uaci': r'UACI[\s:]+([\d.]+)%',
        'entropy': r'Entropy[\s:]+([\d.]+)',
        'corr_h': r'Corr_H[\s:]+([-\d.]+)',
        'corr_v': r'Corr_V[\s:]+([-\d.]+)',
        'corr_d': r'Corr_D[\s:]+([-\d.]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                results[key] = float(match.group(1))
            except:
                pass
    
    return results


def collect_image_results(results_dir):
    """收集图像实验结果"""
    print("\n>>> Collecting Image Results...")
    
    all_results = []
    
    # 基础实验 exp_base
    base_path = os.path.join(results_dir, 'image_exp', 'exp_base', 'report.txt')
    if os.path.exists(base_path):
        data = parse_report_file(base_path)
        data['experiment'] = 'base'
        data['category'] = 'basic'
        all_results.append(data)
    
    # 迭代实验 exp2_iter
    iter_dir = os.path.join(results_dir, 'image_exp', 'exp2_iter')
    if os.path.isdir(iter_dir):
        for subdir in os.listdir(iter_dir):
            report_path = os.path.join(iter_dir, subdir, 'report.txt')
            if os.path.exists(report_path):
                data = parse_report_file(report_path)
                data['experiment'] = 'iter'
                data['parameter'] = subdir.replace('iter_', '')
                data['category'] = 'parameter_sweep'
                all_results.append(data)
    
    # Lambda实验
    lam_dir = os.path.join(results_dir, 'image_exp', 'exp2_lam')
    if os.path.isdir(lam_dir):
        for subdir in os.listdir(lam_dir):
            report_path = os.path.join(lam_dir, subdir, 'report.txt')
            if os.path.exists(report_path):
                data = parse_report_file(report_path)
                data['experiment'] = 'lambda'
                data['parameter'] = subdir.replace('lam_', '')
                data['category'] = 'parameter_sweep'
                all_results.append(data)
    
    # 尺寸实验 exp5_size
    size_dir = os.path.join(results_dir, 'image_exp', 'exp5_size')
    if os.path.isdir(size_dir):
        for subdir in os.listdir(size_dir):
            report_path = os.path.join(size_dir, subdir, 'report.txt')
            if os.path.exists(report_path):
                data = parse_report_file(report_path)
                data['experiment'] = 'size'
                data['parameter'] = subdir.replace('size_', '')
                data['category'] = 'parameter_sweep'
                all_results.append(data)
    
    # 密钥测试 exp6_keytest
    key_dir = os.path.join(results_dir, 'image_exp', 'exp6_keytest')
    if os.path.isdir(key_dir):
        for subdir in os.listdir(key_dir):
            report_path = os.path.join(key_dir, subdir, 'report.txt')
            if os.path.exists(report_path):
                data = parse_report_file(report_path)
                data['experiment'] = 'key_test'
                data['parameter'] = subdir
                data['category'] = 'security'
                all_results.append(data)
    
    # 选择加密 exp7_selenc
    selenc_dir = os.path.join(results_dir, 'image_exp', 'exp7_selenc')
    if os.path.isdir(selenc_dir):
        for subdir in os.listdir(selenc_dir):
            report_path = os.path.join(selenc_dir, subdir, 'report.txt')
            if os.path.exists(report_path):
                data = parse_report_file(report_path)
                data['experiment'] = 'selective_enc'
                data['parameter'] = subdir.replace('r', '')
                data['category'] = 'security'
                all_results.append(data)
    
    # 鲁棒性 exp8_robust
    robust_dir = os.path.join(results_dir, 'image_exp', 'exp8_robust')
    if os.path.isdir(robust_dir):
        for subdir in os.listdir(robust_dir):
            if os.path.isdir(os.path.join(robust_dir, subdir)):
                report_path = os.path.join(robust_dir, subdir, 'report.txt')
                if os.path.exists(report_path):
                    data = parse_report_file(report_path)
                    data['experiment'] = 'robustness'
                    data['parameter'] = subdir
                    data['category'] = 'security'
                    all_results.append(data)
    
    # CR可调实验
    cr_dirs = glob.glob(os.path.join(results_dir, 'image_exp', 'rd_cr*'))
    for cr_dir in cr_dirs:
        subdir = os.path.basename(cr_dir)
        report_path = os.path.join(cr_dir, 'report.txt')
        if os.path.exists(report_path):
            data = parse_report_file(report_path)
            data['experiment'] = 'cr_sweep'
            data['parameter'] = subdir.replace('rd_cr', '')
            data['category'] = 'compression'
            all_results.append(data)
    
    print(f"   Collected {len(all_results)} image experiment records")
    
    return all_results


def collect_video_results(results_dir):
    """收集视频实验结果"""
    print("\n>>> Collecting Video Results...")
    
    all_results = []
    
    video_exp_dir = os.path.join(results_dir, 'video_exp', 'exp_video_out')
    if os.path.isdir(video_exp_dir):
        for txt_file in glob.glob(os.path.join(video_exp_dir, '*.txt')):
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            data = {'experiment': 'video', 'category': 'video'}
            
            # 提取指标
            for key, pattern in [
                ('psnr', r'PSNR[\s:]+([\d.]+)'),
                ('ssim', r'SSIM[\s:]+([\d.]+)'),
            ]:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    data[key] = float(match.group(1))
            
            all_results.append(data)
    
    print(f"   Collected {len(all_results)} video experiment records")
    
    return all_results


def collect_audio_results(results_dir):
    """收集音频实验结果"""
    print("\n>>> Collecting Audio Results...")
    
    all_results = []
    
    audio_exp_dir = os.path.join(results_dir, 'audio_exp')
    if os.path.isdir(audio_exp_dir):
        for txt_file in glob.glob(os.path.join(audio_exp_dir, '*.txt')):
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            data = {'experiment': 'audio', 'category': 'audio'}
            
            # 提取指标
            for key, pattern in [
                ('snr', r'SNR[\s:]+([\d.]+)'),
                ('psnr', r'PSNR[\s:]+([\d.]+)'),
            ]:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    data[key] = float(match.group(1))
            
            all_results.append(data)
    
    print(f"   Collected {len(all_results)} audio experiment records")
    
    return all_results


def generate_summary_table(all_results):
    """生成汇总表格"""
    print("\n>>> Generating Summary Table...")
    
    df = pd.DataFrame(all_results)
    
    if df.empty:
        return None
    
    # 按实验类型分组统计
    summary_stats = []
    
    for exp in df['experiment'].unique():
        exp_data = df[df['experiment'] == exp]
        
        stats = {
            'experiment': exp,
            'count': len(exp_data),
        }
        
        # 计算各指标的平均值和标准差
        for col in ['psnr', 'ssim', 'npcr', 'uaci', 'entropy', 'corr_h', 'corr_v']:
            if col in exp_data.columns:
                stats[f'{col}_mean'] = exp_data[col].mean()
                stats[f'{col}_std'] = exp_data[col].std()
        
        summary_stats.append(stats)
    
    return pd.DataFrame(summary_stats)


def create_comparison_plots(all_results, output_dir):
    """创建对比图"""
    print("\n>>> Creating Comparison Plots...")
    
    df = pd.DataFrame(all_results)
    
    if df.empty or 'psnr' not in df.columns:
        print("   No data to plot")
        return
    
    # 1. PSNR对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 按实验类型
    exp_groups = df.groupby('experiment')['psnr'].mean().sort_values(ascending=False)
    
    axes[0, 0].barh(range(len(exp_groups)), exp_groups.values, color='steelblue', alpha=0.7)
    axes[0, 0].set_yticks(range(len(exp_groups)))
    axes[0, 0].set_yticklabels(exp_groups.index)
    axes[0, 0].set_xlabel('PSNR (dB)')
    axes[0, 0].set_title('PSNR by Experiment Type')
    axes[0, 0].axvline(x=30, color='r', linestyle='--', alpha=0.5, label='30 dB')
    axes[0, 0].legend()
    
    # 2. NPCR对比
    if 'npcr' in df.columns:
        npcr_groups = df.groupby('experiment')['npcr'].mean().sort_values(ascending=False)
        axes[0, 1].barh(range(len(npcr_groups)), npcr_groups.values, color='green', alpha=0.7)
        axes[0, 1].set_yticks(range(len(npcr_groups)))
        axes[0, 1].set_yticklabels(npcr_groups.index)
        axes[0, 1].set_xlabel('NPCR (%)')
        axes[0, 1].set_title('NPCR by Experiment Type')
        axes[0, 1].axvline(x=99.6, color='r', linestyle='--', alpha=0.5, label='99.6%')
        axes[0, 1].legend()
    
    # 3. SSIM对比
    if 'ssim' in df.columns:
        ssim_groups = df.groupby('experiment')['ssim'].mean().sort_values(ascending=False)
        axes[1, 0].barh(range(len(ssim_groups)), ssim_groups.values, color='orange', alpha=0.7)
        axes[1, 0].set_yticks(range(len(ssim_groups)))
        axes[1, 0].set_yticklabels(ssim_groups.index)
        axes[1, 0].set_xlabel('SSIM')
        axes[1, 0].set_title('SSIM by Experiment Type')
        axes[1, 0].axvline(x=0.9, color='r', linestyle='--', alpha=0.5, label='0.9')
        axes[1, 0].legend()
    
    # 4. 熵对比
    if 'entropy' in df.columns:
        ent_groups = df.groupby('experiment')['entropy'].mean().sort_values(ascending=False)
        axes[1, 1].barh(range(len(ent_groups)), ent_groups.values, color='purple', alpha=0.7)
        axes[1, 1].set_yticks(range(len(ent_groups)))
        axes[1, 1].set_yticklabels(ent_groups.index)
        axes[1, 1].set_xlabel('Entropy')
        axes[1, 1].set_title('Information Entropy by Experiment Type')
        axes[1, 1].axvline(x=8.0, color='r', linestyle='--', alpha=0.5, label='8.0 (ideal)')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experiment_comparison.png'), dpi=150)
    plt.close()
    
    # 5. 参数敏感性图
    if 'parameter' in df.columns:
        param_data = df[df['parameter'].notna()]
        
        if not param_data.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # 提取数值参数
            param_data = param_data.copy()
            param_data['param_num'] = param_data['parameter'].astype(float)
            
            # 按参数排序
            param_data = param_data.sort_values('param_num')
            
            if 'psnr' in param_data.columns:
                axes[0].plot(param_data['param_num'], param_data['psnr'], 'b-o', linewidth=2)
                axes[0].set_xlabel('Parameter Value')
                axes[0].set_ylabel('PSNR (dB)')
                axes[0].set_title('PSNR vs Parameter')
                axes[0].grid(True, alpha=0.3)
            
            if 'npcr' in param_data.columns:
                axes[1].plot(param_data['param_num'], param_data['npcr'], 'r-o', linewidth=2)
                axes[1].set_xlabel('Parameter Value')
                axes[1].set_ylabel('NPCR (%)')
                axes[1].set_title('NPCR vs Parameter')
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'parameter_sensitivity.png'), dpi=150)
            plt.close()
    
    print("   Plots saved")


def generate_html_dashboard(all_results, output_dir):
    """生成HTML仪表板"""
    print("\n>>> Generating HTML Dashboard...")
    
    df = pd.DataFrame(all_results)
    
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Chaos-CS Encryption Results Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { display: inline-block; margin: 10px 20px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { font-size: 12px; color: #666; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #007bff; color: white; }
        tr:hover { background: #f5f5f5; }
        .chart { text-align: center; margin: 20px 0; }
        .chart img { max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .status-pass { color: green; font-weight: bold; }
        .status-fail { color: red; font-weight: bold; }
    </style>
</head>
<body>
    <h1>🛡️ Chaos-CS Multimedia Encryption Results</h1>
    <p>Comprehensive experiment results dashboard</p>
"""
    
    # 1. 总体统计
    html += """
    <div class="card">
        <h2>📊 Overall Statistics</h2>
"""
    
    total_experiments = len(df)
    html += f"<p>Total Experiments: <strong>{total_experiments}</strong></p>"
    
    # 计算平均值
    metrics = {
        'psnr': ('PSNR', 'dB', 30),
        'ssim': ('SSIM', '', 0.9),
        'npcr': ('NPCR', '%', 99.6),
        'uaci': ('UACI', '%', 33.4),
        'entropy': ('Entropy', '', 8.0),
    }
    
    for col, (name, unit, threshold) in metrics.items():
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            status = 'status-pass' if mean_val >= threshold else 'status-fail'
            
            html += f"""
        <div class="metric">
            <div class="metric-value">{mean_val:.2f} {unit}</div>
            <div class="metric-label">{name} (avg ± std: {std_val:.2f}) <span class="{status}">{'✓' if mean_val >= threshold else '✗'}</span></div>
        </div>
"""
    
    html += "    </div>"
    
    # 2. 实验类型分布
    html += """
    <div class="card">
        <h2>📁 Experiment Types</h2>
        <table>
            <tr><th>Type</th><th>Count</th><th>Avg PSNR</th><th>Avg NPCR</th></tr>
"""
    
    if 'experiment' in df.columns:
        for exp in df['experiment'].unique():
            exp_data = df[df['experiment'] == exp]
            count = len(exp_data)
            psnr = exp_data['psnr'].mean() if 'psnr' in exp_data.columns else 0
            npcr = exp_data['npcr'].mean() if 'npcr' in exp_data.columns else 0
            
            html += f"""
            <tr>
                <td>{exp}</td>
                <td>{count}</td>
                <td>{psnr:.2f} dB</td>
                <td>{npcr:.2f}%</td>
            </tr>
"""
    
    html += "        </table>\n    </div>"
    
    # 3. 关键指标表
    html += """
    <div class="card">
        <h2>📋 Key Metrics Summary</h2>
        <table>
            <tr><th>Metric</th><th>Min</th><th>Max</th><th>Mean</th><th>Std</th><th>Status</th></tr>
"""
    
    for col, (name, unit, threshold) in metrics.items():
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            std_val = df[col].std()
            status = '✓' if mean_val >= threshold else '✗'
            
            html += f"""
            <tr>
                <td>{name}</td>
                <td>{min_val:.4f}</td>
                <td>{max_val:.4f}</td>
                <td>{mean_val:.4f}</td>
                <td>{std_val:.4f}</td>
                <td>{status}</td>
            </tr>
"""
    
    html += "        </table>\n    </div>"
    
    # 4. 可视化
    html += """
    <div class="card">
        <h2>📈 Visualizations</h2>
"""
    
    if os.path.exists(os.path.join(output_dir, 'experiment_comparison.png')):
        html += '<div class="chart"><img src="experiment_comparison.png" alt="Experiment Comparison"></div>'
    
    if os.path.exists(os.path.join(output_dir, 'parameter_sensitivity.png')):
        html += '<div class="chart"><img src="parameter_sensitivity.png" alt="Parameter Sensitivity"></div>'
    
    html += "    </div>"
    
    # 5. 结论
    html += """
    <div class="card">
        <h2>🎯 Conclusions</h2>
        <ul>
"""
    
    # 自动生成结论
    if 'psnr' in df.columns:
        avg_psnr = df['psnr'].mean()
        if avg_psnr >= 30:
            html += f"<li>✅ Excellent reconstruction quality: {avg_psnr:.2f} dB</li>"
        elif avg_psnr >= 25:
            html += f"<li>✅ Good reconstruction quality: {avg_psnr:.2f} dB</li>"
        else:
            html += f"<li>⚠️ Reconstruction quality needs improvement: {avg_psnr:.2f} dB</li>"
    
    if 'npcr' in df.columns:
        avg_npcr = df['npcr'].mean()
        if avg_npcr >= 99.5:
            html += f"<li>✅ Strong avalanche effect: {avg_npcr:.2f}%</li>"
        else:
            html += f"<li>⚠️ Avalanche effect needs improvement: {avg_npcr:.2f}%</li>"
    
    if 'entropy' in df.columns:
        avg_ent = df['entropy'].mean()
        if avg_ent >= 7.95:
            html += f"<li>✅ High randomness: {avg_ent:.4f}</li>"
        else:
            html += f"<li>⚠️ Randomness needs improvement: {avg_ent:.4f}</li>"
    
    html += """
        </ul>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    # 保存HTML
    html_path = os.path.join(output_dir, 'dashboard.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"   HTML dashboard saved: {html_path}")
    
    return html_path


def organize_all_results(base_dir, output_dir):
    """整合所有结果"""
    print("=" * 70)
    print("Comprehensive Results Organizer")
    print("整合所有实验结果")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集结果
    all_results = []
    
    # 图像
    all_results.extend(collect_image_results(base_dir))
    
    # 视频
    all_results.extend(collect_video_results(base_dir))
    
    # 音频
    all_results.extend(collect_audio_results(base_dir))
    
    print(f"\nTotal records collected: {len(all_results)}")
    
    if not all_results:
        print("No results found!")
        return
    
    # 保存原始数据
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'all_results_raw.csv')
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved: {csv_path}")
    
    # 生成汇总表
    summary_df = generate_summary_table(all_results)
    if summary_df is not None:
        summary_path = os.path.join(output_dir, 'summary_by_experiment.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved: {summary_path}")
    
    # 创建对比图
    create_comparison_plots(all_results, output_dir)
    
    # 生成HTML仪表板
    generate_html_dashboard(all_results, output_dir)
    
    # 生成JSON报告
    report = {
        'total_experiments': len(all_results),
        'experiments': list(df['experiment'].unique()) if 'experiment' in df.columns else [],
        'categories': list(df['category'].unique()) if 'category' in df.columns else [],
        'metrics': {}
    }
    
    for col in ['psnr', 'ssim', 'npcr', 'uaci', 'entropy', 'corr_h', 'corr_v']:
        if col in df.columns:
            report['metrics'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
    
    json_path = os.path.join(output_dir, 'comprehensive_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nJSON report saved: {json_path}")
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    
    print(f"\nTotal experiments: {len(all_results)}")
    
    if 'experiment' in df.columns:
        print("\nExperiments by type:")
        for exp, count in df['experiment'].value_counts().items():
            print(f"  - {exp}: {count}")
    
    print("\nKey Metrics:")
    for col, (name, unit, threshold) in {
        'psnr': ('PSNR', 'dB', 30),
        'ssim': ('SSIM', '', 0.9),
        'npcr': ('NPCR', '%', 99.6),
        'uaci': ('UACI', '%', 33.4),
        'entropy': ('Entropy', '', 8.0),
    }.items():
        if col in df.columns:
            mean_val = df[col].mean()
            status = 'PASS' if mean_val >= threshold else 'FAIL'
            print(f"  {name}: {mean_val:.4f} {unit} [{status}]")
    
    print("\n" + "=" * 70)
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize All Experiment Results')
    parser.add_argument('--base', type=str, default='results', help='Base results directory')
    parser.add_argument('--out', type=str, default='results/consolidated', help='Output directory')
    
    args = parser.parse_args()
    
    organize_all_results(args.base, args.out)
