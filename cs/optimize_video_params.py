#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频加密参数优化实验
"""

import os
import sys
import json
import subprocess

# 检查依赖
try:
    import cv2
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False
    print("Warning: OpenCV not available")


def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """计算SSIM"""
    return ssim(img1, img2, data_range=255)


def extract_frames(video_path, max_frames=20):
    """提取视频帧"""
    if not VIDEO_AVAILABLE:
        return None
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    
    cap.release()
    return frames


def run_video_optimize():
    """运行视频参数优化"""
    print("=" * 70)
    print("Video Parameter Optimization")
    print("视频加密参数优化")
    print("=" * 70)
    
    if not VIDEO_AVAILABLE:
        print("Skipping - OpenCV not available")
        return []
    
    video_path = "resources/video/small.mp4"
    key = "my-secret-key-2026"
    out_dir = "results/optimize_video"
    os.makedirs(out_dir, exist_ok=True)
    
    # 提取原始帧
    print("\n>>> 提取原始视频帧...")
    orig_frames = extract_frames(video_path, max_frames=20)
    print(f"   提取了 {len(orig_frames)} 帧")
    
    results = []
    
    # 1. m_rate_key 优化
    print("\n>>> [1/3] 优化 m_rate_key (关键帧压缩率)")
    
    m_rates = [0.5, 0.6, 0.7, 0.8]
    
    for m_rate_key in m_rates:
        print(f"\n   Testing m_rate_key={m_rate_key}")
        
        npz_path = os.path.join(out_dir, f"video_m{m_rate_key}.npz")
        
        cmd = f'python video/video_3dcs_spcmm.py encrypt --in_file {video_path} --out_file {npz_path} --key "{key}" --chunk 20 --cube_t 8 --block 16 --key_stride 2 --m_rate_key {m_rate_key} --arx 1 --sbox 1'
        
        try:
            subprocess.run(cmd, shell=True, capture_output=True, timeout=300)
        except:
            pass
        
        # 解密
        dec_path = os.path.join(out_dir, f"video_m{m_rate_key}_dec.mp4")
        cmd_d = f'python video/video_3dcs_spcmm.py decrypt --in_file {npz_path} --out_file {dec_path} --key "{key}" --arx 1 --sbox 1'
        
        try:
            subprocess.run(cmd_d, shell=True, capture_output=True, timeout=300)
        except:
            pass
        
        # 计算质量
        if os.path.exists(dec_path):
            dec_frames = extract_frames(dec_path, max_frames=20)
            if dec_frames and orig_frames:
                psnr = np.mean([calculate_psnr(o, d) for o, d in zip(orig_frames, dec_frames)])
                ssim_val = np.mean([calculate_ssim(o, d) for o, d in zip(orig_frames, dec_frames)])
                
                print(f"   PSNR: {psnr:.2f} dB, SSIM: {ssim_val:.4f}")
                
                results.append({
                    'param': 'm_rate_key',
                    'value': m_rate_key,
                    'psnr': psnr,
                    'ssim': ssim_val,
                })
    
    # 2. key_stride 优化
    print("\n>>> [2/3] 优化 key_stride (关键帧间隔)")
    
    key_strides = [1, 2, 3, 4]
    
    for key_stride in key_strides:
        print(f"\n   Testing key_stride={key_stride}")
        
        npz_path = os.path.join(out_dir, f"video_s{key_stride}.npz")
        
        cmd = f'python video/video_3dcs_spcmm.py encrypt --in_file {video_path} --out_file {npz_path} --key "{key}" --chunk 20 --cube_t 8 --block 16 --key_stride {key_stride} --m_rate_key 0.7 --arx 1 --sbox 1'
        
        try:
            subprocess.run(cmd, shell=True, capture_output=True, timeout=300)
        except:
            pass
        
        dec_path = os.path.join(out_dir, f"video_s{key_stride}_dec.mp4")
        cmd_d = f'python video/video_3dcs_spcmm.py decrypt --in_file {npz_path} --out_file {dec_path} --key "{key}" --arx 1 --sbox 1'
        
        try:
            subprocess.run(cmd_d, shell=True, capture_output=True, timeout=300)
        except:
            pass
        
        if os.path.exists(dec_path):
            dec_frames = extract_frames(dec_path, max_frames=20)
            if dec_frames and orig_frames:
                psnr = np.mean([calculate_psnr(o, d) for o, d in zip(orig_frames, dec_frames)])
                ssim_val = np.mean([calculate_ssim(o, d) for o, d in zip(orig_frames, dec_frames)])
                
                print(f"   PSNR: {psnr:.2f} dB, SSIM: {ssim_val:.4f}")
                
                results.append({
                    'param': 'key_stride',
                    'value': key_stride,
                    'psnr': psnr,
                    'ssim': ssim_val,
                })
    
    # 3. iters 优化
    print("\n>>> [3/3] 优化 iters (迭代次数)")
    
    iters_list = [50, 100, 150]
    
    for iters in iters_list:
        print(f"\n   Testing iters={iters}")
        
        npz_path = os.path.join(out_dir, f"video_i{iters}.npz")
        
        cmd = f'python video/video_3dcs_spcmm.py encrypt --in_file {video_path} --out_file {npz_path} --key "{key}" --chunk 20 --cube_t 8 --block 16 --key_stride 2 --m_rate_key 0.7 --arx 1 --sbox 1 --iter {iters}'
        
        try:
            subprocess.run(cmd, shell=True, capture_output=True, timeout=300)
        except:
            pass
        
        dec_path = os.path.join(out_dir, f"video_i{iters}_dec.mp4")
        cmd_d = f'python video/video_3dcs_spcmm.py decrypt --in_file {npz_path} --out_file {dec_path} --key "{key}" --arx 1 --sbox 1'
        
        try:
            subprocess.run(cmd_d, shell=True, capture_output=True, timeout=300)
        except:
            pass
        
        if os.path.exists(dec_path):
            dec_frames = extract_frames(dec_path, max_frames=20)
            if dec_frames and orig_frames:
                psnr = np.mean([calculate_psnr(o, d) for o, d in zip(orig_frames, dec_frames)])
                ssim_val = np.mean([calculate_ssim(o, d) for o, d in zip(orig_frames, dec_frames)])
                
                print(f"   PSNR: {psnr:.2f} dB, SSIM: {ssim_val:.4f}")
                
                results.append({
                    'param': 'iters',
                    'value': iters,
                    'psnr': psnr,
                    'ssim': ssim_val,
                })
    
    # 保存结果
    with open(os.path.join(out_dir, "video_optimization.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("Video Optimization Summary")
    print("=" * 70)
    
    print("\n>>> m_rate_key 影响:")
    m_rate_results = [r for r in results if r['param'] == 'm_rate_key']
    for r in sorted(m_rate_results, key=lambda x: x['value']):
        print(f"   m_rate_key={r['value']}: PSNR={r['psnr']:.2f}dB, SSIM={r['ssim']:.4f}")
    
    print("\n>>> key_stride 影响:")
    stride_results = [r for r in results if r['param'] == 'key_stride']
    for r in sorted(stride_results, key=lambda x: x['value']):
        print(f"   key_stride={r['value']}: PSNR={r['psnr']:.2f}dB, SSIM={r['ssim']:.4f}")
    
    print("\n>>> iters 影响:")
    iters_results = [r for r in results if r['param'] == 'iters']
    for r in sorted(iters_results, key=lambda x: x['value']):
        print(f"   iters={r['value']}: PSNR={r['psnr']:.2f}dB, SSIM={r['ssim']:.4f}")
    
    return results


if __name__ == "__main__":
    run_video_optimize()
