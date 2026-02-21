#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Ablation Experiment
视频消融实验
"""

import os
import sys
import json
import subprocess

# Test if video dependencies are available
try:
    import cv2
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False
    print("Warning: OpenCV or scikit-image not available. Video ablation will be simplified.")


def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """计算SSIM"""
    return ssim(img1, img2, data_range=255)


def extract_frames(video_path, max_frames=30):
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


def run_video_ablation():
    """运行视频消融实验"""
    print("=" * 70)
    print("Video Ablation Study")
    print("视频消融实验")
    print("=" * 70)
    
    if not VIDEO_AVAILABLE:
        print("Skipping video ablation - dependencies not available")
        return []
    
    video_path = "resources/video/small.mp4"
    key = "my-secret-key-2026"
    out_dir = "results/video_ablation"
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    
    # Extract original frames
    print("\n>>> Extracting original frames...")
    orig_frames = extract_frames(video_path, max_frames=30)
    print(f"   Extracted {len(orig_frames)} frames")
    
    # 1. Full encryption (baseline) - ARX + S-Box
    print("\n>>> [1/4] Full Encryption (ARX + S-Box)")
    npz_path1 = os.path.join(out_dir, "full.npz")
    cmd1 = f'python video/video_3dcs_spcmm.py encrypt --in_file {video_path} --out_file {npz_path1} --key "{key}" --chunk 30 --cube_t 8 --block 16 --key_stride 2 --m_rate_key 0.7 --arx 1 --sbox 1'
    
    try:
        subprocess.run(cmd1, shell=True, capture_output=True, timeout=300)
    except:
        pass
    
    # Try to decrypt
    dec_path1 = os.path.join(out_dir, "full_dec.mp4")
    cmd1d = f'python video/video_3dcs_spcmm.py decrypt --in_file {npz_path1} --out_file {dec_path1} --key "{key}" --arx 1 --sbox 1'
    
    try:
        subprocess.run(cmd1d, shell=True, capture_output=True, timeout=300)
    except:
        pass
    
    # Calculate metrics
    if os.path.exists(dec_path1):
        dec_frames = extract_frames(dec_path1, max_frames=30)
        if dec_frames and orig_frames:
            psnr = np.mean([calculate_psnr(o, d) for o, d in zip(orig_frames, dec_frames)])
            ssim_val = np.mean([calculate_ssim(o, d) for o, d in zip(orig_frames, dec_frames)])
            results.append({'config': 'full_arx_sbox', 'psnr': psnr, 'ssim': ssim_val})
            print(f"   PSNR: {psnr:.2f} dB, SSIM: {ssim_val:.4f}")
    
    # 2. No ARX (S-Box only)
    print("\n>>> [2/4] No ARX (S-Box only)")
    npz_path2 = os.path.join(out_dir, "no_arx.npz")
    cmd2 = f'python video/video_3dcs_spcmm.py encrypt --in_file {video_path} --out_file {npz_path2} --key "{key}" --chunk 30 --cube_t 8 --block 16 --key_stride 2 --m_rate_key 0.7 --arx 0 --sbox 1'
    
    try:
        subprocess.run(cmd2, shell=True, capture_output=True, timeout=300)
    except:
        pass
    
    dec_path2 = os.path.join(out_dir, "no_arx_dec.mp4")
    cmd2d = f'python video/video_3dcs_spcmm.py decrypt --in_file {npz_path2} --out_file {dec_path2} --key "{key}" --arx 0 --sbox 1'
    
    try:
        subprocess.run(cmd2d, shell=True, capture_output=True, timeout=300)
    except:
        pass
    
    if os.path.exists(dec_path2):
        dec_frames2 = extract_frames(dec_path2, max_frames=30)
        if dec_frames2 and orig_frames:
            psnr2 = np.mean([calculate_psnr(o, d) for o, d in zip(orig_frames, dec_frames2)])
            ssim2 = np.mean([calculate_ssim(o, d) for o, d in zip(orig_frames, dec_frames2)])
            results.append({'config': 'no_arx_sbox', 'psnr': psnr2, 'ssim': ssim2})
            print(f"   PSNR: {psnr2:.2f} dB, SSIM: {ssim2:.4f}")
    
    # 3. ARX only (no S-Box)
    print("\n>>> [3/4] ARX only (no S-Box)")
    npz_path3 = os.path.join(out_dir, "no_sbox.npz")
    cmd3 = f'python video/video_3dcs_spcmm.py encrypt --in_file {video_path} --out_file {npz_path3} --key "{key}" --chunk 30 --cube_t 8 --block 16 --key_stride 2 --m_rate_key 0.7 --arx 1 --sbox 0'
    
    try:
        subprocess.run(cmd3, shell=True, capture_output=True, timeout=300)
    except:
        pass
    
    dec_path3 = os.path.join(out_dir, "no_sbox_dec.mp4")
    cmd3d = f'python video/video_3dcs_spcmm.py decrypt --in_file {npz_path3} --out_file {dec_path3} --key "{key}" --arx 1 --sbox 0'
    
    try:
        subprocess.run(cmd3d, shell=True, capture_output=True, timeout=300)
    except:
        pass
    
    if os.path.exists(dec_path3):
        dec_frames3 = extract_frames(dec_path3, max_frames=30)
        if dec_frames3 and orig_frames:
            psnr3 = np.mean([calculate_psnr(o, d) for o, d in zip(orig_frames, dec_frames3)])
            ssim3 = np.mean([calculate_ssim(o, d) for o, d in zip(orig_frames, dec_frames3)])
            results.append({'config': 'arx_no_sbox', 'psnr': psnr3, 'ssim': ssim3})
            print(f"   PSNR: {psnr3:.2f} dB, SSIM: {ssim3:.4f}")
    
    # 4. No encryption (baseline CS only)
    print("\n>>> [4/4] No Encryption (CS only)")
    npz_path4 = os.path.join(out_dir, "no_enc.npz")
    cmd4 = f'python video/video_3dcs_spcmm.py encrypt --in_file {video_path} --out_file {npz_path4} --key "{key}" --chunk 30 --cube_t 8 --block 16 --key_stride 2 --m_rate_key 0.7 --arx 0 --sbox 0'
    
    try:
        subprocess.run(cmd4, shell=True, capture_output=True, timeout=300)
    except:
        pass
    
    dec_path4 = os.path.join(out_dir, "no_enc_dec.mp4")
    cmd4d = f'python video/video_3dcs_spcmm.py decrypt --in_file {npz_path4} --out_file {dec_path4} --key "{key}" --arx 0 --sbox 0'
    
    try:
        subprocess.run(cmd4d, shell=True, capture_output=True, timeout=300)
    except:
        pass
    
    if os.path.exists(dec_path4):
        dec_frames4 = extract_frames(dec_path4, max_frames=30)
        if dec_frames4 and orig_frames:
            psnr4 = np.mean([calculate_psnr(o, d) for o, d in zip(orig_frames, dec_frames4)])
            ssim4 = np.mean([calculate_ssim(o, d) for o, d in zip(orig_frames, dec_frames4)])
            results.append({'config': 'no_encryption', 'psnr': psnr4, 'ssim': ssim4})
            print(f"   PSNR: {psnr4:.2f} dB, SSIM: {ssim4:.4f}")
    
    # Save results
    with open(os.path.join(out_dir, "ablation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Video Ablation Summary")
    print("=" * 70)
    
    print(f"\n{'Config':<25} {'PSNR (dB)':<12} {'SSIM':<12}")
    print("-" * 50)
    
    for r in results:
        print(f"{r['config']:<25} {r['psnr']:<12.2f} {r['ssim']:<12.4f}")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    run_video_ablation()
