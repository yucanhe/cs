#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像加密测试脚本 - 支持AMP/FISTA重建算法对比
使用内置加密/解密流程
"""

import os
import sys
import time
import argparse
import numpy as np
from PIL import Image

# 确保可以导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image.demo.demo3 import (
    CSChaosCrypto, ensure_dir
)
from image.amp_reconstructor import amp_l1, amp_l1_numba


def load_image(img_path):
    """加载图像"""
    img = Image.open(img_path).convert("L")
    return np.array(img, dtype=np.uint8)


def test_reconstructor(img_path, key, reconstructor="fista", cr=0.5, lam=0.005, fista_iter=150):
    """测试不同重建算法"""
    
    print(f"\n{'='*60}")
    print(f"Testing Reconstructor: {reconstructor.upper()}")
    print(f"Image: {img_path}")
    print(f"CR: {cr}, Lam: {lam}, Iter: {fista_iter}")
    print(f"{'='*60}")
    
    # 加载图像
    img = load_image(img_path)
    H, W = img.shape
    B = 8
    
    # 裁剪到8的倍数
    H2, W2 = (H // B) * B, (W // B) * B
    img = img[:H2, :W2]
    
    # 输出目录
    ensure_dir("results/amp_test")
    out_dir = f"results/amp_test/{reconstructor}_cr{int(cr*100)}"
    ensure_dir(out_dir)
    
    # 加密 - 不使用float diffusion
    crypto = CSChaosCrypto(key, B=B, m_meas=int(cr*64))
    
    print("\n[1] Encrypting...")
    t0 = time.time()
    D, state = crypto.encrypt_prequant_D(img, float_diffusion=False)
    enc_time = time.time() - t0
    print(f"    Encrypt time: {enc_time:.2f}s")
    
    # 直接使用内置decrypt_from_prequant_D
    # 但我们需要替换fista_l1为自定义重建器
    
    # 手动实现解密核心逻辑（复制decrypt_from_prequant_D但使用自定义重建器）
    print(f"\n[2] Decrypting with {reconstructor.upper()}...")
    
    from image.demo.demo3 import frac_arr, spcmm_generate_xyz, phi_from_key_z, unblockify, idct2, fista_l1
    
    gh, gw = state.grid_h, state.grid_w
    D = frac_arr(D.astype(np.float64))
    _, _, zs = spcmm_generate_xyz(key, state.chaos_need)
    
    flat_mask = zs[state.mask_start:state.mask_start + gh * gw * state.m_meas]
    mask = frac_arr(flat_mask.reshape((gh, gw, state.m_meas)))
    
    a, b, c = state.diff_abc
    # 不使用float diffusion
    Yn = D.copy()
    
    denom = (state.y_max - state.y_min) if (state.y_max - state.y_min) > 1e-12 else 1.0
    Y = Yn * denom + state.y_min
    
    Y_meas = Y.reshape((-1, state.m_meas))
    N = Y_meas.shape[0]
    Omegas_hat = np.zeros((N, 64), dtype=np.float64)
    
    # Phi cache for static case
    Phi_static_cache = None
    if state.static_phi:
        from image.demo.demo3 import phi_static
        Phi_static_cache = phi_static(key, state.m_meas, 64)
    
    t1 = time.time()
    
    # 选择重建算法
    if reconstructor == "fista":
        for bi in range(N):
            if state.static_phi:
                Phi = Phi_static_cache
            else:
                Phi = phi_from_key_z(key, zs[bi], block_id=bi, m_meas=state.m_meas, n=64)
            y = Y_meas[bi]
            if state.m_meas == 64:
                Omegas_hat[bi] = Phi.T @ y
            else:
                Omegas_hat[bi] = fista_l1(Phi, y, lam=lam, max_iter=fista_iter, tol=1e-5)
    elif reconstructor == "amp":
        for bi in range(N):
            if state.static_phi:
                Phi = Phi_static_cache
            else:
                Phi = phi_from_key_z(key, zs[bi], block_id=bi, m_meas=state.m_meas, n=64)
            y = Y_meas[bi]
            if state.m_meas == 64:
                Omegas_hat[bi] = Phi.T @ y
            else:
                Omegas_hat[bi] = amp_l1(Phi, y, lam=lam, max_iter=fista_iter, tol=1e-5)
    elif reconstructor == "amp_numba":
        for bi in range(N):
            if state.static_phi:
                Phi = Phi_static_cache
            else:
                Phi = phi_from_key_z(key, zs[bi], block_id=bi, m_meas=state.m_meas, n=64)
            y = Y_meas[bi]
            if state.m_meas == 64:
                Omegas_hat[bi] = Phi.T @ y
            else:
                Omegas_hat[bi] = amp_l1_numba(Phi, y, lam=lam, max_iter=fista_iter, tol=1e-5)
    
    dec_time = time.time() - t1
    print(f"    Decrypt time: {dec_time:.2f}s")
    
    # 重建图像
    blocks_rec = []
    for i in range(N):
        C_rec = Omegas_hat[i].reshape((B, B))
        blk = idct2(C_rec)
        blocks_rec.append(blk)
    blocks_rec = np.array(blocks_rec, dtype=np.float64)
    
    Pn_hat = unblockify(blocks_rec, (gh, gw), B=B)
    Pn_hat = np.clip(Pn_hat, 0, 1)
    decrypted = (Pn_hat * 255.0 + 0.5).astype(np.uint8)
    
    # 保存解密图像
    Image.fromarray(decrypted).save(os.path.join(out_dir, "decrypted.png"))
    Image.fromarray(img).save(os.path.join(out_dir, "plain.png"))
    
    # 计算质量指标
    from skimage.metrics import structural_similarity as ssim
    
    mse = np.mean((img.astype(float) - decrypted.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    ssim_val = ssim(img, decrypted, data_range=255)
    mae = np.mean(np.abs(img.astype(float) - decrypted.astype(float)))
    
    print(f"\n[3] Quality Metrics:")
    print(f"    MSE:  {mse:.2f}")
    print(f"    MAE:  {mae:.2f}")
    print(f"    PSNR: {psnr:.2f} dB")
    print(f"    SSIM: {ssim_val:.4f}")
    print(f"    Total time: {enc_time + dec_time:.2f}s")
    
    return {
        "reconstructor": reconstructor,
        "cr": cr,
        "psnr": psnr,
        "ssim": ssim_val,
        "mae": mae,
        "enc_time": enc_time,
        "dec_time": dec_time,
        "total_time": enc_time + dec_time
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="resources/images/lena.png")
    parser.add_argument("--key", type=str, default="my-secret-key-2026")
    parser.add_argument("--cr", type=float, default=0.7)
    parser.add_argument("--lam", type=float, default=0.005)
    parser.add_argument("--iter", type=int, default=150)
    parser.add_argument("--reconstructor", type=str, default="all", 
                       choices=["fista", "amp", "amp_numba", "all"])
    args = parser.parse_args()
    
    results = []
    
    # 根据选择运行
    if args.reconstructor == "all":
        reconstructors = ["fista", "amp", "amp_numba"]
    else:
        reconstructors = [args.reconstructor]
    
    for recon in reconstructors:
        result = test_reconstructor(
            args.img, args.key, 
            reconstructor=recon, 
            cr=args.cr, 
            lam=args.lam, 
            fista_iter=args.iter
        )
        results.append(result)
    
    # 打印对比表
    print("\n" + "="*70)
    print("Reconstructor Comparison")
    print("="*70)
    print(f"\n{'Reconstructor':<15} {'CR':<6} {'PSNR':<10} {'SSIM':<10} {'EncTime':<10} {'DecTime':<10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['reconstructor']:<15} {r['cr']:<6.2f} {r['psnr']:<10.2f} {r['ssim']:<10.4f} {r['enc_time']:<10.2f}s {r['dec_time']:<10.2f}s")
    
    # 保存结果
    import json
    with open("results/amp_test/comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n>>> Results saved to results/amp_test/")


if __name__ == "__main__":
    main()
