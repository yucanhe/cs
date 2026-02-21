#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMP参数测试脚本
"""

import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image.demo.demo3 import (
    CSChaosCrypto, frac_arr, spcmm_generate_xyz, 
    phi_from_key_z, unblockify, idct2
)
from image.amp_reconstructor import amp_l1


def test_amp_lam(img_path, key, cr, lam):
    img = np.array(Image.open(img_path).convert('L'), dtype=np.uint8)
    img = img[:256, :256]

    crypto = CSChaosCrypto(key, B=8, m_meas=int(cr*64))
    D, state = crypto.encrypt_prequant_D(img, float_diffusion=False)

    gh, gw = state.grid_h, state.grid_w
    D = frac_arr(D.astype(np.float64))
    _, _, zs = spcmm_generate_xyz(key, state.chaos_need)
    flat_mask = zs[state.mask_start:state.mask_start + gh * gw * state.m_meas]
    mask = frac_arr(flat_mask.reshape((gh, gw, state.m_meas)))
    a, b, c = state.diff_abc
    Yn = D.copy()
    denom = (state.y_max - state.y_min) if (state.y_max - state.y_min) > 1e-12 else 1.0
    Y = Yn * denom + state.y_min
    Y_meas = Y.reshape((-1, state.m_meas))
    N = Y_meas.shape[0]

    Omegas_hat = np.zeros((N, 64), dtype=np.float64)
    t0 = time.time()
    for bi in range(N):
        Phi = phi_from_key_z(key, zs[bi], block_id=bi, m_meas=state.m_meas, n=64)
        y = Y_meas[bi]
        Omegas_hat[bi] = amp_l1(Phi, y, lam=lam, max_iter=150, tol=1e-5)
    dec_time = time.time() - t0

    blocks_rec = []
    for i in range(N):
        C_rec = Omegas_hat[i].reshape((8, 8))
        blk = idct2(C_rec)
        blocks_rec.append(blk)
    blocks_rec = np.array(blocks_rec, dtype=np.float64)
    Pn_hat = unblockify(blocks_rec, (gh, gw), B=8)
    Pn_hat = np.clip(Pn_hat, 0, 1)
    decrypted = (Pn_hat * 255.0 + 0.5).astype(np.uint8)

    mse = np.mean((img.astype(float) - decrypted.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    from skimage.metrics import structural_similarity as ssim
    ssim_val = ssim(img, decrypted, data_range=255)
    
    return psnr, ssim_val, dec_time


if __name__ == "__main__":
    print("Testing different lambda values for AMP...")
    print("-" * 50)
    
    for lam in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        psnr, ssim_val, t = test_amp_lam("resources/images/lena.png", "my-secret-key-2026", 0.7, lam)
        print(f"lam={lam:6.4f}: PSNR={psnr:6.2f}dB, SSIM={ssim_val:.4f}, time={t:.2f}s")
