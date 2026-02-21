# exp_7_stats_plus.py
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# 确保可以从父目录导入 exp_common（脚本已移动到 image/experiments/）
BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from exp_common import ensure_dir, write_csv

def chi_square_u8(img_u8: np.ndarray):
    hist = np.bincount(img_u8.flatten(), minlength=256).astype(np.float64)
    expected = hist.sum() / 256.0
    chi = np.sum((hist - expected) ** 2 / (expected + 1e-12))
    return float(chi)

def local_corr_blocks(img_u8: np.ndarray, block=32):
    H, W = img_u8.shape
    vals = []
    for r in range(0, H, block):
        for c in range(0, W, block):
            patch = img_u8[r:min(H,r+block), c:min(W,c+block)]
            if patch.shape[0] < 2 or patch.shape[1] < 2:
                continue
            a = patch[:, :-1].astype(np.float64).ravel()
            b = patch[:, 1:].astype(np.float64).ravel()
            a -= a.mean(); b -= b.mean()
            denom = (np.sqrt((a*a).mean()) * np.sqrt((b*b).mean()) + 1e-12)
            vals.append(float((a*b).mean()/denom))
    return float(np.mean(vals)) if vals else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="outputs_fista_full", help="a demo3 output dir containing cipher_uint8.png")
    ap.add_argument("--out", default="exp7_stats_plus")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out = Path(args.out).resolve()
    ensure_dir(out)

    cipher_path = run_dir / "cipher_uint8.png"
    img = np.array(Image.open(cipher_path).convert("L"), dtype=np.uint8)

    chi = chi_square_u8(img)
    lc = local_corr_blocks(img, block=32)

    rows = [{"run_dir": str(run_dir), "chi_square": chi, "local_corr_mean": lc}]
    write_csv(rows, out / "stats_plus.csv")
    print("Saved:", out / "stats_plus.csv")

if __name__ == "__main__":
    main()
