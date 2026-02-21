# exp_5_attacks.py
import os
import sys
import argparse
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

# 确保可以从父目录导入 exp_common（脚本已移动到 image/experiments/）
BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from exp_common import ensure_dir, write_csv, parse_report

def run_demo(demo, img, out_dir, key, wrong_key, cr, lam, it, extra_args, override_cipher=None):
    cmd = [sys.executable, demo, "--img", img, "--out", str(out_dir),
           "--key", key, "--wrong_key", wrong_key, "--cr", str(cr), "--lam", str(lam), "--iter", str(it)]
    # 可选：如果你在 demo3.py 里加了 “--cipher_in” 从外部密文解密的模式
    if override_cipher is not None:
        cmd += ["--cipher_in", str(override_cipher)]
    cmd += extra_args
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (out_dir / "console.log").write_text(p.stdout, encoding="utf-8", errors="ignore")
    return p.returncode

def add_gaussian_rgba(in_png: Path, out_png: Path, sigma: float):
    im = Image.open(in_png).convert("RGBA")
    arr = np.array(im, dtype=np.int16)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.int16)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_png)

def salt_pepper_rgba(in_png: Path, out_png: Path, p: float):
    im = Image.open(in_png).convert("RGBA")
    arr = np.array(im, dtype=np.uint8)
    rnd = np.random.rand(*arr.shape[:2])
    out = arr.copy()
    # p/2 -> 0, p/2 -> 255
    out[rnd < (p/2)] = 0
    out[(rnd >= (p/2)) & (rnd < p)] = 255
    Image.fromarray(out).save(out_png)

def crop_zero_rgba(in_png: Path, out_png: Path, ratio: float):
    im = Image.open(in_png).convert("RGBA")
    arr = np.array(im, dtype=np.uint8)
    H, W, _ = arr.shape
    ch = int(H * ratio)
    cw = int(W * ratio)
    out = arr.copy()
    out[:ch, :cw, :] = 0
    Image.fromarray(out).save(out_png)

def jpeg_reencode_rgba(in_png: Path, out_png: Path, quality: int):
    # RGBA -> RGB -> JPEG -> PNG (必然破坏 alpha 与字节，实验用)
    im = Image.open(in_png).convert("RGB")
    tmp_jpg = out_png.with_suffix(".jpg")
    im.save(tmp_jpg, quality=quality)
    im2 = Image.open(tmp_jpg).convert("RGBA")
    Image.fromarray(np.array(im2, dtype=np.uint8)).save(out_png)

def main():
    ap = argparse.ArgumentParser()
    demo_default = str((Path(__file__).resolve().parent.parent / "demo" / "demo3.py"))
    ap.add_argument("--demo", default=demo_default)
    ap.add_argument("--img", default="1.png")
    ap.add_argument("--out", default="exp5_attacks")
    ap.add_argument("--key", default="my-secret-key-2026")
    ap.add_argument("--wrong_key", default="my-secret-key-2026_wrong")
    ap.add_argument("--cr", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--iter", type=int, default=150)
    ap.add_argument("--extra", default="")
    args = ap.parse_args()

    root = Path(args.out).resolve()
    ensure_dir(root)
    extra_args = args.extra.split() if args.extra.strip() else []

    # 先跑一次正常加密，拿到 cipher_rgba.png
    base_dir = root / "base"
    ensure_dir(base_dir)
    rc = run_demo(args.demo, args.img, base_dir, args.key, args.wrong_key, args.cr, args.lam, args.iter, extra_args)
    assert rc == 0, "base run failed"
    cipher = base_dir / "cipher_rgba.png"

    attacks = []
    # Gaussian
    for sigma in [0.5, 1.0, 2.0, 5.0]:
        attacks.append(("gauss", f"sigma{sigma}", lambda ip, op: add_gaussian_rgba(ip, op, sigma)))
    # S&P
    for p in [0.001, 0.005, 0.01]:
        attacks.append(("sp", f"p{p}", lambda ip, op: salt_pepper_rgba(ip, op, p)))
    # crop
    for ratio in [0.05, 0.1, 0.2]:
        attacks.append(("crop0", f"r{ratio}", lambda ip, op: crop_zero_rgba(ip, op, ratio)))
    # jpeg
    for q in [90, 70, 50]:
        attacks.append(("jpeg", f"q{q}", lambda ip, op: jpeg_reencode_rgba(ip, op, q)))

    rows = []
    for i, (atype, tag, fn) in enumerate(attacks, 1):
        adir = root / f"{atype}_{tag}"
        ensure_dir(adir)
        attacked = adir / "cipher_attacked.png"
        fn(cipher, attacked)

        # 这里需要你在 demo3.py 加一个：--cipher_in attacked.png 只做解密流程
        rc2 = run_demo(args.demo, args.img, adir, args.key, args.wrong_key, args.cr, args.lam, args.iter, extra_args, override_cipher=attacked)

        row = {"attack": atype, "level": tag, "return_code": rc2, "run_dir": str(adir)}
        rep = adir / "report.txt"
        if rc2 == 0 and rep.exists():
            row.update(parse_report(rep))
        rows.append(row)
        print(f"[{i}/{len(attacks)}] {atype}-{tag} rc={rc2} PSNR={row.get('PSNR')} SSIM={row.get('SSIM')}")

    write_csv(rows, root / "attacks.csv")

if __name__ == "__main__":
    main()
