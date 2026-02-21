# exp_4_keys.py
import os
import sys
import argparse
import subprocess
import random
import string
from pathlib import Path

# 确保可以从父目录导入 exp_common（脚本已移动到 image/experiments/）
BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from exp_common import ensure_dir, parse_report, write_csv

def rand_key(prefix="k", n=16):
    return prefix + "_" + "".join(random.choice(string.ascii_letters + string.digits) for _ in range(n))

def run_demo(demo, img, out_dir, key, wrong_key, cr, lam, it, extra_args):
    cmd = [sys.executable, demo, "--img", img, "--out", str(out_dir),
           "--key", key, "--wrong_key", wrong_key, "--cr", str(cr), "--lam", str(lam), "--iter", str(it)]
    cmd += extra_args
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (out_dir / "console.log").write_text(p.stdout, encoding="utf-8", errors="ignore")
    return p.returncode

def main():
    ap = argparse.ArgumentParser()
    demo_default = str((Path(__file__).resolve().parent.parent / "demo" / "demo3.py"))
    ap.add_argument("--demo", default=demo_default)
    ap.add_argument("--img", default="1.png")
    ap.add_argument("--out", default="exp4_keys")
    ap.add_argument("--cr", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--iter", type=int, default=150)
    ap.add_argument("--K", type=int, default=20, help="number of random keys")
    ap.add_argument("--extra", default="")
    args = ap.parse_args()

    root = Path(args.out).resolve()
    ensure_dir(root)
    extra_args = args.extra.split() if args.extra.strip() else []

    rows = []
    base_key = "my-secret-key-2026"
    for i in range(1, args.K + 1):
        key = rand_key("key", 18)
        wrong_key = key + "_wrong"
        run_dir = root / f"key_{i:03d}"
        ensure_dir(run_dir)
        rc = run_demo(args.demo, args.img, run_dir, key, wrong_key, args.cr, args.lam, args.iter, extra_args)
        row = {"idx": i, "key": key, "return_code": rc, "run_dir": str(run_dir)}
        rep = run_dir / "report.txt"
        if rc == 0 and rep.exists():
            row.update(parse_report(rep))
        rows.append(row)
        print(f"[{i}/{args.K}] rc={rc} PSNR={row.get('PSNR')} NPCR={row.get('NPCR')} UACI={row.get('UACI')}")

    write_csv(rows, root / "per_key.csv")

if __name__ == "__main__":
    main()
