# exp_6_ablation.py
import os
import sys
import argparse
import subprocess
from pathlib import Path

# 确保可以从父目录导入 exp_common（脚本已移动到 image/experiments/）
BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from exp_common import ensure_dir, parse_report, write_csv

def run_demo(demo, img, out_dir, key, wrong_key, cr, lam, it, flags):
    cmd = [sys.executable, demo, "--img", img, "--out", str(out_dir),
           "--key", key, "--wrong_key", wrong_key,
           "--cr", str(cr), "--lam", str(lam), "--iter", str(it)]
    cmd += flags
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (out_dir / "console.log").write_text(p.stdout, encoding="utf-8", errors="ignore")
    return p.returncode

def main():
    ap = argparse.ArgumentParser()
    demo_default = str((Path(__file__).resolve().parent.parent / "demo" / "demo3.py"))
    ap.add_argument("--demo", default=demo_default)
    ap.add_argument("--img", default="1.png")
    ap.add_argument("--out", default="exp6_ablation")
    ap.add_argument("--key", default="my-secret-key-2026")
    ap.add_argument("--wrong_key", default="my-secret-key-2026_wrong")
    ap.add_argument("--cr", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--iter", type=int, default=150)
    args = ap.parse_args()

    root = Path(args.out).resolve()
    ensure_dir(root)

    cases = [
        ("FULL", []),
        ("NO_FLOAT_DIFF", ["--no_float_diff"]),
        ("NO_U32_ARX", ["--no_u32_arx"]),
        ("NO_SBOX", ["--no_sbox"]),
        ("NO_PASS2", ["--no_pass2"]),
        ("WITH_PERMUTE", ["--permute_blocks"]),
        ("PHI_FIXED", ["--phi_fixed"]),
        ("PHI_ORTH_ROWS", ["--phi_mode", "orth_rows"]),
    ]

    rows = []
    for i, (name, flags) in enumerate(cases, 1):
        run_dir = root / name
        ensure_dir(run_dir)
        rc = run_demo(args.demo, args.img, run_dir, args.key, args.wrong_key, args.cr, args.lam, args.iter, flags)
        row = {"case": name, "flags": " ".join(flags), "return_code": rc, "run_dir": str(run_dir)}
        rep = run_dir / "report.txt"
        if rc == 0 and rep.exists():
            row.update(parse_report(rep))
        rows.append(row)
        print(f"[{i}/{len(cases)}] {name} PSNR={row.get('PSNR')} NPCR={row.get('NPCR')} UACI={row.get('UACI')}")
    write_csv(rows, root / "ablation.csv")

if __name__ == "__main__":
    main()
