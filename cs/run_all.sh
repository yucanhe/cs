#!/usr/bin/env bash
set -e
source .venv/bin/activate

# 2.1 lam sweep
mkdir -p exp2_lam
for lam in 0.003 0.005 0.008 0.01 0.012 0.015 0.02; do
  python3 image/demo/demo3.py --img 1.png --cr 0.5 --lam $lam --iter 120 --out exp2_lam/lam_${lam}
done

# 2.2 iter sweep
mkdir -p exp2_iter
for it in 50 80 100 120 150 200 250; do
  python3 image/demo/demo3.py --img 1.png --cr 0.5 --lam 0.01 --iter $it --out exp2_iter/iter_${it}
done

# 4 multi images
mkdir -p exp4_multi
for img in images/lena.png images/baboon.png images/cameraman.png; do
  base=$(basename "$img" .png)
  python3 image/demo/demo3.py --img "$img" --cr 0.5 --lam 0.01 --iter 120 --out "exp4_multi/${base}"
done

# 5 size
mkdir -p exp5_size
for img in images/img256.png images/img512.png images/img1024.png; do
  base=$(basename "$img" .png)
  python3 image/demo/demo3.py --img "$img" --cr 0.5 --lam 0.01 --iter 120 --out "exp5_size/${base}"
done

# 6 key test
mkdir -p exp6_keytest
for i in $(seq 1 20); do
  python3 image/demo/demo3.py --img 1.png --cr 0.5 --lam 0.01 --iter 120 \
    --key "key_${i}_$(python3 - <<'PY'
import time,random
print(int(time.time()*1e9) ^ random.getrandbits(32))
PY
)" \
    --out exp6_keytest/key_${i}
done

# 7 selective encryption (requires --enc_ratio)
mkdir -p exp7_selenc
for r in 0.25 0.5 0.75 1.0; do
  python3 image/demo/demo3.py --img 1.png --cr 0.5 --lam 0.01 --iter 120 \
    --enc_ratio $r --out exp7_selenc/r${r}
done

echo "ALL DONE."
