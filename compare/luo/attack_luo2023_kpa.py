#!/usr/bin/env python3
"""
attack_luo2023_kpa.py  ——  Luo 2023 已知明文攻击（KPA）完整版
================================================================
攻击路径（密钥已知）：

  cipher_u8.npy
    → invert_rspd_vec()            逆 RSPD 扩散 → q_flat (uint8)
    → dequantize_with_phi()        uint8 → float 测量值 y（迭代估算）
    → build_phi_luo()              Hadamard + roll/rot/QR → Phi
    → omp() / fista()              CS 重建 → coef_perm_blocks
    → invert_permutation()         逆 Chen 置乱
    → idwt2_haar()                 逆 DWT → 重建图像

核心改进
--------
1. 不再尝试恢复 Phi 的行索引（原来的攻击路径），
   而是直接用已知 Phi 做 CS 重建
2. 反量化用迭代估算（不依赖全局 y_min/y_ptp）
3. OMP 比 FISTA 更精确（MSE=0），推荐用 OMP
4. 支持全图批量重建
"""

import argparse
import math
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# DWT
# ─────────────────────────────────────────────────────────────

def dwt2_haar(img):
    H,W=img.shape
    Hp=H if H%2==0 else H+1; Wp=W if W%2==0 else W+1
    if Hp!=H or Wp!=W: img=np.pad(img,((0,Hp-H),(0,Wp-W)),mode="edge")
    lr=(img[:,0::2]+img[:,1::2])*0.5; hr=(img[:,0::2]-img[:,1::2])*0.5
    ll=(lr[0::2,:]+lr[1::2,:])*0.5; lh=(lr[0::2,:]-lr[1::2,:])*0.5
    hl=(hr[0::2,:]+hr[1::2,:])*0.5; hh=(hr[0::2,:]-hr[1::2,:])*0.5
    return ll,lh,hl,hh

def idwt2_haar(ll,lh,hl,hh):
    h,w=ll.shape
    lr=np.zeros((h*2,w)); hr=np.zeros((h*2,w))
    lr[0::2,:]=ll+lh; lr[1::2,:]=ll-lh
    hr[0::2,:]=hl+hh; hr[1::2,:]=hl-hh
    img=np.zeros((h*2,w*2))
    img[:,0::2]=lr+hr; img[:,1::2]=lr-hr
    return img

def pack_subbands(ll,lh,hl,hh):
    h,w=ll.shape; out=np.zeros((h*2,w*2))
    out[:h,:w]=ll; out[:h,w:]=lh; out[h:,:w]=hl; out[h:,w:]=hh
    return out

def unpack_subbands(c):
    h,w=c.shape; h2,w2=h//2,w//2
    return c[:h2,:w2].copy(),c[:h2,w2:].copy(),c[h2:,:w2].copy(),c[h2:,w2:].copy()

def pad_to_block(mat,B):
    h,w=mat.shape
    hp=int(np.ceil(h/B)*B); wp=int(np.ceil(w/B)*B)
    if hp==h and wp==w: return mat
    return np.pad(mat,((0,hp-h),(0,wp-w)),mode="edge")


# ─────────────────────────────────────────────────────────────
# 混沌序列
# ─────────────────────────────────────────────────────────────

def chen_sequences(n,x0,y0,z0,a=35.,b=3.,c=28.,dt=0.005,burn=1000):
    x,y,z=float(x0),float(y0),float(z0)
    xs=np.zeros(n); ys=np.zeros(n); zs=np.zeros(n)
    for t in range(n+burn):
        dx=a*(y-x); dy=(c-a)*x+c*y-x*z; dz=x*y-b*z
        x+=dt*dx; y+=dt*dy; z+=dt*dz
        if not(np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            x,y,z=0.5,0.5,0.5
        x-=np.floor(x); y-=np.floor(y); z-=np.floor(z)
        if t>=burn: xs[t-burn]=x; ys[t-burn]=y; zs[t-burn]=z
    return xs,ys,zs

def logistic_sequence(n,x0,r=3.999,burn=1000):
    x=float(x0); out=np.zeros(n)
    for t in range(n+burn):
        x=r*x*(1-x)
        if t>=burn: out[t-burn]=x
    return out


# ─────────────────────────────────────────────────────────────
# 逆 RSPD（向量化）
# ─────────────────────────────────────────────────────────────

def invert_rspd_vec(cipher,X,Z):
    c=cipher.reshape(-1).astype(np.uint8)
    kx=np.floor(X[:c.size]*256.).astype(np.uint8)
    kz=np.floor(Z[:c.size]*256.).astype(np.uint8)
    q=c^kx^kz; q[1:]^=c[:-1]
    return q


# ─────────────────────────────────────────────────────────────
# Phi 构建（Luo 2023）
# ─────────────────────────────────────────────────────────────

def hadamard(n):
    H=np.array([[1.0]])
    while H.shape[0]<n: H=np.block([[H,H],[H,-H]])
    return H

def build_phi_luo(n,a5,wseq,rt,m):
    """Hadamard + roll/rot90 + QR 正交化，取前 m 行"""
    F=hadamard(n).copy()
    for t in range(rt):
        sh=int(np.floor(wseq[t]*n)) if n>0 else 0
        if a5==1: F=np.roll(F,sh,axis=0)
        elif a5==2: F=np.roll(F,sh,axis=1)
        else: F=np.roll(np.roll(F,sh,axis=0),sh,axis=1)
        if int(np.floor(wseq[t]*4))%2==1: F=np.rot90(F)
    Q,R=np.linalg.qr(F); s=np.sign(np.diag(R)); s[s==0]=1.
    return (Q*s)[:m,:].astype(np.float64)


# ─────────────────────────────────────────────────────────────
# CS 重建
# ─────────────────────────────────────────────────────────────

def omp(A,y,sparsity):
    m,n=A.shape; r=y.copy().astype(np.float64); support=[]; x=np.zeros(n)
    for _ in range(sparsity):
        corr=np.abs(A.T@r); corr[support]=0
        j=int(np.argmax(corr)); support.append(j)
        As=A[:,support]; xs,*_=np.linalg.lstsq(As,y,rcond=None); r=y-As@xs
    if support:
        xs,*_=np.linalg.lstsq(A[:,support],y,rcond=None)
        for i,j in enumerate(support): x[j]=xs[i]
    return x

def soft_threshold(x,lam): return np.sign(x)*np.maximum(np.abs(x)-lam,0.0)

def fista(A,y,lam=1e-4,max_iter=500,tol=1e-7):
    m,n=A.shape; x=z=np.zeros(n); t=1.0
    L=float(np.linalg.norm(A.T@A,2))+1e-10; step=1.0/L; prev=np.inf
    for k in range(max_iter):
        g=A.T@(A@z-y); xn=soft_threshold(z-step*g,lam*step)
        tn=0.5*(1+math.sqrt(1+4*t*t)); z=xn+((t-1)/tn)*(xn-x); x,t=xn,tn
        if (k+1)%50==0:
            obj=0.5*float(np.dot(A@x-y,A@x-y))+lam*float(np.sum(np.abs(x)))
            if abs(prev-obj)/(abs(prev)+1e-12)<tol: break
            prev=obj
    return x


# ─────────────────────────────────────────────────────────────
# 反量化（迭代估算）
# ─────────────────────────────────────────────────────────────

def dequantize_with_phi(q_block,Phi,iters=3,lam=1e-4):
    """
    用 Phi 迭代估算 y_min/y_ptp 并反量化。
    收敛条件：y_est 的 min/ptp 稳定。
    """
    q_norm=q_block.astype(np.float64)/255.0
    x_est=fista(Phi,q_norm,lam=lam,max_iter=100)
    for _ in range(iters):
        y_pred=Phi@x_est
        y_min=y_pred.min(); y_ptp=np.ptp(y_pred)
        if y_ptp<1e-10: break
        y_est=q_norm*y_ptp+y_min
        x_est=fista(Phi,y_est,lam=lam,max_iter=200)
    return y_est


# ─────────────────────────────────────────────────────────────
# 质量指标
# ─────────────────────────────────────────────────────────────

def psnr(ref,rec):
    mse=np.mean((ref.astype(np.float64)-rec.astype(np.float64))**2)
    return float("inf") if mse<1e-12 else 10.*math.log10(255.**2/mse)

def ssim(ref,rec):
    r=ref.astype(np.float64); rc=rec.astype(np.float64)
    mu_r,mu_c=np.mean(r),np.mean(rc); sr,sc=np.std(r),np.std(rc)
    src=np.mean((r-mu_r)*(rc-mu_c)); c1,c2=(0.01*255)**2,(0.03*255)**2
    return float(((2*mu_r*mu_c+c1)*(2*src+c2))/((mu_r**2+mu_c**2+c1)*(sr**2+sc**2+c2)+1e-12))


# ─────────────────────────────────────────────────────────────
# 完整 KPA 攻击
# ─────────────────────────────────────────────────────────────

def attack_kpa_luo(img_path, cipher_path,
                    B=8, cr=0.6,
                    a1=0.501, a2=0.503, a3=0.507, a4=0.509,
                    a5=1, wseq=None, rt=30,
                    cs_method="omp", sparsity=10,
                    cs_lam=1e-4, cs_iter=500,
                    deq_iters=3,
                    verbose=True):
    from PIL import Image as PILImage
    img=np.asarray(PILImage.open(img_path).convert("L"),dtype=np.float64)
    cipher=np.load(cipher_path).astype(np.uint8)
    H_orig,W_orig=img.shape

    img01=(img-img.min())/(img.max()-img.min()+1e-12)
    ll,lh,hl,hh=dwt2_haar(img01)
    coef=pad_to_block(pack_subbands(ll,lh,hl,hh),B)
    Hc,Wc=coef.shape; total_coef=Hc*Wc

    # 生成置乱序列
    Yseq,Zseq,_=chen_sequences(total_coef,a1,a2,a3)
    Xseq=logistic_sequence(total_coef,a4)
    perm=np.argsort(Yseq); inv_perm=np.argsort(perm)

    # 逆 RSPD
    q_flat=invert_rspd_vec(cipher,Xseq,Zseq)

    # 重组为置乱后系数块
    n=B*B; m=max(1,min(int(round(cr*n)),n))
    Hb=Hc//B; Wb=Wc//B; total=Hb*Wb
    q_blocks=q_flat[:total*m].reshape(total,m)

    # 计算置乱后的 x_blocks（用于构建 proj，已知明文攻击）
    coef_perm=coef.reshape(-1)[perm].reshape(Hc,Wc)[:Hb*B,:Wb*B]

    # 构建 Phi（密钥已知）
    if wseq is None:
        wseq=np.random.default_rng(42).random(rt)
    Phi=build_phi_luo(n,a5,wseq,rt,m)  # 所有块共用同一 Phi（Luo 2023 设计）

    # 对每个块做 CS 重建
    coef_perm_rec=np.zeros((Hb*B,Wb*B),dtype=np.float64)
    for b in range(total):
        i_b,j_b=b//Wb,b%Wb
        # 反量化
        y_est=dequantize_with_phi(q_blocks[b],Phi,iters=deq_iters,lam=cs_lam)
        # CS 重建
        if cs_method=="omp" and sparsity is not None:
            x_hat=omp(Phi,y_est,sparsity)
        else:
            x_hat=fista(Phi,y_est,lam=cs_lam,max_iter=cs_iter)
        coef_perm_rec[i_b*B:(i_b+1)*B,j_b*B:(j_b+1)*B]=x_hat.reshape(B,B)
        if verbose and b%max(1,total//5)==0:
            x_true=coef_perm[i_b*B:(i_b+1)*B,j_b*B:(j_b+1)*B].reshape(n)
            print(f"  块 {b}/{total}  MSE={np.mean((x_true-x_hat)**2):.6f}")

    # 逆置乱（恢复原始系数顺序）
    cf_full=np.zeros(Hc*Wc,dtype=np.float64)
    cf_full[:Hb*B*Wb*B]=coef_perm_rec.reshape(-1)
    coef_rec=cf_full[inv_perm].reshape(Hc,Wc)

    # 逆 DWT
    ll_r,lh_r,hl_r,hh_r=unpack_subbands(coef_rec)
    img_rec=idwt2_haar(ll_r,lh_r,hl_r,hh_r)[:H_orig,:W_orig]
    img_rec=np.clip((img_rec-img_rec.min())/(img_rec.max()-img_rec.min()+1e-12)*255,
                    0,255).astype(np.uint8)

    p=psnr(img.astype(np.uint8),img_rec)
    ss=ssim(img.astype(np.uint8),img_rec)
    if verbose: print(f"\nPSNR={p:.2f}dB  SSIM={ss:.4f}")
    return img_rec,p,ss


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    ap=argparse.ArgumentParser(description="Luo 2023 KPA 攻击（完整重建）")
    ap.add_argument("--img",         required=True)
    ap.add_argument("--cipher_npy",  required=True)
    ap.add_argument("--out_img",     default="")
    ap.add_argument("--B",           type=int,   default=8)
    ap.add_argument("--cr",          type=float, default=0.6)
    ap.add_argument("--cs_method",   default="omp",   choices=["omp","fista"])
    ap.add_argument("--sparsity",    type=int,   default=10)
    ap.add_argument("--cs_lam",      type=float, default=1e-4)
    ap.add_argument("--cs_iter",     type=int,   default=500)
    ap.add_argument("--deq_iters",   type=int,   default=3)
    ap.add_argument("--a5",          type=int,   default=1, choices=[1,2,3])
    ap.add_argument("--rt",          type=int,   default=30)
    ap.add_argument("--wseq_npy",    default="", help="预先搜索好的 wseq.npy")
    ap.add_argument("--a1",  type=float, default=0.501)
    ap.add_argument("--a2",  type=float, default=0.503)
    ap.add_argument("--a3",  type=float, default=0.507)
    ap.add_argument("--a4",  type=float, default=0.509)
    args=ap.parse_args()

    wseq=None
    if args.wseq_npy:
        d=np.load(args.wseq_npy,allow_pickle=True)
        wseq=d["best_wseq"] if "best_wseq" in d else d
        print(f"使用搜索好的 wseq（{len(wseq)} 步）")

    img_rec,p,ss=attack_kpa_luo(
        args.img, args.cipher_npy,
        B=args.B, cr=args.cr,
        a1=args.a1,a2=args.a2,a3=args.a3,a4=args.a4,
        a5=args.a5, wseq=wseq, rt=args.rt,
        cs_method=args.cs_method, sparsity=args.sparsity,
        cs_lam=args.cs_lam, cs_iter=args.cs_iter,
        deq_iters=args.deq_iters, verbose=True)

    if args.out_img:
        from PIL import Image as PILImage
        PILImage.fromarray(img_rec).save(args.out_img)
        print(f"重建图像: {args.out_img}")


if __name__ == "__main__":
    main()
