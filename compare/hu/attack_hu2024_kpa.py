#!/usr/bin/env python3
"""
attack_hu2024_kpa.py  ——  Hu 2024 已知明文攻击（KPA）完整版
==============================================================
攻击路径（密钥已知 / 论文给出的默认参数）：

  cipher_1.npy
    → invert_diffusion_correct()   逆 HCM 树形扩散 → q_flat (uint8)
    → q_blocks[B, m]
    → dequantize()                 uint8 → float 测量值 y
    → build_phi()                  LTMM 序列 → Hadamard 行索引 → Phi
    → omp() / fista()              CS 重建 → coef_blocks
    → invert_arnold()              逆 Arnold
    → idwt2_haar()                 逆 DWT → 重建图像

设计原则
--------
- 逆扩散使用正确的 forward_v2 对应的逆操作
- 量化反标定：全局 min/max 存储在密文 meta 中，或者通过 Phi 估算
- CS 重建用 OMP（稀疏度已知时最优）和 FISTA（稀疏度未知时）
- 所有参数有合理默认值（来自 Hu 2024 论文）
"""

import argparse
import math
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# 小波
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

def pad_to_square(mat):
    h,w=mat.shape; n=max(h,w)
    return mat if h==n and w==n else np.pad(mat,((0,n-h),(0,n-w)),mode="edge")


# ─────────────────────────────────────────────────────────────
# Arnold
# ─────────────────────────────────────────────────────────────

def arnold_transform(mat,iters):
    n=mat.shape[0]; out=mat.copy()
    xi,yi=np.meshgrid(np.arange(n),np.arange(n),indexing="ij")
    for _ in range(iters):
        nxt=np.zeros_like(out); nxt[(xi+yi)%n,(xi+2*yi)%n]=out[xi,yi]; out=nxt
    return out

def invert_arnold(mat,iters):
    n=mat.shape[0]; out=mat.copy()
    xi,yi=np.meshgrid(np.arange(n),np.arange(n),indexing="ij")
    for _ in range(iters):
        nxt=np.zeros_like(out); nxt[(2*xi-yi)%n,(-xi+yi)%n]=out[xi,yi]; out=nxt
    return out

def blats_sparsify(mat,block=32,alpha=0.5):
    h,w=mat.shape; out=mat.copy()
    for i in range(h//block):
        for j in range(w//block):
            blk=out[i*block:(i+1)*block,j*block:(j+1)*block]
            blk[np.abs(blk)<np.mean(np.abs(blk))+alpha*np.std(blk)]=0.
            out[i*block:(i+1)*block,j*block:(j+1)*block]=blk
    return out


# ─────────────────────────────────────────────────────────────
# 混沌序列
# ─────────────────────────────────────────────────────────────

def ltmm_sequence(n,x0,y0,a0,b0,burn=1000):
    x,y=float(x0),float(y0); out=np.zeros(n)
    for t in range(n+burn):
        x=(a0*x*(1-x)+b0*y)%1.; y=(a0*y*(1-y)+b0*x)%1.
        if t>=burn: out[t-burn]=x
    return out

def hcm_sequences(n,x0,y0,z0,a1,b1,t1,t2,t3,t4,t5,burn=1000):
    x,y,z=float(x0),float(y0),float(z0)
    xs=np.zeros(n); ys=np.zeros(n)
    for t in range(n+burn):
        x=(a1*x*(1-x)+t1*y+t2*z)%1.; y=(a1*y*(1-y)+t3*z+t4*x)%1.
        z=(b1*z*(1-z)+t5*x)%1.
        if t>=burn: xs[t-burn]=x; ys[t-burn]=y
    return xs,ys


# ─────────────────────────────────────────────────────────────
# 正确的逆扩散（对应 Hu 2024 forward_v2）
# ─────────────────────────────────────────────────────────────

def invert_diffusion_correct(cipher,hcm_x,hcm_y,f0):
    """
    逆 HCM 树形扩散。
    正向：前向树CBC → 后向子节点 XOR（用 cipher 自身子节点）
    逆向：逆后向 → 逆前向
    """
    N=cipher.reshape(-1).size; cf=cipher.reshape(-1).astype(np.uint8)
    k1=np.floor(hcm_x[:N]*256).astype(np.uint8)
    k2=np.floor(hcm_y[:N]*256).astype(np.uint8)
    # 逆后向：c1[i] = cipher[i] ^ k2[i] ^ (cipher[l] ^ cipher[r])
    c1=np.zeros(N,dtype=np.uint8)
    for i in range(N):
        l,r=2*i+1,2*i+2; child=np.uint8(0)
        if l<N: child^=cf[l]
        if r<N: child^=cf[r]
        c1[i]=cf[i]^k2[i]^child
    # 逆前向：q[i] = c1[i] ^ k1[i] ^ c1[parent]
    q=np.zeros(N,dtype=np.uint8)
    q[0]=c1[0]^k1[0]^np.uint8(f0)
    for i in range(1,N):
        q[i]=c1[i]^k1[i]^c1[(i-1)//2]
    return q


# ─────────────────────────────────────────────────────────────
# Hadamard + 感知矩阵
# ─────────────────────────────────────────────────────────────

def hadamard(n):
    H=np.array([[1.0]])
    while H.shape[0]<n: H=np.block([[H,H],[H,-H]])
    return H

def build_phi(n,m,ltmm):
    """用 LTMM 序列生成 Phi = H[idx,:]（行选取，无正交化）"""
    H=hadamard(n)
    idx=np.argsort(ltmm)[:m]
    return H[idx,:].astype(np.float64), idx


# ─────────────────────────────────────────────────────────────
# CS 重建
# ─────────────────────────────────────────────────────────────

def omp(A,y,sparsity):
    """OMP——稀疏度已知时最优"""
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
    """FISTA-L1——稀疏度未知时"""
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

def cs_reconstruct(Phi,y,method="fista",sparsity=None,lam=1e-4,max_iter=500):
    """统一 CS 重建接口"""
    if method=="omp" and sparsity is not None:
        return omp(Phi,y,sparsity)
    return fista(Phi,y,lam=lam,max_iter=max_iter)


# ─────────────────────────────────────────────────────────────
# 反量化策略
# ─────────────────────────────────────────────────────────────

def dequantize_with_phi(q_block, Phi, x_init=None, iters=3):
    """
    迭代反量化：用 Phi 估算 y_min/y_ptp。
    步骤：
      1. 用 q_norm = q/255 做粗略 FISTA 重建 x_crude
      2. y_pred = Phi @ x_crude
      3. 用 y_pred 的 min/ptp 标定 q → y_est
      4. 重复
    """
    q_norm=q_block.astype(np.float64)/255.0
    if x_init is not None:
        x_crude=x_init.copy()
    else:
        x_crude=fista(Phi,q_norm,lam=1e-4,max_iter=100)

    for _ in range(iters):
        y_pred=Phi@x_crude
        y_min=y_pred.min(); y_ptp=np.ptp(y_pred)
        if y_ptp<1e-10: y_ptp=1.0
        y_est=q_norm*y_ptp+y_min
        x_crude=fista(Phi,y_est,lam=1e-4,max_iter=200)
    return y_est, x_crude


def dequantize_global(q_block, y_min, y_ptp):
    """直接反量化（需要知道全局 y_min/y_ptp）"""
    return q_block.astype(np.float64)/255.0*y_ptp+y_min


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

def attack_kpa(img_path, cipher_path,
               block=32, cr=0.5,
               arnold_iters=5, blats_alpha=0.5,
               hcm_params=None, ltmm_params=None,
               f0=123,
               cs_method="fista", sparsity=None,
               cs_lam=1e-4, cs_iter=500,
               deq_iters=3,
               verbose=True):
    """
    完整 KPA 攻击。
    hcm_params: dict(x0,y0,z0,a1,b1,t1,t2,t3,t4,t5)
    ltmm_params: dict(x0,y0,a0,b0)
    """
    from PIL import Image as PILImage
    img=np.asarray(PILImage.open(img_path).convert("L"),dtype=np.float64)
    cipher=np.load(cipher_path)
    H_orig,W_orig=img.shape

    # 正向预处理（与加密端相同）
    img01=(img-img.min())/(img.max()-img.min()+1e-12)
    ll,lh,hl,hh=dwt2_haar(img01)
    coef=pad_to_square(pack_subbands(ll,lh,hl,hh))
    N_sq=coef.shape[0]
    coef=arnold_transform(coef,arnold_iters)
    coef=blats_sparsify(coef,block=block,alpha=blats_alpha)

    h,w=coef.shape; hb,wb=h//block,w//block
    n=block*block; m=max(1,min(int(round(cr*n)),n))
    total=hb*wb

    x_blocks=np.array([
        coef[(b//wb)*block:(b//wb+1)*block,
             (b%wb)*block:(b%wb+1)*block].reshape(n)
        for b in range(total)])

    # 逆扩散
    needed=max(4096,cipher.size)
    hcm_x,hcm_y=hcm_sequences(needed,**hcm_params)
    q_flat=invert_diffusion_correct(cipher,hcm_x,hcm_y,f0)
    q_blocks=q_flat[:total*m].reshape(total,m)

    # 构建 Phi（密钥已知）
    ltmm=ltmm_sequence(max(4096,total*n),**ltmm_params)

    coef_rec=np.zeros((h,w),dtype=np.float64)
    for b in range(total):
        i_b,j_b=b//wb,b%wb
        seg=ltmm[b*n:(b+1)*n]
        Phi,idx=build_phi(n,m,seg)

        # 反量化（迭代估算）
        y_est,x_crude=dequantize_with_phi(q_blocks[b],Phi,iters=deq_iters)

        # CS 重建
        x_hat=cs_reconstruct(Phi,y_est,method=cs_method,
                               sparsity=sparsity,lam=cs_lam,max_iter=cs_iter)
        coef_rec[i_b*block:(i_b+1)*block,j_b*block:(j_b+1)*block]=x_hat.reshape(block,block)

        if verbose and b%max(1,total//5)==0:
            mse_b=np.mean((x_blocks[b]-x_hat)**2)
            print(f"  块 {b}/{total}  block_MSE={mse_b:.6f}")

    # 逆 Arnold
    cf_full=np.zeros((N_sq,N_sq)); cf_full[:hb*block,:wb*block]=coef_rec
    cf_full=invert_arnold(cf_full,arnold_iters)

    # 逆 DWT
    ll_r,lh_r,hl_r,hh_r=unpack_subbands(cf_full)
    img_rec=idwt2_haar(ll_r,lh_r,hl_r,hh_r)[:H_orig,:W_orig]
    img_rec=np.clip((img_rec-img_rec.min())/(img_rec.max()-img_rec.min()+1e-12)*255,
                    0,255).astype(np.uint8)

    p=psnr(img.astype(np.uint8),img_rec)
    ss=ssim(img.astype(np.uint8),img_rec)
    if verbose:
        print(f"\nPSNR={p:.2f}dB  SSIM={ss:.4f}")
    return img_rec,p,ss


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    ap=argparse.ArgumentParser(description="Hu 2024 KPA 攻击（完整重建）")
    ap.add_argument("--img",          required=True)
    ap.add_argument("--cipher_npy",   required=True)
    ap.add_argument("--out_img",      default="")
    ap.add_argument("--block",        type=int,   default=32)
    ap.add_argument("--cr",           type=float, default=0.5)
    ap.add_argument("--arnold",       type=int,   default=5)
    ap.add_argument("--blats_alpha",  type=float, default=0.5)
    ap.add_argument("--f0",           type=int,   default=123)
    ap.add_argument("--cs_method",    default="fista", choices=["fista","omp"])
    ap.add_argument("--sparsity",     type=int,   default=None)
    ap.add_argument("--cs_lam",       type=float, default=1e-4)
    ap.add_argument("--cs_iter",      type=int,   default=500)
    ap.add_argument("--deq_iters",    type=int,   default=3,
                    help="反量化迭代次数")
    # HCM 参数
    ap.add_argument("--x1", type=float, default=0.23)
    ap.add_argument("--y1", type=float, default=0.29)
    ap.add_argument("--z1", type=float, default=0.31)
    ap.add_argument("--a1", type=float, default=3.97)
    ap.add_argument("--b1", type=float, default=3.83)
    ap.add_argument("--t1", type=float, default=0.11)
    ap.add_argument("--t2", type=float, default=0.07)
    ap.add_argument("--t3", type=float, default=0.09)
    ap.add_argument("--t4", type=float, default=0.13)
    ap.add_argument("--t5", type=float, default=0.05)
    # LTMM 参数
    ap.add_argument("--x0", type=float, default=0.41)
    ap.add_argument("--y0", type=float, default=0.37)
    ap.add_argument("--a0", type=float, default=3.91)
    ap.add_argument("--b0", type=float, default=0.33)
    args=ap.parse_args()

    hcm_p=dict(x0=args.x1,y0=args.y1,z0=args.z1,
               a1=args.a1,b1=args.b1,
               t1=args.t1,t2=args.t2,t3=args.t3,t4=args.t4,t5=args.t5)
    ltmm_p=dict(x0=args.x0,y0=args.y0,a0=args.a0,b0=args.b0)

    img_rec,p,ss=attack_kpa(
        args.img, args.cipher_npy,
        block=args.block, cr=args.cr,
        arnold_iters=args.arnold, blats_alpha=args.blats_alpha,
        hcm_params=hcm_p, ltmm_params=ltmm_p, f0=args.f0,
        cs_method=args.cs_method, sparsity=args.sparsity,
        cs_lam=args.cs_lam, cs_iter=args.cs_iter,
        deq_iters=args.deq_iters, verbose=True)

    if args.out_img:
        from PIL import Image as PILImage
        PILImage.fromarray(img_rec).save(args.out_img)
        print(f"重建图像: {args.out_img}")


if __name__ == "__main__":
    main()
