import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chisquare
import time

# ==========================================
# 1. 系统定义: nD-SPCMM (支持任意维度扩展)
# ==========================================
def nD_spcmm(n, steps, x0, a, b, w, c, theta=1.0):
    """
    n维正弦-多项式复合模映射
    级联构造保证雅可比矩阵为三角阵，LE解析可控。
    """
    x_history = np.zeros((steps, n))
    x_history[0] = x0
    for i in range(steps - 1):
        curr = x_history[i]
        nxt = np.zeros(n)
        for k in range(n-1, -1, -1): # 级联依赖逻辑
            term = a[k] * np.sin(b[k] * curr[k])
            coupling = np.sum(w[k+1:] * (curr[k+1:]**2)) if k < n-1 else 0
            nxt[k] = (term + coupling + c) % theta
        x_history[i+1] = nxt
    return x_history

def calculate_le_analytical(x_history, a, b):
    """解析计算LE (基于雅可比矩阵对角线偏导)"""
    le = []
    eps = 1e-15
    for k in range(x_history.shape[1]):
        vals = np.abs(a[k] * b[k] * np.cos(b[k] * x_history[:, k]))
        le.append(np.mean(np.log(vals + eps)))
    return le

# ==========================================
# 2. 基础仿真参数
# ==========================================
n_dim = 3
steps_main = 50000
x_init = np.array([0.123, 0.456, 0.789])
a_base = np.array([3.1, 3.3, 3.5])
b_base = np.array([10.0, 11.0, 12.0])
w_base = np.array([0, 0.6, 0.5])
c_base = 0.7
data = nD_spcmm(n_dim, steps_main, x_init, a_base, b_base, w_base, c_base)

# ==========================================
# 3. 独立绘图模块 (按论文标准保存)
# ==========================================

# --- 图 1: 3D 相图 ---
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(data[:, 0], data[:, 1], data[:, 2], s=0.2, c=data[:, 2], cmap='viridis', alpha=0.5)
ax1.set_title('Test 1: 3D Phase Portrait')
plt.savefig('1_phase_portrait_3d.png', dpi=300)
plt.close()

# --- 图 2: PDF 均匀性分析 ---
observed_freq, _ = np.histogram(data[:, 0], bins=50, range=(0, 1))
expected_freq = np.full(50, len(data) / 50)
chi2_stat, p_val = chisquare(observed_freq, f_exp=expected_freq)
plt.figure(figsize=(10, 6))
plt.hist(data[:, 0], bins=50, range=(0, 1), density=True, color='skyblue', edgecolor='black', alpha=0.7)
plt.axhline(1.0, color='red', linestyle='--')
plt.title(f'Test 2: PDF (Chi2 P-value={p_val:.4e})')
plt.savefig('2_pdf_analysis.png', dpi=300)
plt.close()

# --- 图 3: 动力学退化 (精度敏感性) ---
data_f32 = nD_spcmm(n_dim, 300, x_init.astype(np.float32), a_base.astype(np.float32), b_base.astype(np.float32), w_base.astype(np.float32), c_base)
diff = np.sqrt(np.sum((data[:300] - data_f32)**2, axis=1))
plt.figure(figsize=(10, 6))
plt.plot(diff, color='orange')
plt.yscale('log')
plt.title('Test 3: Orbit Divergence (Float64 vs Float32)')
plt.savefig('3_precision_degradation.png', dpi=300)
plt.close()

# --- 图 4: 高清晰分叉图 (低频观察机理) ---
b_low = 2.0
a_range = np.linspace(0.1, 4.0, 800)
plt.figure(figsize=(12, 7))
for av in a_range:
    traj = nD_spcmm(n_dim, 500, x_init, [av, 3.3, 3.5], [b_low]*3, w_base, c_base)
    plt.plot([av]*150, traj[-150:, 0], ',k', markersize=1, alpha=0.5)
plt.title(f'Test 4: Bifurcation Diagram (b={b_low})')
plt.savefig('4_bifurcation_diagram.png', dpi=300)
plt.close()

# --- 图 5: 一维 LE 曲线 ---
a_range_le = np.linspace(0.5, 5.0, 300)
le_results = []
for av in a_range_le:
    le_data = nD_spcmm(n_dim, 1000, x_init, [av, 3.3, 3.5], b_base, w_base, c_base)
    le_results.append(calculate_le_analytical(le_data[200:], [av, 3.3, 3.5], b_base))
le_results = np.array(le_results)
plt.figure(figsize=(10, 6))
plt.plot(a_range_le, le_results[:, 0], label='LE1'); plt.plot(a_range_le, le_results[:, 1], label='LE2'); plt.plot(a_range_le, le_results[:, 2], label='LE3')
plt.axhline(0, color='red', linestyle='--'); plt.legend()
plt.title('Test 5: Lyapunov Exponents Analysis')
plt.savefig('5_lyapunov_analysis.png', dpi=300)
plt.close()

# --- 图 6: 二维参数空间 MLE 热力图 (新增) ---
res = 100 # 分辨率 100x100
a1_v = np.linspace(1.0, 5.0, res)
a2_v = np.linspace(1.0, 5.0, res)
mle_map = np.zeros((res, res))
print(f"开始生成参数空间分析图 (预计耗时较长)...")
for i, a1 in enumerate(a1_v):
    for j, a2 in enumerate(a2_v):
        temp_a = [a1, a2, 3.5]
        t_data = nD_spcmm(n_dim, 600, x_init, temp_a, b_base, w_base, c_base)
        mle_map[j, i] = max(calculate_le_analytical(t_data[100:], temp_a, b_base))
plt.figure(figsize=(10, 8))
plt.contourf(a1_v, a2_v, mle_map, levels=50, cmap='jet')
plt.colorbar(label='MLE Value')
plt.title('Test 6: 2D Parameter Space Analysis (a1 vs a2)')
plt.xlabel('a1'); plt.ylabel('a2')
plt.savefig('6_parameter_space_mle.png', dpi=300)
plt.close()

print("所有动力学测试已完成，图片已保存在当前目录下。")