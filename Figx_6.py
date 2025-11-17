#%%
# Discrete-time simulation for x[t+1] = A x[t] + B u[t]
# A includes a 10 Hz damped mode (ζ = 0.6). We sweep input frequency ω (rad/sample)
# and plot tr(Q^{-1}) where Q is the sample covariance of x in steady state.

import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from scipy.spatial.transform import Rotation as R
import os

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.default'] = 'regular'
# --------------------
# System definition
# --------------------
fs = 100.0        # sampling rate [Hz]  (edit as needed)
Ts = 1.0 / fs

Time = 10
T_total = int(fs*Time)          # total steps
T_burn  = T_total*4//5            # discard initial transient

# ===== 例 =====
# 6次元: 複素ペア(3Hz, r=0.95, dt=0.01), 複素ペア(5Hz, r=1.0, dt=0.01), 実固有値(0.8, -0.5)
# rとf_hzを配列で指定

#Simulation condition
trial_num = 1


def block_discrete(r, f_hz):
    """離散時間: 固有値 r e^{± i 2π f dt} の2x2ブロック"""
    dt = Ts
    theta = 2*np.pi*f_hz*dt
    return np.array([[r*np.cos(theta), -r*np.sin(theta)],
                     [r*np.sin(theta),  r*np.cos(theta)]])

def make_block_diag(blocks):
    n = sum(B.shape[0] for B in blocks)
    A = np.zeros((n, n))
    i = 0
    for B in blocks:
        k = B.shape[0]
        A[i:i+k, i:i+k] = B
        i += k
    return A

def random_similarity(A, seed=None):
    """ブロック対角を一般形に混ぜたいとき"""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    P = rng.normal(size=(n, n))
    while np.linalg.cond(P) > 1e6:
        P = rng.normal(size=(n, n))
    return P @ A @ np.linalg.inv(P)

# --------------------
# Simulation settings
# --------------------
assert T_burn < T_total
t_idx = np.arange(T_total)

r_list = [0.9, 0.94]
hz_list = [10, 20]

blocks = [block_discrete(r, f_hz) for r, f_hz in zip(r_list, hz_list)]
A = make_block_diag(blocks)

eigvals_A = np.linalg.eigvals(A)
for i, eig in enumerate(eigvals_A):
    r = np.abs(eig)
    theta = np.angle(eig)
    freq_hz = theta / (2 * np.pi * Ts)
    print(f"eig {i}: r={r:.4f}, 周波数={freq_hz:.4f} Hz, 減衰率={r:.4f}")

A = random_similarity(A, seed=0)  # 一般形に変換

eigvals_A = np.linalg.eigvals(A)
for i, eig in enumerate(eigvals_A):
    r = np.abs(eig)
    theta = np.angle(eig)
    freq_hz = theta / (2 * np.pi * Ts)
    print(f"eig {i}: r={r:.4f}, 周波数={freq_hz:.4f} Hz, 減衰率={r:.4f}")
A_both = A
for i, eig in enumerate(eigvals_A):
    x = np.log(eig)
    freq_hz = x.imag / (2 * np.pi * Ts)
    print(f"eig {i}: e^(x), x = {x.real:.4f} + {x.imag:.4f}j (real: {x.real:.4f}, freq: {freq_hz:.4f} Hz)")
B = np.eye(A.shape[0])
B_both = B

n = A.shape[0]

plt.figure(figsize=(5, 5))
im = plt.imshow(A, cmap='viridis')
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        plt.text(j, i, f"{A[i, j]:.3f}", ha='center', va='center',
                 color='white' if np.abs(A[i, j]) < np.max(np.abs(A))/2 else 'black', fontsize=20)
plt.tight_layout()
plt.xticks([], fontsize=20)
plt.yticks([], fontsize=20)
program_name = os.path.splitext(os.path.basename(__file__))[0]
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
filename = f"{program_name}_A_matrix.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()

#%%
freqs = np.arange(1, 31, 1)
ws = 2 * np.pi * freqs / fs

inv_lambda_both_2d = np.zeros((len(freqs), len(freqs)))  # (w1, w2, eig)

# 2つの入力: u[t] = cos(w1*t) + cos(w2*t)
def simulate_cov_for_omega2(A, B, w1, w2, T_total, T_burn, seed=0):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    sigma = 1e-2

    x = np.zeros((n, T_total))
    noise = rng.normal(scale=sigma, size=(n, T_total-1))
    # 入力信号 u を事前に生成
    if w1 == 0 and w2 == 0:
        u = np.zeros(T_total-1)
    elif w1 == 0:
        u = np.cos(w2 * np.arange(T_total-1))
        u = u / np.sqrt(np.mean(u**2))
    elif w2 == 0:
        u = np.cos(w1 * np.arange(T_total-1))
        u = u / np.sqrt(np.mean(u**2))
    else:
        u = np.cos(w1 * np.arange(T_total-1)) + np.cos(w2 * np.arange(T_total-1)) 
        u = u / np.sqrt(np.mean(u**2))

    u = u * 0.1  # 入力の大きさを調整

    for t in range(T_total-1):
        x[:, t+1] = A @ x[:, t] + B[:, 0] * u[t] + noise[:, t]
    X_prev = x[:, T_burn-1:T_total-1]
    Qx = np.cov(X_prev)
    
    return Qx

for i, w1 in enumerate(ws):
    for j, w2 in enumerate(ws):
        Qx = simulate_cov_for_omega2(A_both, B_both, w1, w2, T_total, T_burn)
        print(w1, w2)
        eigvals = np.linalg.eigvalsh(Qx)
        inv_lambda_both_2d[i, j] = np.sum(1.0 / eigvals)

# %%
plt.figure(figsize=(8, 6))
im = plt.imshow(np.log(inv_lambda_both_2d), extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]],
                origin='lower', aspect='auto', cmap='viridis', norm=None)
plt.xlabel('Input frequency 1 [Hz]', fontsize=28)
plt.ylabel('Input frequency 2 [Hz]', fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
cbar = plt.colorbar(im)
cbar.set_label('log(sum(1/λ))', fontsize=24)
cbar.ax.tick_params(labelsize=22)
plt.tight_layout()
program_name = os.path.splitext(os.path.basename(__file__))[0]
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
filename = f"{program_name}_sum_inv_lambda_2d.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()

# 最小のインデックスを表示
min_idx = np.unravel_index(np.argmin(inv_lambda_both_2d), inv_lambda_both_2d.shape)
print(f"最小値のインデックス: {min_idx}, 周波数: ({freqs[min_idx[0]]:.2f} Hz, {freqs[min_idx[1]]:.2f} Hz)")

#%%
def plot_rotated_ellipsoid_with_half_axes(
    a, b, c,
    vecs,                             # 3x3（各列が主軸方向の単位ベクトル）
    center=(0.0, 0.0, 0.0),
    phi=0.0, theta=0.0, psi=0.0,      # 追加回転（ZYX）
    xlim=None, ylim=None, zlim=None,  # Noneなら自動調整
    n_u=60, n_v=30, ax=None
):
    vecs = np.asarray(vecs, dtype=float)
    assert vecs.shape == (3,3), "vecs は形状 (3,3) で、各列が主軸方向ベクトルである必要があります。"

    # まず主軸方向に合わせる回転（列が各主軸）
    rot = vecs

    # オイラー角が非ゼロなら、ワールド座標でさらに回す
    if not (phi == 0 and theta == 0 and psi == 0):
        rot_euler = R.from_euler('zyx', [phi, theta, psi]).as_matrix()
        rot = rot_euler @ rot

    # 少し調整
    axis = np.array([0, 0, 1], dtype=float)      # 例: z軸
    axis /= np.linalg.norm(axis)
    alpha = 90.0

    R_extra = R.from_rotvec(np.deg2rad(alpha) * axis).as_matrix()

    # ワールド回転（全体を地球基準でクルッと）
    rot = R_extra @ rot

    # パラメトリック楕円体（軸長 a,b,c を回転前座標で）
    u = np.linspace(0, 2*np.pi, n_u)
    v = np.linspace(0, np.pi,   n_v)
    X = a * np.outer(np.cos(u), np.sin(v))
    Y = b * np.outer(np.sin(u), np.sin(v))
    Z = c * np.outer(np.ones_like(u), np.cos(v))

    # 回転＋平行移動
    xyz = np.stack([X, Y, Z], axis=-1)                  # [nu, nv, 3]
    xyz_rot = np.einsum('ij,klj->kli', rot, xyz)        # rot を最後の次元に適用
    x_rot = xyz_rot[..., 0] + center[0]
    y_rot = xyz_rot[..., 1] + center[1]
    z_rot = xyz_rot[..., 2] + center[2]

    # 半軸ベクトル（中心→外周方向へ“半分の長さ”）
    half_axes = rot @ np.diag([a*0.97, b*0.97, c*0.97])    # 3x3（各列が矢印ベクトル）

    # 楕円体
    ax.plot_wireframe(x_rot, y_rot, z_rot, color='gray', linewidth=1, alpha=0.7)

    # 各半軸の矢印（線分。必要なら quiver に置換可）
    colors = ['r', 'g', 'b']  # x, y, z軸ごとに色を指定
    for i in range(3):
        ax.plot(
            [center[0], center[0] + half_axes[0, i]],
            [center[1], center[1] + half_axes[1, i]],
            [center[2], center[2] + half_axes[2, i]],
            linewidth=5, color=colors[i]
        )

    # 自動調整: xlim, ylim, zlimがNoneならデータ範囲＋余白で設定
    pad = 0.1  # 余白割合
    if xlim is None:
        xmin, xmax = np.min(x_rot), np.max(x_rot)
        dx = xmax - xmin
        xlim = (xmin - pad*dx, xmax + pad*dx)
    if ylim is None:
        ymin, ymax = np.min(y_rot), np.max(y_rot)
        dy = ymax - ymin
        ylim = (ymin - pad*dy, ymax + pad*dy)
    if zlim is None:
        zmin, zmax = np.min(z_rot), np.max(z_rot)
        dz = zmax - zmin
        zlim = (zmin - pad*dz, zmax + pad*dz)

    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
    ax.set_box_aspect([xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]])
    ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    ax.grid(True, linestyle='--', alpha=0.5)


def align_eigenvalues(vals, target_vecs, origin_vecs):
    """valsをtarget_vecsに合わせて並び替える"""
    aligned_vals = np.zeros_like(vals)
    used_indices = set()
    for i in range(target_vecs.shape[1]):
        target_vec = target_vecs[:, i]
        # origin_vecsの中でtarget_vecに最も近いものを探す
        similarities = [np.abs(np.dot(target_vec, origin_vecs[:, j])) if j not in used_indices else -1 for j in range(origin_vecs.shape[1])]
        best_j = np.argmax(similarities)
        aligned_vals[i] = vals[best_j]
        used_indices.add(best_j)
    return aligned_vals


# 0Hz入力のみ
Qx_0Hz = simulate_cov_for_omega2(A_both, B_both, 0, 0, T_total, T_burn)
eigvals_0Hz, eigvecs_0Hz = np.linalg.eigh(Qx_0Hz)

# 10Hz入力のみ
Qx_10Hz = simulate_cov_for_omega2(A_both, B_both, 2*np.pi*10/fs, 0, T_total, T_burn)
eigvals_10Hz, eigvecs_10Hz = np.linalg.eigh(Qx_10Hz)

# 20Hz入力のみ
Qx_20Hz = simulate_cov_for_omega2(A_both, B_both, 0, 2*np.pi*20/fs, T_total, T_burn)
eigvals_20Hz, eigvecs_20Hz = np.linalg.eigh(Qx_20Hz)

# 10Hz+20Hz入力
Qx_10_20Hz = simulate_cov_for_omega2(A_both, B_both, 2*np.pi*10/fs, 2*np.pi*20/fs, T_total, T_burn)
eigvals_10_20Hz, eigvecs_10_20Hz = np.linalg.eigh(Qx_10_20Hz)

Qx_10Hz_aligned = align_eigenvalues(eigvals_10Hz, eigvecs_0Hz, eigvecs_10Hz)
Qx_20Hz_aligned = align_eigenvalues(eigvals_20Hz, eigvecs_0Hz, eigvecs_20Hz)
Qx_10_20Hz_aligned = align_eigenvalues(eigvals_10_20Hz, eigvecs_0Hz, eigvecs_10_20Hz)

# 楕円体の主軸長を計算
def get_ellipsoid_axes(eigvals):
    eigvals_sorted = np.sort(eigvals)[::-1]
    axes = eigvals_sorted[:3]
    return axes

eigvals_list = [eigvals_0Hz, eigvals_10Hz, eigvals_20Hz, eigvals_10_20Hz]
titles = ["Passive", "10Hz", "20Hz", "10Hz + 20Hz"]

fig = plt.figure(figsize=(18, 9))
scale_eig_list = [2.5, 2.5, 2.5, 2.5]  # 例: 各入力ごとに異なるスケールを設定
lim=10

for i, eigvals in enumerate(eigvals_list):
    scale_eig = scale_eig_list[i]
    a, b, c = get_ellipsoid_axes(eigvals)
    vecs = eigvecs_10Hz[:, np.argsort(eigvals_10Hz)[::-1]][:3,:3]  # 10Hzの固有ベクトルに合わせる

    print(titles[i], "eigenvalues", a,b,c)

    # 上段: 通常の楕円体
    ax1 = fig.add_subplot(2, 4, i+1, projection='3d')
    plot_rotated_ellipsoid_with_half_axes(
        a=a*scale_eig, b=b*scale_eig, c=c*scale_eig,
        vecs=vecs,
        center=(0,0,0), phi=np.deg2rad(30), theta=np.deg2rad(20), psi=np.deg2rad(10),
        xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim),
        n_u=60, n_v=30,
        ax=ax1
    )
    ax1.set_title(titles[i], fontsize=24)

    # 下段: 通常の楕円体（scaleは自動調整）
    ax2 = fig.add_subplot(2, 4, i+5, projection='3d')
    # scaleを自動調整: 最大主軸長がlimの半分になるように
    max_axis = max(a, b, c)
    auto_scale = lim / 2 / max_axis if max_axis != 0 else 1.0
    plot_rotated_ellipsoid_with_half_axes(
        a=a*auto_scale, b=b*auto_scale, c=c*auto_scale,
        vecs=vecs,
        center=(0,0,0), phi=np.deg2rad(30), theta=np.deg2rad(20), psi=np.deg2rad(10),
        n_u=60, n_v=30,
        ax=ax2
    )
    ax2.set_title(f"{titles[i]}", fontsize=24)

plt.tight_layout(rect=[0, 0, 1, 0.97])  # suptitleの余白を少しだけ確保
plt.subplots_adjust(hspace=0.1)        # 上下段の間隔を控えめに調整
program_name = os.path.splitext(os.path.basename(__file__))[0]
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
filename = f"{program_name}_cov_ellipsoids.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()

fig = plt.figure(figsize=(18, 9))
scale_inv_list = [1e-2, 1e-2, 1e-2, 1e-2]
lim = 10
for i, eigvals in enumerate(eigvals_list):
    scale_eig = scale_eig_list[i]
    a, b, c = get_ellipsoid_axes(eigvals)
    vecs = eigvecs_10Hz[:, np.argsort(eigvals_10Hz)[::-1]][:3,:3]  # 10Hzの固有ベクトルに合わせる

    print(titles[i], "eigenvalues", a,b,c)

    # 上段: 逆数の楕円体
    scale_inv = scale_inv_list[i]
    ax1 = fig.add_subplot(2, 4, i+1, projection='3d')
    plot_rotated_ellipsoid_with_half_axes(
        a=1/a*scale_inv, b=1/b*scale_inv, c=1/c*scale_inv,
        vecs=vecs,
        center=(0,0,0), phi=np.deg2rad(30), theta=np.deg2rad(20), psi=np.deg2rad(10),
        xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim),
        n_u=60, n_v=30,
        ax=ax1
    )
    ax1.set_title(f"{titles[i]}", fontsize=24)

    # 下段: 逆数の楕円体
    ax2 = fig.add_subplot(2, 4, i+5, projection='3d')
    plot_rotated_ellipsoid_with_half_axes(
        a=1/a*scale_inv, b=1/b*scale_inv, c=1/c*scale_inv,
        vecs=vecs,
        center=(0,0,0), phi=np.deg2rad(30), theta=np.deg2rad(20), psi=np.deg2rad(10),
        n_u=60, n_v=30,
        ax=ax2
    )
    ax2.set_title(f"{titles[i]}", fontsize=24)

plt.tight_layout(rect=[0, 0, 1, 0.97])  # suptitleの余白を少しだけ確保
plt.subplots_adjust(hspace=0.1)        # 上下段の間隔を控えめに調整
program_name = os.path.splitext(os.path.basename(__file__))[0]
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
filename = f"{program_name}_cov_inv_ellipsoids.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()

# %%
# 2つの入力: u[t] = cos(w1*t) + cos(w2*t)
def simulate_cov_for_omega_flat(A, B, ws, T_total, T_burn, seed=0):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    sigma = 1e-2

    x = np.zeros((n, T_total))
    noise = rng.normal(scale=sigma, size=(n, T_total-1))

    u=0
    for w in ws:
        u += np.cos(w * np.arange(T_total-1))
    u = u / np.sqrt(np.mean(u**2))

    u = u * 0.1  # 入力の大きさを調整

    for t in range(T_total-1):
        x[:, t+1] = A @ x[:, t] + B[:, 0] * u[t] + noise[:, t]
    X_prev = x[:, T_burn-1:T_total-1]
    Qx = np.cov(X_prev)
    
    return Qx

ws_flat = range(1, 31)
Qx_flat = simulate_cov_for_omega_flat(A_both, B_both, [2*np.pi*w/fs for w in ws_flat], T_total, T_burn)
eigvals_flat, eigvecs_flat = np.linalg.eigh(Qx_flat)
print("Flat input eigenvalues:", eigvals_flat)

# 比較用の棒グラフ: eigvals_flat と eigvals_10_20Hz の逆数の和
eps = 1e-12
eig_flat = np.clip(eigvals_flat, eps, None)
eig_10_20 = np.clip(eigvals_10_20Hz, eps, None)

inv_sum_flat = np.sum(1.0 / eig_flat)
inv_sum_10_20 = np.sum(1.0 / eig_10_20)

labels = ["Flat (1-30Hz)", "10Hz + 20Hz"]
values = [inv_sum_flat, inv_sum_10_20]

# カラーマップ viridis から色を取得
cmap = plt.get_cmap('viridis')
colors = [cmap(0.25), cmap(0.75)]

fig, ax = plt.subplots(figsize=(3, 1.5), dpi=150)
bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)

# プロット全体を枠で囲む（スパインを表示・太くする）
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)

# 上の余白を少し開ける（上限を値の110-120%に設定）
ymax = max(values) if len(values) > 0 else 1.0
ax.set_ylim(0, ymax * 1.2)

# レイアウト調整（top領域を確保）
plt.tight_layout(rect=[0, 0, 1, 0.95])

# 保存
filename = f"{program_name}_invsum_comparison.svg"
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, filename), format="svg", bbox_inches='tight', pad_inches=0.02)
plt.show()