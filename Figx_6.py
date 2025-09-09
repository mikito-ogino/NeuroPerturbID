#%%
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

Time = 50
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

r_list = [0.6, 0.99]
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
plt.title('A matrix', fontsize=24)
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
    for t in range(T_total-1):
        if w1==0:
            u = np.cos(w2 * t)
        elif w2==0:
            u = np.cos(w1 * t)
        else:
            u = np.cos(w1 * t) + np.cos(w2 * t)
        x[:, t+1] = A @ x[:, t] + B[:,0]*u + noise[:, t]
    X_prev = x[:, T_burn-1:T_total-1]
    Qx = np.cov(X_prev)
    return Qx

for i, w1 in enumerate(ws):
    for j, w2 in enumerate(ws):
        Qx = simulate_cov_for_omega2(A_both, B_both, w1, w2, T_total, T_burn)
        eigvals = np.linalg.eigvalsh(Qx)
        inv_lambda_both_2d[i, j] = np.sum(1.0 / eigvals)

#%%
plt.figure(figsize=(8, 6))
im = plt.imshow(np.log10(inv_lambda_both_2d), extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]],
                origin='lower', aspect='auto', cmap='viridis', norm=None)
plt.xlabel('Input frequency 1 [Hz]', fontsize=20)
plt.ylabel('Input frequency 2 [Hz]', fontsize=20)
plt.title(f'r={r_list}, freq={hz_list} Hz \n combination stimuli', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
cbar = plt.colorbar(im)
cbar.set_label('log10(sum(1/λ))', fontsize=18)
cbar.ax.tick_params(labelsize=16)
plt.tight_layout()
program_name = os.path.splitext(os.path.basename(__file__))[0]
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
filename = f"{program_name}_sum_inv_lambda_2d.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()

# 最小のインデックスを表示
min_idx = np.unravel_index(np.argmin(sum_inv_lambda_2d), sum_inv_lambda_2d.shape)
print(f"最小値のインデックス: {min_idx}, 周波数: ({freqs[min_idx[0]]:.2f} Hz, {freqs[min_idx[1]]:.2f} Hz)")

#%%
def plot_rotated_ellipsoid_with_half_axes(a, b, c, center=(0.0, 0.0, 0.0), phi=0, theta=0, psi=0, xlim=(-10,10), ylim=(-10,10), zlim=(-10,10)):
    """
    3次元楕円体（回転あり）と各軸方向の半分だけ矢印を描画
    a, b, c : 楕円体の半径（主軸長）
    phi, theta, psi : ZYXオイラー角 [rad]（回転）
    center : 楕円体中心 (x0, y0, z0)
    """

    # 回転行列
    rot = R.from_euler('zyx', [phi, theta, psi]).as_matrix()

    # 楕円体の点群生成
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))

    # 回転・平行移動
    xyz = np.stack([x, y, z], axis=-1)
    xyz_rot = np.einsum('ij,klj->kli', rot, xyz)
    x_rot = xyz_rot[..., 0] + center[0]
    y_rot = xyz_rot[..., 1] + center[1]
    z_rot = xyz_rot[..., 2] + center[2]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    # ワイヤーフレームで楕円体を描画
    ax.plot_wireframe(x_rot, y_rot, z_rot, color='gray', linewidth=1, alpha=0.7)

    # 主軸方向ベクトル
    axes = np.array([[a,0,0],[0,b,0],[0,0,c]])
    axes_rot = rot @ axes.T

    # 各軸に半分だけ矢印（headなし）
    for i in range(3):
        ax.plot([center[0], center[0] + axes_rot[0, i]],
                [center[1], center[1] + axes_rot[1, i]],
                [center[2], center[2] + axes_rot[2, i]],
                color='k', linewidth=2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# 10Hz:20Hzの比率ごとに楕円体をプロット
scales = [1e-1, 10]  # [通常, 逆数]
lims = [20, 10]   # [通常, 逆数]

# 比率リスト
w1s = [10, 20, 10] 
w2s = [10, 20, 20]

for w1, w2 in zip(w1s,w2s):
    Qx = simulate_cov_for_omega2(A_both, B_both, w1, w2, T_total, T_burn)
    eigvals, eigvecs = np.linalg.eigh(Qx)
    # ソート（大きい順）
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    print(f"Ratio {r:.2f} eigvals:", eigvals)

    a, b, c = eigvals[0], eigvals[1], eigvals[2]

    # 通常の楕円体
    plot_rotated_ellipsoid_with_half_axes(
        a=a*scales[0], b=b*scales[0], c=c*scales[0],
        center=(0,0,0), phi=np.deg2rad(30), theta=np.deg2rad(20), psi=np.deg2rad(10),
        xlim=(-lims[0], lims[0]), ylim=(-lims[0], lims[0]), zlim=(-lims[0], lims[0])
    )

    # 逆数の楕円体
    plot_rotated_ellipsoid_with_half_axes(
        a=1/a*scales[1], b=1/b*scales[1], c=1/c*scales[1],
        center=(0,0,0), phi=np.deg2rad(30), theta=np.deg2rad(20), psi=np.deg2rad(10),
        xlim=(-lims[1], lims[1]), ylim=(-lims[1], lims[1]), zlim=(-lims[1], lims[1])
    )

    print(f"Sum of eigvals (ratio={r:.2f}):", np.sum(eigvals))
    print(f"Sum of inverse eigvals (ratio={r:.2f}):", np.sum(1/eigvals))

# %%
def simulate_cov_with_bandstop_noise(A, B, notch_freqs, fs, T_total, T_burn, seed=0):
    """
    入力u[t]として、notch_freqs以外のホワイトノイズを加える
    notch_freqs: 除去したい周波数 [Hz] のリストまたは配列
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    sigma = 1e-2

    # ホワイトノイズ生成
    noise_input = rng.normal(scale=1.0, size=T_total-1)
    nyq = fs / 2

    # notchフィルタを順に適用
    filtered_input = noise_input.copy()
    if np.isscalar(notch_freqs):
        notch_freqs = [notch_freqs]
    for notch_freq in notch_freqs:
        b, a = scipy.signal.iirnotch(notch_freq/nyq, Q=30)
        filtered_input = scipy.signal.filtfilt(b, a, filtered_input)

    x = np.zeros((n, T_total))
    noise = rng.normal(scale=sigma, size=(n, T_total-1))
    for t in range(T_total-1):
        u = filtered_input[t]
        x[:, t+1] = A @ x[:, t] + B[:,0]*u + noise[:, t]
    X_prev = x[:, T_burn-1:T_total-1]
    Qx = np.cov(X_prev)
    return Qx

Qx = simulate_cov_with_bandstop_noise(A_both, B_both, [20], fs, T_total, T_burn, seed=0)
eigvals, eigvecs = np.linalg.eigh(Qx)
# ソート（大きい順）
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]
print(f"Ratio {r:.2f} eigvals:", eigvals)

a, b, c = eigvals[0], eigvals[1], eigvals[2]

# 通常の楕円体
plot_rotated_ellipsoid_with_half_axes(
    a=a*scales[0], b=b*scales[0], c=c*scales[0],
    center=(0,0,0), phi=np.deg2rad(30), theta=np.deg2rad(20), psi=np.deg2rad(10),
    xlim=(-lims[0], lims[0]), ylim=(-lims[0], lims[0]), zlim=(-lims[0], lims[0])
)

# 逆数の楕円体
plot_rotated_ellipsoid_with_half_axes(
    a=1/a*scales[1], b=1/b*scales[1], c=1/c*scales[1],
    center=(0,0,0), phi=np.deg2rad(30), theta=np.deg2rad(20), psi=np.deg2rad(10),
    xlim=(-lims[1], lims[1]), ylim=(-lims[1], lims[1]), zlim=(-lims[1], lims[1])
)

print(f"Sum of eigvals (ratio={r:.2f}):", np.sum(eigvals))
print(f"Sum of inverse eigvals (ratio={r:.2f}):", np.sum(1/eigvals))
# %%
