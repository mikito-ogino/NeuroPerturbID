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

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.default'] = 'regular'
# --------------------
# System definition
# --------------------
fs = 1000.0        # sampling rate [Hz]  (edit as needed)
Ts = 1.0 / fs

Time = 5
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

# Frequency sweep: specify directly in Hz
freqs = np.arange(1, 51, 1)  # 1 Hz to 100 Hz, 1 Hz steps
ws = 2 * np.pi * freqs / fs  # Convert Hz to rad/sample

def simulate_cov_for_omega_knownB(A, B, w, T_total, T_burn, seed=0):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    sigma = 1e-2

    x = np.zeros((n, T_total))
    noise = rng.normal(scale=sigma, size=(n, T_total-1))  # shape: (n, T_total-1), independent per dimension
    for t in range(T_total-1):
        u = np.cos(w * t) * np.ones(n)         # u is now a vector of shape (n,)
        x[:, t+1] = A @ x[:, t] + B @ u + noise[:, t]  # B @ u handles vector input

    # 回帰に使う区間
    X_prev = x[:, T_burn-1:T_total-1]        # (n, T)
    t_range = np.arange(T_burn-1, T_total-1)
    Urow   = np.cos(w * t_range) * np.ones((n, len(t_range)))  # (n, T)

    # Y = x[t+1] - B u[t]
    Y = x[:, T_burn:T_total] - B @ Urow      # (n, T)

    # OLS:  A_hat = Y X^T (X X^T)^{-1}
    G = X_prev @ X_prev.T                    # (n, n)
    A_hat = (Y @ X_prev.T) @ np.linalg.pinv(G)

    # サンプル共分散
    Qx = np.cov(X_prev)                   # 中心でΣ_x
    E  = np.cov(noise[:, T_burn-1:])      # 中心でΣ_E
    A_err = np.linalg.norm(A_hat - A, ord='fro')

    return Qx, E, A_hat, A_err

r_list = [0.6, 0.99]
hz_list = [10, 40]

blocks = [block_discrete(r, f_hz) for r, f_hz in zip(r_list, hz_list)]
A = make_block_diag(blocks)

eigvals_A = np.linalg.eigvals(A)
for i, eig in enumerate(eigvals_A):
    r = np.abs(eig)
    theta = np.angle(eig)
    freq_hz = theta / (2 * np.pi * Ts)
    print(f"eig {i}: r={r:.4f}, 周波数={freq_hz:.4f} Hz, 減衰率={r:.4f}")
A_both = A
B = np.eye(A.shape[0])
B_both = B
B_both = np.zeros_like(B)
B_both[0, 0] = 1
B_both[1, 1] = 1

n = A.shape[0]

inv_lambda_both = []
for w in ws:
    Qx, E, _, _ = simulate_cov_for_omega_knownB(A_both, B_both, w, T_total, T_burn)
    eigvals = np.linalg.eigvalsh(Qx)
    inv_lambda_both.append(1.0 / eigvals)
inv_lambda_both = np.array(inv_lambda_both)

#%%
# Hzの入力で各系のxをプロット
input_freqs = [10]  # Hz
w_Hz_list = [2 * np.pi * f / fs for f in input_freqs]

# シミュレーション
x_both = np.zeros((A_both.shape[0], T_total))

rng = np.random.default_rng(0)
noise = rng.normal(scale=1, size=(A_both.shape[0], T_total-1))

for t in range(T_total-1):
    # 合成波: hz_listの各周波数でコサイン波を合成（scaleは1で統一）
    ub = sum(np.cos(2 * np.pi * hz * t / fs) for hz in hz_list) * np.ones(A_both.shape[0])*1.3
    x_both[:, t+1] = A_both @ x_both[:, t] + B_both @ ub + noise[:A_both.shape[0], t]

# プロット: A_both
plt.figure(figsize=(12, 4))
for i in range(x_both.shape[0]):
    plt.subplot(1, x_both.shape[0], i+1)
    plt.plot(np.arange(T_burn, T_total), x_both[i, T_burn:])
    plt.title(f'A_both dim {i+1}')
    plt.xlabel('Time step')
plt.suptitle('A_both, input 25Hz (after T_burn)')
plt.tight_layout()
plt.show()

# x_both の共分散行列をプロット（値を重ねて表示）
plt.figure(figsize=(6, 5))
Qx_perturbation = np.cov(x_both[:, T_burn:])
im = plt.imshow(Qx_perturbation, cmap='viridis')
plt.title('Cov(x_both)')
plt.colorbar(im)
for i in range(Qx_perturbation.shape[0]):
    for j in range(Qx_perturbation.shape[1]):
        plt.text(j, i, f"{Qx_perturbation[i, j]:.2e}", ha='center', va='center',
                 color='white' if Qx_perturbation[i, j] < Qx_perturbation.max()/2 else 'black', fontsize=8)
plt.suptitle('Covariance matrix of x (input 10Hz, after T_burn)')
plt.tight_layout()
plt.show()

vals_perturbation, vecs_perturbation = np.linalg.eigh(Qx_perturbation) #昇順
print("Eigenvalues of Cov(x_perturbation):", vals_perturbation)

#%%
# Passive case: no input (u = 0)
x_passive = np.zeros((A_both.shape[0], T_total))
noise_passive = rng.normal(scale=1, size=(A_both.shape[0], T_total-1))

for t in range(T_total-1):
    x_passive[:, t+1] = A_both @ x_passive[:, t] + noise_passive[:, t]

# Plot time series for passive case
plt.figure(figsize=(12, 4))
for i in range(x_passive.shape[0]):
    plt.subplot(1, x_passive.shape[0], i+1)
    plt.plot(np.arange(T_burn, T_total), x_passive[i, T_burn:])
    plt.title(f'Passive dim {i+1}')
    plt.xlabel('Time step')
plt.suptitle('Passive system (after T_burn)')
plt.tight_layout()
plt.show()

# Covariance matrix for passive case
plt.figure(figsize=(6, 5))
Qx_passive = np.cov(x_passive[:, T_burn:])
im = plt.imshow(Qx_passive, cmap='viridis')
plt.title('Cov(x_passive)')
plt.colorbar(im)
for i in range(Qx_passive.shape[0]):
    for j in range(Qx_passive.shape[1]):
        plt.text(j, i, f"{Qx_passive[i, j]:.2e}", ha='center', va='center',
                 color='white' if Qx_passive[i, j] < Qx_passive.max()/2 else 'black', fontsize=8)
plt.suptitle('Covariance matrix of x (passive, after T_burn)')
plt.tight_layout()
plt.show()

vals_passive, vecs_passive = np.linalg.eigh(Qx_passive) #昇順
# x_passive と x_perturbation の時系列プロット（黒線, subplot）
plt.figure(figsize=(12, 9))

# 両方のデータのy範囲を取得
ymin = min(np.min(x_passive[:, T_burn:]), np.min(x_both[:, T_burn:]), -1.5)
ymax = max(np.max(x_passive[:, T_burn:]), np.max(x_both[:, T_burn:]), 1.5)

# u（入力）を計算
u_perturb = np.zeros(T_total - T_burn)
for i, w in enumerate(w_Hz_list):
    u_perturb += np.cos(w * np.arange(T_burn, T_total))

# x_passive
for i in range(x_passive.shape[0]):
    plt.subplot(3, x_passive.shape[0], i+1)
    plt.plot(np.arange(T_burn, T_total), x_passive[i, T_burn:], color='black')
    plt.title(f'$x_{{{i+1}}}$')
    plt.xlabel('Time step')
    plt.ylim(ymin, ymax)
    plt.tight_layout()

# x_perturbation
for i in range(x_both.shape[0]):
    plt.subplot(3, x_both.shape[0], x_both.shape[0]+i+1)
    plt.plot(np.arange(T_burn, T_total), x_both[i, T_burn:], color='black')
    plt.title(f'$x_{{{i+1}}}$')
    plt.xlabel('Time step')
    plt.ylim(ymin, ymax)
    plt.tight_layout()

# u（入力）プロット（全次元同じなので1つだけ表示）
plt.subplot(3, x_both.shape[0], 2*x_both.shape[0]+1)
plt.plot(np.arange(T_burn, T_total), u_perturb, color='k')
plt.title('Input u (10Hz)')
plt.xlabel('Time step')
plt.ylim(-1.5, 1.5)
plt.tight_layout()

plt.suptitle('Time series: passive (top), perturbation (middle), input u (bottom)')
plt.show()

# fig, ax = plt.subplots(figsize=(6, 6))

# # 楕円の中心
# center = [0, 0]
   
# def plot_rotated_ellipse_with_half_axes(a, b, phi, center=(0.0, 0.0),
#                                         arrow_kwargs=None, ellipse_kwargs=None, xlim=[0,0], ylim=[0,0]):
#     """
#     回転楕円と長軸・短軸を半分だけ矢印で描画する。
#     a, b : 楕円の半径（順不同; 大きい方を長軸として扱う）
#     phi  : 回転角 [rad]（x軸から反時計回り）
#     center : 楕円中心 (x0, y0)
#     arrow_kwargs : ax.arrow に渡すオプション（色や矢印頭サイズなど）
#     ellipse_kwargs : Ellipse パッチに渡すオプション（線種など）
#     """
#     x0, y0 = center

#     # 長軸・短軸の判定（A >= B）
#     if a >= b:
#         A, B = a, b
#         phi_major = phi
#     else:
#         A, B = b, a
#         phi_major = phi + np.pi/2

#     # 軸の単位ベクトル
#     u_major = np.array([np.cos(phi_major), np.sin(phi_major)])
#     u_minor = np.array([-np.sin(phi_major), np.cos(phi_major)])

#     # デフォルト描画パラメータ
#     if arrow_kwargs is None:
#         arrow_kwargs = dict(head_width=0.25, head_length=0.3, linewidth=2, length_includes_head=True)
#     if ellipse_kwargs is None:
#         ellipse_kwargs = dict(fill=False, linewidth=2)

#     fig, ax = plt.subplots(figsize=(6, 6))

#     # 楕円（MatplotlibのEllipseは角度を度で指定）
#     e = Ellipse(xy=(x0, y0), width=2*A, height=2*B, angle=np.degrees(phi_major), **ellipse_kwargs)
#     ax.add_patch(e)

#     # 長軸方向に矢印（半分だけ、矢印の長さを少し短くする）
#     arrow_scale = 0.98  # 1より小さくすると楕円からはみ出しにくい
#     dx_major, dy_major = arrow_scale * A * u_major
#     ax.arrow(x0, y0, dx_major, dy_major, color="black", **arrow_kwargs)

#     # 短軸方向に矢印（半分だけ、矢印の長さを少し短くする）
#     dx_minor, dy_minor = arrow_scale * B * u_minor
#     ax.arrow(x0, y0, dx_minor, dy_minor, color="black", **arrow_kwargs)

#     # 見やすさ調整
#     ax.set_aspect('equal', adjustable='box')
#     margin = 1.2 * max(A, B)
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
#     ax.tick_params(labelbottom=False, labelleft=False)
#     ax.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout(pad=0.05)
#     plt.show()

# print(vals_passive)
# print(vals_perturbation)

# lim_value = 10
# plot_rotated_ellipse_with_half_axes(
#     a=vals_passive[0], b=vals_passive[3], phi=np.deg2rad(30), center=(0.0, 0.0), xlim=(-lim_value, lim_value), ylim=(-lim_value, lim_value)
# )

# plot_rotated_ellipse_with_half_axes(
#     a=vals_perturbation[0], b=vals_perturbation[3], phi=np.deg2rad(30), center=(0.0, 0.0), xlim=(-lim_value, lim_value), ylim=(-lim_value, lim_value)
# )

# lim_value = 10
# scale = 10
# plot_rotated_ellipse_with_half_axes(
#     a=1/vals_passive[0]*scale, b=1/vals_passive[3]*scale, phi=np.deg2rad(30), center=(0.0, 0.0), xlim=(-lim_value, lim_value), ylim=(-lim_value, lim_value)
# )

# plot_rotated_ellipse_with_half_axes(
#     a=1/vals_perturbation[0]*scale, b=1/vals_perturbation[3]*scale, phi=np.deg2rad(30), center=(0.0, 0.0), xlim=(-lim_value, lim_value), ylim=(-lim_value, lim_value)
# )


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

lim_value = 40
# 例: passive
plot_rotated_ellipsoid_with_half_axes(
    a=vals_passive[1], b=vals_passive[0], c=vals_passive[3],
    center=(0,0,0), phi=np.deg2rad(30), theta=np.deg2rad(20), psi=np.deg2rad(10),
    xlim=(-lim_value, lim_value), ylim=(-lim_value, lim_value), zlim=(-lim_value, lim_value)
)

# 例: perturbation
plot_rotated_ellipsoid_with_half_axes(
    a=vals_perturbation[2], b=vals_perturbation[0], c=vals_perturbation[3],
    center=(0,0,0), phi=np.deg2rad(30), theta=np.deg2rad(20), psi=np.deg2rad(10),
    xlim=(-lim_value, lim_value), ylim=(-lim_value, lim_value), zlim=(-lim_value, lim_value)
)

lim_value = 5
# 例: passive (逆数 × 10)
plot_rotated_ellipsoid_with_half_axes(
    a=1/vals_passive[1]*10, b=1/vals_passive[0]*10, c=1/vals_passive[3]*10,
    center=(0,0,0), phi=np.deg2rad(30), theta=np.deg2rad(20), psi=np.deg2rad(10),
    xlim=(-lim_value, lim_value), ylim=(-lim_value, lim_value), zlim=(-lim_value, lim_value)
)

# 例: perturbation (逆数 × 10)
plot_rotated_ellipsoid_with_half_axes(
    a=1/vals_perturbation[0]*10, b=1/vals_perturbation[2]*10, c=1/vals_perturbation[3]*10,
    center=(0,0,0), phi=np.deg2rad(30), theta=np.deg2rad(20), psi=np.deg2rad(10),
    xlim=(-lim_value, lim_value), ylim=(-lim_value, lim_value), zlim=(-lim_value, lim_value)
)
# %%
print("Sum of vals_passive:", np.sum(vals_passive))
print("Sum of vals_perturbation:", np.sum(vals_perturbation))
print("Sum of inverse vals_passive:", np.sum(1/vals_passive))
print("Sum of inverse vals_perturbation:", np.sum(1/vals_perturbation))
