#%%
# Ellipsoid plot image
###############

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
def plot_rotated_ellipsoid_with_half_axes(a, b, c, center=(0.0, 0.0, 0.0), phi=0, theta=0, psi=0, xlim=(-10,10), ylim=(-10,10), zlim=(-10,10), save_name=None):
    """
    3次元楕円体（回転あり）と各軸方向の半分だけ矢印を描画
    a, b, c : 楕円体の半径（主軸長）
    phi, theta, psi : ZYXオイラー角 [rad]（回転）
    center : 楕円体中心 (x0, y0, z0)
    save_name : 保存するファイル名（拡張子なし、svgで保存）
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
    ax.set_box_aspect([1,1,1])
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_name is not None:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        program_name = os.path.splitext(os.path.basename(__file__))[0]
        file_path = os.path.join(results_dir, f"{program_name}_{save_name}.svg")
        plt.savefig(file_path, format="svg")
        plt.close(fig)
    else:
        plt.show()

vals_passive=[1, 1, 8]
vals_perturbation=[3, 3, 5]

lim_value = 5
# 例: passive
plot_rotated_ellipsoid_with_half_axes(
    a=vals_passive[0], b=vals_passive[1], c=vals_passive[2],
    center=(0,0,0), phi=np.deg2rad(30), theta=np.deg2rad(20), psi=np.deg2rad(10),
    xlim=(-lim_value, lim_value), ylim=(-lim_value, lim_value), zlim=(-lim_value, lim_value),
    save_name="passive"
)

# 例: perturbation
plot_rotated_ellipsoid_with_half_axes(
    a=vals_perturbation[0], b=vals_perturbation[1], c=vals_perturbation[2],
    center=(0,0,0), phi=np.deg2rad(30), theta=np.deg2rad(20), psi=np.deg2rad(10),
    xlim=(-lim_value, lim_value), ylim=(-lim_value, lim_value), zlim=(-lim_value, lim_value),
    save_name="perturbation"
)

lim_value = 8
# 例: passive (逆数 × 10)
plot_rotated_ellipsoid_with_half_axes(
    a=1/vals_passive[0]*10, b=1/vals_passive[1]*10, c=1/vals_passive[2]*10,
    center=(0,0,0), phi=np.deg2rad(30), theta=np.deg2rad(20), psi=np.deg2rad(10),
    xlim=(-lim_value, lim_value), ylim=(-lim_value, lim_value), zlim=(-lim_value, lim_value),
    save_name="passive_inv"
)

# 例: perturbation (逆数 × 10)
plot_rotated_ellipsoid_with_half_axes(
    a=1/vals_perturbation[0]*10, b=1/vals_perturbation[1]*10, c=1/vals_perturbation[2]*10,
    center=(0,0,0), phi=np.deg2rad(30), theta=np.deg2rad(20), psi=np.deg2rad(10),
    xlim=(-lim_value, lim_value), ylim=(-lim_value, lim_value), zlim=(-lim_value, lim_value),
    save_name="perturbation_inv"
)
# %%
# 単一周波数の時系列 + ノイズ（振幅を小さく）
t = np.linspace(0, 1, 500)
freqs = [3, 7]  # Hz
np.random.seed(0)
noise_level = 0.3
amp_single = 0.2  # 振幅を小さく
signal = sum(amp_single * np.sin(2 * np.pi * f * t) for f in freqs) + noise_level * np.random.randn(len(t))

fig1, ax1 = plt.subplots(figsize=(7, 3))
ax1.plot(t, signal, color='k', linewidth=2)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylim(-2, 2)
plt.tight_layout()
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
program_name = os.path.splitext(os.path.basename(__file__))[0]
file_path1 = os.path.join(results_dir, f"{program_name}_simple_freq.svg")
plt.savefig(file_path1, format="svg")
plt.close(fig1)

# 複数周波数の時系列 + ノイズ
freqs_multi = [3, 7, 13, 20]
signal_multi = sum(np.sin(2 * np.pi * f * t) for f in freqs_multi) + noise_level * np.random.randn(len(t))

fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.plot(t, signal_multi, color='k', linewidth=2)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_xticks([])
ax2.set_yticks([])
ax1.set_ylim(-2, 2)
plt.tight_layout()
file_path2 = os.path.join(results_dir, f"{program_name}_multi_freq.svg")
plt.savefig(file_path2, format="svg")
plt.close(fig2)

freqs_multi = [13, 20]
signal_multi = sum(np.sin(2 * np.pi * f * t) for f in freqs_multi)

fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.plot(t, signal_multi, color='k', linewidth=2)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_xticks([])
ax2.set_yticks([])
ax1.set_ylim(-2, 2)
plt.tight_layout()
file_path2 = os.path.join(results_dir, f"{program_name}_u.svg")
plt.savefig(file_path2, format="svg")
plt.close(fig2)
