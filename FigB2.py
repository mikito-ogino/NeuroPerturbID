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
import networkx as nx

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.default'] = 'regular'
# --------------------
# System definition
# --------------------
fs = 100.0        # sampling rate [Hz]  (edit as needed)
Ts = 1.0 / fs

Time = 100
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

# 複数入力: u[t] = sum_i cos(ws[i]*t)
def simulate_cov_for_omega_ws(A, B, ws, T_total, T_burn, seed=0, white_input=False):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    sigma = 1e-2

    x = np.zeros((n, T_total))
    noise = rng.normal(scale=sigma, size=(n, T_total-1))
    # 入力信号 u を事前に生成
    if white_input:
        u = rng.normal(scale=1.0, size=T_total-1)
        u = u / np.sqrt(np.mean(u**2))
    else:
        if ws is None or len(ws) == 0 or np.all(np.array(ws) == 0):
            u = np.zeros(T_total-1)
        else:
            u = np.zeros(T_total-1)
            for w in ws:
                if w != 0:
                    u += np.cos(w * np.arange(T_total-1))
            u = u / np.sqrt(np.mean(u**2))

    u = u * 0.1  # 入力の大きさを調整

    for t in range(T_total-1):
        x[:, t+1] = A @ x[:, t] + B[:, 0] * u[t] + noise[:, t]
    X_prev = x[:, T_burn-1:T_total-1]
    Qx = np.cov(X_prev)

    # A, Bの推定: x[t+1] ≈ A_hat x[t] + B_hat u[t]
    X1 = x[:, T_burn:T_total-1]  # shape: (n, T)
    X2 = x[:, T_burn+1:T_total]  # shape: (n, T)
    U1 = u[T_burn:T_total-1][None, :]  # shape: (1, T)
    XU = np.vstack([X1, U1])           # shape: (n+1, T)
    AB_hat = X2 @ np.linalg.pinv(XU)   # shape: (n, n+1)
    A_hat = AB_hat[:, :n]              # shape: (n, n)
    B_hat = AB_hat[:, n:]              # shape: (n, 1)

    return Qx, A_hat, B_hat

white_input = False  # Trueなら入力uはホワイトノイズ

for i, w1 in enumerate(ws):
    for j, w2 in enumerate(ws):
        Qx, A_hat, B_hat = simulate_cov_for_omega_ws(A_both, B_both, [w1, w2], T_total, T_burn, white_input=white_input)
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
Qx_0Hz, A_hat_0Hz, B_hat_0Hz = simulate_cov_for_omega_ws(A_both, B_both, [0, 0], T_total, T_burn)
eigvals_0Hz, eigvecs_0Hz = np.linalg.eigh(Qx_0Hz)

# 10Hz入力のみ
Qx_10Hz, A_hat_10Hz, B_hat_10Hz = simulate_cov_for_omega_ws(A_both, B_both, [2*np.pi*10/fs, 0], T_total, T_burn)
eigvals_10Hz, eigvecs_10Hz = np.linalg.eigh(Qx_10Hz)

# 20Hz入力のみ
Qx_20Hz, A_hat_20Hz, B_hat_20Hz = simulate_cov_for_omega_ws(A_both, B_both, [0, 2*np.pi*20/fs], T_total, T_burn)
eigvals_20Hz, eigvecs_20Hz = np.linalg.eigh(Qx_20Hz)

# 10Hz+20Hz入力
Qx_10_20Hz, A_hat_10_20Hz, B_hat_10_20Hz = simulate_cov_for_omega_ws(A_both, B_both, [2*np.pi*10/fs, 2*np.pi*20/fs], T_total, T_burn)
eigvals_10_20Hz, eigvecs_10_20Hz = np.linalg.eigh(Qx_10_20Hz)

#ノイズ入力
Qx_noise, A_hat_noise, B_hat_noise = simulate_cov_for_omega_ws(A_both, B_both, [0, 0], T_total, T_burn, white_input=True)
eigvals_noise, eigvecs_noise = np.linalg.eigh(Qx_noise)

Qx_10Hz_aligned = align_eigenvalues(eigvals_10Hz, eigvecs_0Hz, eigvecs_10Hz)
Qx_20Hz_aligned = align_eigenvalues(eigvals_20Hz, eigvecs_0Hz, eigvecs_20Hz)
Qx_10_20Hz_aligned = align_eigenvalues(eigvals_10_20Hz, eigvecs_0Hz, eigvecs_10_20Hz)
Qx_noise_aligned = align_eigenvalues(eigvals_noise, eigvecs_0Hz, eigvecs_noise)

# 楕円体の主軸長を計算
def get_ellipsoid_axes(eigvals):
    eigvals_sorted = np.sort(eigvals)[::-1]
    axes = eigvals_sorted[:3]
    return axes

eigvals_list = [eigvals_10_20Hz, eigvals_noise]
titles = ["10Hz + 20Hz", "White noise"]

fig = plt.figure(figsize=(18, 9))
scale_eig_list = [2.5, 2.5]  # 例: 各入力ごとに異なるスケールを設定
lim = 10

for i, eigvals in enumerate(eigvals_list):
    scale_eig = scale_eig_list[i]
    a, b, c = get_ellipsoid_axes(eigvals)
    vecs = eigvecs_10Hz[:, np.argsort(eigvals_10Hz)[::-1]][:3, :3]  # 10Hzの固有ベクトルに合わせる

    print(titles[i], "eigenvalues", a, b, c)

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
scale_inv_list = [5e-1, 5e-1]
lim = 10
for i, eigvals in enumerate(eigvals_list):
    scale_eig = scale_eig_list[i]
    a, b, c = get_ellipsoid_axes(eigvals)
    vecs = eigvecs_10Hz[:, np.argsort(eigvals_10Hz)[::-1]][:3, :3]  # 10Hzの固有ベクトルに合わせる

    print(titles[i], "eigenvalues", a, b, c)

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
inv_sum_10_20Hz = np.sum(1.0 / eigvals_10_20Hz)
inv_sum_noise = np.sum(1.0 / eigvals_noise)

plt.figure(figsize=(4, 4))
plt.bar(['10Hz+20Hz', 'White noise'], [inv_sum_10_20Hz, inv_sum_noise], color=['skyblue', 'orange'])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
program_name = os.path.splitext(os.path.basename(__file__))[0]
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
filename = f"{program_name}_inv_sum_bar.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()
# %%
# --------------------
# Simulation settings
# --------------------
Time = 1000

assert T_burn < T_total
t_idx = np.arange(T_total)

# 16個のランダムなr, hzを生成（r_listは0.1以上、0.9以上が多め）
rng = np.random.default_rng(42)
# 12個は0.9〜0.95、4個は0.1〜0.9
r_high = rng.uniform(0.9, 0.95, size=16)
r_low = rng.uniform(0.1, 0.9, size=0)
r_list = np.concatenate([r_high, r_low])
rng.shuffle(r_list)
hz_list = rng.uniform(1, 100, size=16)

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
plt.tight_layout()
plt.xticks([], fontsize=20)
plt.yticks([], fontsize=20)
program_name = os.path.splitext(os.path.basename(__file__))[0]
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
filename = f"{program_name}_big_A_matrix.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()

#%%
plt.figure(figsize=(8, 8))
G = nx.from_numpy_array(A)
pos = nx.spring_layout(G, seed=42)
edge_weights = [np.abs(A[i, j]) for i, j in G.edges()]
# ループ（自己ループ）を除外
edges_no_selfloop = [(i, j) for i, j in G.edges() if i != j]
edge_weights_no_selfloop = [np.abs(A[i, j]) for i, j in edges_no_selfloop]
# ノード描画（白文字、黒枠）
nx.draw_networkx_nodes(
    G, pos,
    node_size=350,
    node_color='#4C72B0',
    edgecolors='black',
    linewidths=2
)
nx.draw_networkx_edges(
    G, pos,
    edgelist=edges_no_selfloop,
    width=edge_weights_no_selfloop,
    edge_color='black',
    alpha=0.5
)
nx.draw_networkx_labels(
    G, pos,
    font_size=14,
    font_color='white'
)
plt.axis('off')
filename = f"{program_name}_A_network_graph.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()
#%%
# 指定回数だけA_hatの固有値から周波数をwsとしてsimulate_cov_for_omega_wsを繰り返す
num_iterations = 2  # ここで繰り返し回数を指定

ws_current = None
eigvals_list = []
freqs_list = []
Qx_list = []
A_hat_list = []
B_hat_list = []
inv_sum_list = []
inv_sum_dictionary = {}

######################################
# パッシブ入力（u=0, ws=[0,0]）
######################################
Qx_passive, A_hat_passive, B_hat_passive = simulate_cov_for_omega_ws(A_both, B_both, [0, 0], T_total, T_burn, white_input=False)
eigvals_passive, eigvecs_passive = np.linalg.eigh(Qx_passive)

eigvals_list.append(eigvals_passive)
freqs_list.append(np.angle(np.linalg.eigvals(A_hat_passive)) / (2 * np.pi * Ts))
Qx_list.append(Qx_passive)
A_hat_list.append(A_hat_passive)
B_hat_list.append(B_hat_passive)
inv_sum = np.sum(1.0 / np.linalg.eigvalsh(Qx_passive))
inv_sum_list.append(inv_sum)
inv_sum_dictionary[f"Passive"] = inv_sum

# パッシブ入力（u=0, ws=[0,0]）での状態変数xをシミュレートし、FFTでパワースペクトルをプロット
Qx_passive, A_hat_passive, B_hat_passive = simulate_cov_for_omega_ws(A_both, B_both, [0, 0], T_total, T_burn, white_input=False)
# simulate_cov_for_omega_wsの中でxは返していないので、xを返すように一時的に関数を再定義
def simulate_cov_and_x(A, B, ws, T_total, T_burn, seed=0, white_input=False):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    sigma = 1e-2
    x = np.zeros((n, T_total))
    noise = rng.normal(scale=sigma, size=(n, T_total-1))
    if white_input:
        u = rng.normal(scale=1.0, size=T_total-1)
        u = u / np.sqrt(np.mean(u**2))
    else:
        if ws is None or len(ws) == 0 or np.all(np.array(ws) == 0):
            u = np.zeros(T_total-1)
        else:
            u = np.zeros(T_total-1)
            for w in ws:
                if w != 0:
                    u += np.cos(w * np.arange(T_total-1))
            u = u / np.sqrt(np.mean(u**2))
    u = u * 0.1
    for t in range(T_total-1):
        x[:, t+1] = A @ x[:, t] + B[:, 0] * u[t] + noise[:, t]
    return x

x_passive = simulate_cov_and_x(A_both, B_both, [0, 0], T_total, T_burn, white_input=False)
# バーンイン後のみ
x_passive_steady = x_passive[:, T_burn:]

# 各次元ごとにパワースペクトルを計算して平均
fs_fft = fs
nfft = x_passive_steady.shape[1]
freqs_fft = np.fft.rfftfreq(nfft, d=1/fs_fft)
psd = np.abs(np.fft.rfft(x_passive_steady, axis=1))**2
psd_mean = psd.mean(axis=0)

plt.figure(figsize=(2,3))
plt.bar(freqs_fft, psd_mean, color='navy')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Power', fontsize=16)
plt.title('Power Spectrum', fontsize=16, pad=18)  # タイトルの余白を追加
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0, fs/2)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 上部に余白を追加
filename = f"{program_name}_passive_power_spectrum.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg", bbox_inches='tight')  # bbox_inches='tight'で切れ防止
plt.show()

######################################
# ホワイトノイズ入力（u=0, ws=[0,0]）
######################################
Qx_noise, A_hat_noise, B_hat_noise = simulate_cov_for_omega_ws(A_both, B_both, [0, 0], T_total, T_burn, white_input=True)
eigvals_noise, eigvecs_noise = np.linalg.eigh(Qx_noise)

# ノイズ入力のuをシミュレートし、プロットして保存
def simulate_cov_and_u(A, B, ws, T_total, T_burn, seed=0, white_input=False):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    sigma = 1e-2
    if white_input:
        u = rng.normal(scale=1.0, size=T_total-1)
        u = u / np.sqrt(np.mean(u**2))
    else:
        if ws is None or len(ws) == 0 or np.all(np.array(ws) == 0):
            u = np.zeros(T_total-1)
        else:
            u = np.zeros(T_total-1)
            for w in ws:
                if w != 0:
                    u += np.cos(w * np.arange(T_total-1))
            u = u / np.sqrt(np.mean(u**2))
    u = u * 0.1
    return u

u_noise = simulate_cov_and_u(A_both, B_both, [0, 0], T_total, T_burn, white_input=True)
u_noise_steady = u_noise

# x もプロット
def simulate_cov_and_x(A, B, ws, T_total, T_burn, seed=0, white_input=False):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    sigma = 1e-2
    x = np.zeros((n, T_total))
    noise = rng.normal(scale=sigma, size=(n, T_total-1))
    if white_input:
        u = rng.normal(scale=1.0, size=T_total-1)
        u = u / np.sqrt(np.mean(u**2))
    else:
        if ws is None or len(ws) == 0 or np.all(np.array(ws) == 0):
            u = np.zeros(T_total-1)
        else:
            u = np.zeros(T_total-1)
            for w in ws:
                if w != 0:
                    u += np.cos(w * np.arange(T_total-1))
            u = u / np.sqrt(np.mean(u**2))
    u = u * 0.1
    for t in range(T_total-1):
        x[:, t+1] = A @ x[:, t] + B[:, 0] * u[t] + noise[:, t]
    return x

x_noise = simulate_cov_and_x(A_both, B_both, [0, 0], T_total, T_burn, white_input=True)
x_noise_steady = x_noise[:, -100:]

# uのパワーをプロット
# FFTしてパワースペクトラムをプロット（uのパワースペクトル）
u_noise_full = simulate_cov_and_u(A_both, B_both, [0, 0], T_total, T_burn, white_input=True)
u_noise_steady_full = u_noise_full[T_burn:]
fs_fft = fs
nfft = u_noise_steady_full.shape[0]
freqs_fft = np.fft.rfftfreq(nfft, d=1/fs_fft)
psd_u = np.abs(np.fft.rfft(u_noise_steady_full))**2

plt.figure(figsize=(2,3))
plt.bar(freqs_fft, psd_u, color='darkred')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Power', fontsize=16)
plt.title('Input u Power Spectrum (White Noise)', fontsize=16, pad=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0, fs/2)
plt.tight_layout(rect=[0, 0, 1, 0.95])
filename = f"{program_name}_noise_input_u_power_spectrum.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg", bbox_inches='tight')
plt.show()

# FFTしてパワースペクトラムをプロット（xのパワースペクトル）
x_noise_steady_full = x_noise[:, T_burn:]  # バーンイン後全体
fs_fft = fs
nfft = x_noise_steady_full.shape[1]
freqs_fft = np.fft.rfftfreq(nfft, d=1/fs_fft)
psd = np.abs(np.fft.rfft(x_noise_steady_full, axis=1))**2
psd_mean = psd.mean(axis=0)

plt.figure(figsize=(2,3))
plt.bar(freqs_fft, psd_mean, color='navy')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Power', fontsize=16)
plt.title('Power Spectrum (White Noise Input)', fontsize=16, pad=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0, fs/2)
plt.tight_layout(rect=[0, 0, 1, 0.95])
filename = f"{program_name}_noise_input_x_power_spectrum.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg", bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(u_noise_steady.T)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Input u', fontsize=14)
plt.title('Input Trajectories (White Noise Input)', fontsize=16)
plt.tight_layout()
filename = f"{program_name}_noise_input_u_trajectories.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()

eigvals_list.append(eigvals_noise)
freqs_list.append(np.angle(np.linalg.eigvals(A_hat_noise)) / (2 * np.pi * Ts))
Qx_list.append(Qx_noise)
A_hat_list.append(A_hat_noise)
B_hat_list.append(B_hat_noise)
inv_sum = np.sum(1.0 / np.linalg.eigvalsh(Qx_noise))
inv_sum_list.append(inv_sum)
inv_sum_dictionary[f"WhiteNoise"] = inv_sum


for it in range(num_iterations):
    eigvals_A_hat = np.linalg.eigvals(A_hat_current)
    freqs_A_hat = np.angle(eigvals_A_hat) / (2 * np.pi * Ts)
    print(f"Iteration {it+1}")
    for i, (eig, freq) in enumerate(zip(eigvals_A_hat, freqs_A_hat)):
        print(f"A_hat eig {i}: {eig:.4f}, 周波数={freq:.4f} Hz")

    ws_current = 2 * np.pi * freqs_A_hat / fs
    Qx, A_hat_next, B_hat_next = simulate_cov_for_omega_ws(
        A_both, B_both, ws_current, T_total, T_burn, white_input=False
    )

    # ws_currentを持つ入力uをシミュレートし、プロットして保存
    u_ws_current = simulate_cov_and_u(A_both, B_both, ws_current, T_total, T_burn, white_input=False)
    u_ws_current_steady = u_ws_current

    plt.figure(figsize=(8, 4))
    plt.plot(u_ws_current_steady.T)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Input u', fontsize=14)
    plt.title(f'Input Trajectories (ws_current Input, iter={it+1})', fontsize=16)
    plt.tight_layout()
    filename = f"{program_name}_ws_current_input_u_trajectories_iter{it+1}.svg"
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, format="svg")
    plt.show()

    # Qxに対応するxをシミュレートし、パワースペクトラムをプロット
    x_ws_current = simulate_cov_and_x(A_both, B_both, ws_current, T_total, T_burn, white_input=False)
    x_ws_current_steady = x_ws_current[:, T_burn:]

    # uのパワーをプロット
    fs_fft = fs
    nfft = u_ws_current_steady.shape[0]
    freqs_fft = np.fft.rfftfreq(nfft, d=1/fs_fft)
    psd_u = np.abs(np.fft.rfft(u_ws_current_steady))**2

    plt.figure(figsize=(2,3))
    plt.bar(freqs_fft, psd_u, color='darkred')
    plt.xlabel('Frequency [Hz]', fontsize=16)
    plt.ylabel('Power', fontsize=16)
    plt.title(f'Input u Power Spectrum (iter={it+1})', fontsize=16, pad=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, fs/2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f"{program_name}_ws_current_input_u_power_spectrum_iter{it+1}.svg"
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, format="svg", bbox_inches='tight')
    plt.show()

    eigvals_list.append(eigvals_A_hat)
    freqs_list.append(freqs_A_hat)
    Qx_list.append(Qx)
    A_hat_list.append(A_hat_next)
    B_hat_list.append(B_hat_next)
    inv_sum = np.sum(1.0 / np.linalg.eigvalsh(Qx))
    inv_sum_list.append(inv_sum)
    A_hat_current = A_hat_next  # 次のイテレーション用

# 真のAの固有値から周波数を計算し、simulate_cov_for_omega_wsを実行してリストに追加
eigvals_A = np.linalg.eigvals(A)
freqs_A = np.angle(eigvals_A) / (2 * np.pi * Ts)
ws_A = 2 * np.pi * freqs_A / fs
Qx_trueA, A_hat_trueA, B_hat_trueA = simulate_cov_for_omega_ws(
    A_both, B_both, ws_A, T_total, T_burn, white_input=False
)

# ws_Aを持つ入力uをシミュレートし、プロットして保存
u_ws_A = simulate_cov_and_u(A_both, B_both, ws_A, T_total, T_burn, white_input=False)
u_ws_A_steady = u_ws_A[-100:]

plt.figure(figsize=(8, 4))
plt.plot(u_ws_A_steady.T)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Input u', fontsize=14)
plt.title('Input Trajectories (ws_A Input)', fontsize=16)
plt.tight_layout()
filename = f"{program_name}_ws_A_input_u_trajectories.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()

# uのパワースペクトルもプロット
fs_fft = fs
nfft = u_ws_A_steady.shape[0]
freqs_fft = np.fft.rfftfreq(nfft, d=1/fs_fft)
psd_u = np.abs(np.fft.rfft(u_ws_A_steady))**2

plt.figure(figsize=(2,3))
plt.bar(freqs_fft, psd_u, color='darkred')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Power', fontsize=16)
plt.title('Input u Power Spectrum (ws_A)', fontsize=16, pad=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0, fs/2)
plt.tight_layout(rect=[0, 0, 1, 0.95])
filename = f"{program_name}_ws_A_input_u_power_spectrum.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg", bbox_inches='tight')
plt.show()
eigvals_list.append(eigvals_A)
freqs_list.append(freqs_A)
Qx_list.append(Qx_trueA)
A_hat_list.append(A_hat_trueA)
B_hat_list.append(B_hat_trueA)
inv_sum = np.sum(1.0 / np.linalg.eigvalsh(Qx_trueA))
inv_sum_list.append(inv_sum)

# Qx_listの各Qxについて固有値をprint
for idx, Qx in enumerate(Qx_list):
    eigvals = np.linalg.eigvalsh(Qx)
    print(f"Iteration {idx}: Qx eigenvalues = {eigvals}")

# カラーパレット例（Nature系: muted, pastel, earth tones）
nature_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974']  # blue, green, red, purple, yellow
#%%%
plt.figure(figsize=(5, 3))
# passiveを除外
inv_sum_list_nopassive = inv_sum_list[1:][::-1]
labels_nopassive = ['white noise', 'design-1', 'design-2', 'design-3'][:len(inv_sum_list_nopassive)][::-1]
bars = plt.barh(
    range(1, len(inv_sum_list_nopassive) + 1),
    inv_sum_list_nopassive,
    color=nature_colors[0],
    edgecolor='black'
)
plt.yticks(
    range(1, len(inv_sum_list_nopassive) + 1),
    labels_nopassive,
    fontsize=16
)
plt.xticks(fontsize=16)
plt.xlim(1.5e4, max(inv_sum_list_nopassive)*1.01)
# Use ScalarFormatter with larger font for offset text (e.g., x10^4)
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.xaxis.offsetText.set_fontsize(18)  # Make the x10^4 label larger
plt.tight_layout()
program_name = os.path.splitext(os.path.basename(__file__))[0]
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
filename = f"{program_name}_inv_sum_iterations_hbar_nopassive.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()

# "passiveなし"（white noise, design-1, ...）のみの逆数和プロット
plt.figure(figsize=(6, 4))
plt.plot(
    range(2, len(inv_sum_list) + 1),
    inv_sum_list[1:],  # passiveを除く
    marker='o',
    color=nature_colors[2],
    linewidth=2,
    markersize=8
)
plt.xticks(
    range(2, len(inv_sum_list) + 1),
    ['white noise', 'design-1', 'design-2', 'design-3'][:len(inv_sum_list)-1],
    fontsize=16
)
plt.yticks(fontsize=16)
plt.tight_layout()
filename = f"{program_name}_inv_sum_iterations_nopassive.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()

# AとA_hat_listの誤差（ノルム）を計算してプロットで表示
errors = [np.linalg.norm(A - A_hat) for A_hat in A_hat_list]

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(errors) + 1), errors, marker='o', color='orange', linewidth=2)
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('||A - A_hat||', fontsize=16)
plt.title('A and A_hat Error per Iteration', fontsize=18)
plt.xticks(range(1, len(errors) + 1), fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
filename = f"{program_name}_A_hat_error_iterations.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()
# %%
# white noiseとdesign-1のときの固有値の3次元楕円体プロット

# 3次元ずつ抜き出して比較（1-3, 最後の3次元のみ）
def get_ellipsoid_axes_and_vecs(Qx):
    eigvals, eigvecs = np.linalg.eigh(Qx)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    axes = np.sqrt(eigvals)
    return axes, eigvecs

block_indices = [0, -1]  # 1-3, 最後の3次元のみ

fig = plt.figure(figsize=(4 * len(block_indices), 8))  # 横幅を6→4に縮小
lim_fixed = 4

inv_axes_noise_list = []
inv_axes_design1_list = []

for plot_idx, block in enumerate(block_indices):
    if block == 0:
        idx_start = block * 3
        idx_end = idx_start + 3
    elif block == -1:
        idx_start = 29
        idx_end = idx_start + 3
    Qx_noise_3d = Qx_list[1][idx_start:idx_end, idx_start:idx_end]
    Qx_design1_3d = Qx_list[2][idx_start:idx_end, idx_start:idx_end]

    axes_noise, vecs_noise = get_ellipsoid_axes_and_vecs(Qx_noise_3d)
    axes_design1, vecs_design1 = get_ellipsoid_axes_and_vecs(Qx_design1_3d)
    axes_design1_aligned = np.sqrt(align_eigenvalues(axes_design1**2, vecs_noise, vecs_design1))

    # 逆数
    inv_axes_noise = 1.0 / axes_noise
    inv_axes_design1 = 1.0 / axes_design1_aligned
    inv_axes_noise_list.append(inv_axes_noise)
    inv_axes_design1_list.append(inv_axes_design1)

    # 上段: white noise (逆数楕円体)
    ax1 = fig.add_subplot(2, len(block_indices), plot_idx + 1, projection='3d')
    a, b, c = inv_axes_noise
    plot_rotated_ellipsoid_with_half_axes(
        a=a, b=b, c=c,
        vecs=vecs_noise,
        center=(0,0,0),
        phi=0.0, theta=0.0, psi=0.0,
        xlim=(-lim_fixed, lim_fixed), ylim=(-lim_fixed, lim_fixed), zlim=(-lim_fixed, lim_fixed),
        n_u=60, n_v=30,
        ax=ax1
    )
    eig_label = f"Eig {idx_start + 1}-{idx_end}"
    ax1.set_title(eig_label, fontsize=24)

    # 下段: design-1 (逆数楕円体)
    ax2 = fig.add_subplot(2, len(block_indices), len(block_indices) + plot_idx + 1, projection='3d')
    a, b, c = inv_axes_design1
    plot_rotated_ellipsoid_with_half_axes(
        a=a, b=b, c=c,
        vecs=vecs_noise,  # noiseの固有ベクトルで描画
        center=(0,0,0),
        phi=0.0, theta=0.0, psi=0.0,
        xlim=(-lim_fixed, lim_fixed), ylim=(-lim_fixed, lim_fixed), zlim=(-lim_fixed, lim_fixed),
        n_u=60, n_v=30,
        ax=ax2
    )
    ax2.set_title(eig_label, fontsize=24)

plt.tight_layout(pad=0.5, w_pad=0.1)
filename = f"{program_name}_inv_ellipsoid_white_vs_design1_fixedlim_selectedblocks.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()

# %%
