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
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from tqdm import tqdm
import pickle

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.default'] = 'regular'
# --------------------
# System definition
# --------------------
fs = 100.0        # sampling rate [Hz]  (edit as needed)
Ts = 1.0 / fs
trial_num = 10000

Time = 50
T_total = int(fs*Time)          # total steps
T_burn  = T_total*4//5            # discard initial transient

# ===== 例 =====
# 6次元: 複素ペア(3Hz, r=0.95, dt=0.01), 複素ペア(5Hz, r=1.0, dt=0.01), 実固有値(0.8, -0.5)
# rとf_hzを配列で指定


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
freqs = np.arange(1, 20, 0.05)  # 1 Hz to 20 Hz, 0.2 Hz steps
ws = 2 * np.pi * freqs / fs  # Convert Hz to rad/sample

def simulate_cov_for_omega_knownB(A, B, w, T_total, T_burn, seed=0):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    sigma = 1e-2

    x = np.zeros((n, T_total))
    noise = rng.normal(scale=sigma, size=(n, T_total-1))
    u_all = np.cos(w * np.arange(T_total-1)) * np.ones((n, T_total-1))  # shape: (n, T_total-1)
    # 各次元のu_allのパワー（エネルギー）を1に正規化
    u_all = u_all / np.sqrt(np.mean(u_all**2, axis=1, keepdims=True))
    for t in range(T_total-1):
        u = u_all[:, t]
        x[:, t+1] = A @ x[:, t] + B @ u + noise[:, t]

    X_prev = x[:, T_burn-1:T_total-1]
    Urow = u_all[:, T_burn-1:T_total-1]  # shape: (n, T)

    Y = x[:, T_burn:T_total] - B @ Urow

    A_hat = Y @ scipy.linalg.pinv(X_prev)

    Qx = np.cov(X_prev)
    E = np.cov(noise[:, T_burn-1:])
    A_err = np.linalg.norm(A_hat - A, ord='fro')**2

    return Qx, E, A_hat, A_err

r_list = [0.8]
hz_list = [10]

blocks = [block_discrete(r, f_hz) for r, f_hz in zip(r_list, hz_list)]
A = make_block_diag(blocks)



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

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
result_file = os.path.join(results_dir, "freq_sweep_results.pkl")

if os.path.exists(result_file):
    with open(result_file, "rb") as f:
        data = pickle.load(f)
    inv_lambda_both = data["inv_lambda_both"]
    A_error_list = data["A_error_list"]
    tr_E_list = data["tr_E_list"]
else:
    import concurrent.futures

    inv_lambda_both = []
    A_error_list = []
    tr_E_list = []

    def process_one_freq(w):
        A_error_list_local = []
        Qx_list_local = []
        for trial in range(trial_num):
            Qx, E, _, A_error = simulate_cov_for_omega_knownB(A_both, B_both, w, T_total, T_burn, seed=trial)
            A_error_list_local.append(A_error)
            Qx_list_local.append(Qx)
        A_error = np.mean(A_error_list_local)
        Qx = np.mean(Qx_list_local, axis=0)
        eigvals = np.linalg.eigvalsh(Qx)
        return 1.0 / eigvals, A_error, np.trace(E)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_one_freq, ws), total=len(ws)))
    for inv_lambda, A_error, tr_E in results:
        inv_lambda_both.append(inv_lambda)
        A_error_list.append(A_error)
        tr_E_list.append(tr_E)
    inv_lambda_both = np.array(inv_lambda_both)
    A_error_list = np.array(A_error_list)
    tr_E_list = np.array(tr_E_list)
    with open(result_file, "wb") as f:
        pickle.dump({
            "inv_lambda_both": inv_lambda_both,
            "A_error_list": A_error_list,
            "tr_E_list": tr_E_list
        }, f)

#%%
plt.figure(figsize=(5, 5))
im = plt.imshow(A, cmap='viridis')
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        bg_color = im.cmap(im.norm(A[i, j]))
        luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        text_color = 'black' if luminance > 0.5 else 'white'
        plt.text(j, i, f"{A[i, j]:.3f}", ha='center', va='center',
                 color=text_color, fontsize=24)
plt.tight_layout()
plt.xticks([], fontsize=32)
plt.yticks([], fontsize=32)
program_name = os.path.splitext(os.path.basename(__file__))[0]
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
filename = f"{program_name}_A_matrix.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()

# Plot: sum of eigenvalues, sum of inverse eigenvalues, and A_error vs frequency (1-20 Hz)
freq_range = (freqs >= 5) & (freqs <= 15)
freqs_plot = freqs[freq_range]
sum_eigvals = np.sum(1.0 / inv_lambda_both[freq_range], axis=1)  # sum of eigenvalues
sum_inv_eigvals = np.sum(inv_lambda_both[freq_range], axis=1)     # sum of inverse eigenvalues
A_error_plot = A_error_list[freq_range]

#%%
fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=200)

viridis = plt.get_cmap('viridis')
color0 = viridis(0.2)
color1 = viridis(0.5)
color2 = viridis(0.7)

for ax in axes:
    ax.tick_params(labelsize=24)
    ax.xaxis.label.set_size(24)
    ax.yaxis.label.set_size(24)
    ax.title.set_size(24)
    ax.yaxis.get_offset_text().set_fontsize(24)  # y軸の指数表記もフォントサイズ統一

axes[0].plot(freqs_plot, sum_eigvals, marker='o', color=color0)
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Value')
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

axes[1].plot(freqs_plot, sum_inv_eigvals, marker='o', color=color1)
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Value')
axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

axes[2].plot(freqs_plot, A_error_plot, marker='o', color=color2)
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('Value')
axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Add vertical line at 10 Hz to all subplots
for ax in axes:
    ax.axvline(x=10, color='black', linestyle='--', linewidth=2.5, label='10 Hz')
    # Optional: add label only to the first subplot

plt.tight_layout()
program_name = os.path.splitext(os.path.basename(__file__))[0]
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
filename = f"{program_name}_freq_plot.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()



#%%
Qx_0Hz, _, _, _ = simulate_cov_for_omega_knownB(A_both, B_both, 0, T_total, T_burn)
vals_0Hz, vecs_0Hz = np.linalg.eigh(Qx_0Hz)
print("Eigenvalues of Cov(x_perturbation, 0Hz):", vals_0Hz)

Qx_5Hz, _, _, _ = simulate_cov_for_omega_knownB(A_both, B_both, 2 * np.pi * 5 / fs, T_total, T_burn)
vals_5Hz, vecs_5Hz = np.linalg.eigh(Qx_5Hz) #昇順
print("Eigenvalues of Cov(x_perturbation, 5Hz):", vals_5Hz)

Qx_10Hz, _, _, _ = simulate_cov_for_omega_knownB(A_both, B_both, 2 * np.pi * 10 / fs, T_total, T_burn)
vals_10Hz, vecs_10Hz = np.linalg.eigh(Qx_10Hz)
print("Eigenvalues of Cov(x_perturbation, 10Hz):", vals_10Hz)

Qx_15Hz, _, _, _ = simulate_cov_for_omega_knownB(A_both, B_both, 2 * np.pi * 15 / fs, T_total, T_burn)
vals_15Hz, vecs_15Hz = np.linalg.eigh(Qx_15Hz)
print("Eigenvalues of Cov(x_perturbation, 15Hz):", vals_15Hz)

x_passive = np.zeros((A_both.shape[0], T_total))
rng = np.random.default_rng()
noise_passive = rng.normal(scale=1e-2, size=(A_both.shape[0], T_total-1))

for t in range(T_total-1):
    x_passive[:, t+1] = A_both @ x_passive[:, t] + noise_passive[:, t]

Qx_passive = np.cov(x_passive[:, T_burn:])

vals_passive, vecs_passive = np.linalg.eigh(Qx_passive) #昇順

# Qxを全てimshowで可視化
Qx_list = [Qx_passive, Qx_0Hz, Qx_5Hz, Qx_10Hz, Qx_15Hz]
labels = ['Passive', '0Hz', '5Hz', '10Hz', '15Hz']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, Qx, label in zip(axes.flat, Qx_list, labels):
    im = ax.imshow(Qx, cmap='viridis')
    ax.set_title(label, fontsize=18)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
filename = f"{program_name}_Qx_imshow.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()

#%%
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

vals_5Hz = align_eigenvalues(vals_5Hz, vecs_passive, vecs_5Hz)
vals_10Hz = align_eigenvalues(vals_10Hz, vecs_passive, vecs_10Hz)
vals_15Hz = align_eigenvalues(vals_15Hz, vecs_passive, vecs_15Hz)

print(vals_passive)
print(vals_5Hz)
print(vals_10Hz)
print(vals_15Hz)

# 楕円をsubplotで横に並べてプロットする
center = np.array([0.0, 0.0])
scale = 0.5
lim_value = 10

# Nature論文でよく使われる色 (例: muted blue, muted orange)
# Use viridis colormap for colors
viridis = plt.get_cmap('viridis')
nature_blue = viridis(0.2)
nature_orange = viridis(0.7)

vals_list = [vals_passive, vals_5Hz, vals_10Hz, vals_15Hz]
labels = ['Passive', '5Hz', '10Hz', '15Hz']

fig, axes = plt.subplots(1, 4, figsize=(16 , 4))
for ax, vals, label in zip(axes, vals_list, labels):
    eigvecs = vecs_passive
    u_major = eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])
    u_minor = eigvecs[:, 1] / np.linalg.norm(eigvecs[:, 1])
    phi_major = np.arctan2(u_major[1], u_major[0])
    e = Ellipse(xy=center, width=2*vals[0]*scale, height=2*vals[1]*scale,
                angle=np.degrees(phi_major), edgecolor='black', linewidth=3, fill=False, label=label)
    ax.add_patch(e)

    if vals[0]>vals[1]:
        major_vec = u_major * vals[0] * scale * 0.95
        minor_vec = u_minor * vals[1] * scale * 0.95
        arrow_major = ax.arrow(center[0], center[1], major_vec[0], major_vec[1], 
                    head_width=0.5, head_length=0.5, fc=nature_orange, ec=nature_orange, linewidth=3, length_includes_head=True, label=r'$\mu_1$')
        arrow_minor = ax.arrow(center[0], center[1], minor_vec[0], minor_vec[1], 
                    head_width=0.5, head_length=0.5, fc=nature_blue, ec=nature_blue, linewidth=3, length_includes_head=True, label=r'$\mu_2$')
    else:
        major_vec = u_minor * vals[1] * scale * 0.95
        minor_vec = u_major * vals[0] * scale * 0.95
        arrow_major = ax.arrow(center[0], center[1], major_vec[0], major_vec[1], 
                    head_width=0.5, head_length=0.5, fc=nature_orange, ec=nature_orange, linewidth=3, length_includes_head=True, label=r'$\mu_1$')
        arrow_minor = ax.arrow(center[0], center[1], minor_vec[0], minor_vec[1], 
                    head_width=0.5, head_length=0.5, fc=nature_blue, ec=nature_blue, linewidth=3, length_includes_head=True, label=r'$\mu_2$')

    if label == '10Hz':
        ax.set_title(label, fontsize=24, color='red')
    else:
        ax.set_title(label, fontsize=24)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-lim_value, lim_value)
    ax.set_ylim(-lim_value, lim_value)
    ax.tick_params(labelbottom=False, labelleft=False, labelsize=24)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))

    # Add legend only to the last subplot
    if ax == axes[3]:
        handles = [arrow_major, arrow_minor]
        labels_legend = [r'$\mu_1$', r'$\mu_2$']
        ax.legend(handles, labels_legend, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=20, frameon=True)

plt.tight_layout(pad=2.0)  # Increase padding to avoid cutting titles
filename = f"{program_name}_ellipse_overlay_subplot.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg", bbox_inches='tight')  # Use bbox_inches='tight' to include all labels/titles
plt.show()

# --- 逆数プロット subplot ---
scales = [0.0020, 10, 10, 10]  # 個々に設定したい場合はここで調整

lim_value = 10

inv_vals_list = [1/vals for vals in vals_list]
inv_labels = [f'{label}' for label in labels]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, inv_vals, label, scale in zip(axes, inv_vals_list, inv_labels, scales):
    eigvecs = vecs_passive
    u_major = eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])
    u_minor = eigvecs[:, 1] / np.linalg.norm(eigvecs[:, 1])
    phi_major = np.arctan2(u_major[1], u_major[0])
    e = Ellipse(xy=center, width=2*inv_vals[0]*scale, height=2*inv_vals[1]*scale,
                angle=np.degrees(phi_major), edgecolor='black', linewidth=3, fill=False, label=label)
    ax.add_patch(e)

    if vals[0]>vals[1]:
        major_vec = u_major * inv_vals[0] * scale * 0.95
        minor_vec = u_minor * inv_vals[1] * scale * 0.95
        arrow_major = ax.arrow(center[0], center[1], major_vec[0], major_vec[1],
                    head_width=0.5, head_length=0.5, fc=nature_orange, ec=nature_orange, linewidth=3, length_includes_head=True, label=r'$\mu_1$')
        arrow_minor = ax.arrow(center[0], center[1], minor_vec[0], minor_vec[1],
                    head_width=0.5, head_length=0.5, fc=nature_blue, ec=nature_blue, linewidth=3, length_includes_head=True, label=r'$\mu_2$')
    else:
        major_vec = u_minor * inv_vals[1] * scale* 0.95
        minor_vec = u_major * inv_vals[0] * scale* 0.95
        arrow_major = ax.arrow(center[0], center[1], major_vec[0], major_vec[1],
                    head_width=0.5, head_length=0.5, fc=nature_orange, ec=nature_orange, linewidth=3, length_includes_head=True, label=r'$\mu_1$')
        arrow_minor = ax.arrow(center[0], center[1], minor_vec[0], minor_vec[1],
                head_width=0.5, head_length=0.5, fc=nature_blue, ec=nature_blue, linewidth=3, length_includes_head=True, label=r'$\mu_2$')

    if label == 'Passive':
        ax.text(lim_value-0.5, lim_value-0.5, r'$\times 1/5000$', fontsize=24,
                ha='right', va='top', color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    if label == '10Hz':
        ax.set_title(label, fontsize=24, color='red')
    else:
        ax.set_title(label, fontsize=24)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-lim_value, lim_value)
    ax.set_ylim(-lim_value, lim_value)
    ax.tick_params(labelbottom=False, labelleft=False, labelsize=24)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))

    # Add legend only to the last subplot
    if ax == axes[3]:
        handles = [arrow_major, arrow_minor]
        labels_legend = [r'$1/\mu_1$', r'$1/\mu_2$']
        ax.legend(handles, labels_legend, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=20, frameon=True)

plt.tight_layout(pad=2)
filename = f"{program_name}_ellipse_overlay_inv_subplot.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg")
plt.show()
# %%
