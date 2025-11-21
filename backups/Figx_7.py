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
trial_num = 1

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
intensities = np.arange(0.5, 5.0, 0.1)  # 1 Hz to 20 Hz, 0.2 Hz steps

def simulate_cov_for_a_knownB(A, B, T_total, T_burn, seed=0, input_type='step', intensity=1.0):
    """
    A, B: system matrices
    input_type: 'impulse' or 'step'
    intensity: scalar amplitude
    The input (impulse or step) is applied at the first transition after T_burn,
    i.e. at index t = T_burn-1 for u_all (which drives x[:, T_burn]).
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    sigma = 1e-2

    if not (0 <= T_burn < T_total):
        raise ValueError("T_burn must satisfy 0 <= T_burn < T_total")

    x = np.zeros((n, T_total))
    noise = rng.normal(scale=sigma, size=(n, T_total-1))

    # prepare input sequence u_all: shape (n, T_total-1)
    u_all = np.zeros((n, T_total-1))
    start_idx = T_burn - 1  # put the input at the first transition after burn
    if input_type == 'step':
        # constant step input starting at start_idx
        if start_idx < u_all.shape[1]:
            u_all[:, start_idx:] = intensity
    elif input_type == 'impulse':
        # single-sample impulse at t = start_idx
        if 0 <= start_idx < u_all.shape[1]:
            u_all[:, start_idx] = intensity
    else:
        raise ValueError("input_type must be 'impulse' or 'step'")

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

fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
vmin, vmax = A.min(), A.max()
im = ax.imshow(A, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
n = A.shape[0]
ax.set_title("A matrix heatmap", fontsize=12)
ax.set_xlabel("column", fontsize=10)
ax.set_ylabel("row", fontsize=10)
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(np.arange(1, n+1))
ax.set_yticklabels(np.arange(1, n+1))

# overlay numbers on each cell with contrast-based color
for i in range(n):
    for j in range(n):
        val = A[i, j]
        norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        text_color = 'white' if norm_val < 0.5 else 'black'
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=text_color, fontsize=8)

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# Replace frequency sweep with analysis for 'impulse' and 'step' inputs over intensities
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
# remove existing results file if present
result_path = os.path.join(results_dir, "input_type_sweep_results.pkl")
if os.path.exists(result_path):
    try:
        os.remove(result_path)
        print(f"Removed existing file: {result_path}")
    except Exception as e:
        print(f"Failed to remove {result_path}: {e}")
else:
    print(f"No existing file to remove: {result_path}")
result_file = os.path.join(results_dir, "input_type_sweep_results.pkl")

input_types = ['impulse', 'step']

if os.path.exists(result_file):
    with open(result_file, "rb") as f:
        results = pickle.load(f)
else:
    results = {}
    for input_type in input_types:
        inv_lambda_list = []
        A_error_list = []
        tr_E_list = []

        for intensity in tqdm(intensities, desc=f"Running {input_type}"):
            Qx_trials = []
            A_err_trials = []
            trE_val = None
            for trial in range(trial_num):
                Qx, E, _, A_err = simulate_cov_for_a_knownB(
                    A_both, B_both, T_total, T_burn,
                    seed=trial, input_type=input_type, intensity=float(intensity)
                )
                Qx_trials.append(Qx)
                A_err_trials.append(A_err)
                trE_val = np.trace(E)

            Qx_mean = np.mean(Qx_trials, axis=0)
            eigvals = np.linalg.eigvalsh(Qx_mean)
            inv_lambda_list.append(1.0 / eigvals)
            A_error_list.append(np.mean(A_err_trials))
            tr_E_list.append(trE_val)

        results[input_type] = {
            "intensities": np.array(intensities),
            "inv_lambda": np.array(inv_lambda_list), # shape: (len(intensities), n)
            "A_error": np.array(A_error_list),
            "tr_E": np.array(tr_E_list)
        }

    with open(result_file, "wb") as f:
        pickle.dump(results, f)

#%%
# Plot only A_error for each input type
viridis = plt.get_cmap('viridis')
colors = {'impulse': viridis(0.2), 'step': viridis(0.7)}
program_name = os.path.splitext(os.path.basename(__file__))[0]

# font sizes
fontsize_title = 16
fontsize_labels = 14
fontsize_ticks = 12
marker_size = 6

for input_type in input_types:
    data = results[input_type]
    intens = data["intensities"]
    A_error = data["A_error"]

    fig, ax = plt.subplots(figsize=(3, 2.5), dpi=150)
    ax.plot(intens, A_error, marker='o', markersize=marker_size, color=colors.get(input_type, 'C0'))
    ax.set_xlabel('Input intensity', fontsize=fontsize_labels)
    ax.grid(True, linestyle='--', alpha=0.4)

    # major + minor ticks, increase tick label size and tick lengths
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks, length=7, width=1.2)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize_ticks-2, length=4, width=0.8)

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    filename = f"{program_name}_{input_type}_A_error.svg"
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, format='svg')
    plt.show()
    plt.close(fig)


# %%
