#%%
# Discrete-time simulation for x[t+1] = A x[t] + B u[t]
# A includes a 10 Hz damped mode (ζ = 0.6). We sweep input frequency ω (rad/sample)
# and plot tr(Q^{-1}) where Q is the sample covariance of x in steady state.

######################
#Practical Simulation
#####################

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
import re
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.default'] = 'regular'
# --------------------
# System definition
# --------------------
fs = 100.0        # sampling rate [Hz]  (edit as needed)
Ts = 1.0 / fs

Time_for_record = 10
T_total = int(fs*Time_for_record)          # total steps
T_burn  = T_total*4//5            # discard initial transient

# ===== 例 =====
# 6次元: 複素ペア(3Hz, r=0.95, dt=0.01), 複素ペア(5Hz, r=1.0, dt=0.01), 実固有値(0.8, -0.5)
# rとf_hzを配列で指定

#Simulation condition
trial_num = 200

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

# 複数入力: u[t] = sum_i cos(ws[i]*t)
def simulate_cov_for_omega_ws(A, B, ws, T_total, T_burn, seed=0, white_input=False, uniform_input=False, plot_signals=False):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    sigma = 1e-2

    x = np.zeros((n, T_total))
    noise = rng.normal(scale=sigma, size=(n, T_total-1))
    # 入力信号 u を事前に生成
    if white_input:
        u = rng.normal(scale=1.0, size=T_total-1)
        u = u / np.sqrt(np.mean(u**2))
    elif uniform_input:
        u = np.zeros(T_total-1)
        for w in range(0, 100, 1):
            if w != 0:
                u += np.cos(w * np.arange(T_total-1))
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

    if plot_signals:
        # plot rows of X1 as vertically stacked, label-free, blue subplots with minimal spacing

        n_plots = X1.shape[0]
        fig, axes = plt.subplots(n_plots, 1, figsize=(3, n_plots*0.05), sharex=True)
        if n_plots == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(X1[i, :], color='blue', linewidth=0.7)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylim(-5, 5)  # 全サブプロットの表示範囲を固定
            for spine in ax.spines.values():
                spine.set_visible(False)
        plt.subplots_adjust(hspace=0.01)
        plt.tight_layout(pad=0)

        # ensure results_dir exists (use global if present)
        if 'results_dir' not in globals():
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
        else:
            results_dir = globals().get('results_dir')

        # safe program_name fallback
        prog = globals().get('program_name', 'figure')

        filename = f"{prog}_signals_seed{seed}.svg"
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        plt.close(fig)

        # plot input U1 similarly to X1
        n_u = U1.shape[0]
        fig_u, axes_u = plt.subplots(n_u, 1, figsize=(3, n_u * 0.5), sharex=True)
        if n_u == 1:
            axes_u = [axes_u]
        for i, ax in enumerate(axes_u):
            ax.plot(U1[i, :], color='black', linewidth=0.7)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        plt.subplots_adjust(hspace=0.01)
        plt.tight_layout(pad=0)

        filename = f"{prog}_U1_seed{seed}.svg"
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        plt.close(fig_u)

    return Qx, A_hat, B_hat

# --------------------
# Simulation settings
# --------------------
assert T_burn < T_total
t_idx = np.arange(T_total)

number_of_nodes = 16  # 状態変数の次元数

# 16個のランダムなr, hzを生成（r_listは0.1以上、0.9以上が多め）
rng = np.random.default_rng(42)
# 12個は0.9〜0.95、4個は0.1〜0.9
r_high = rng.uniform(0.96, 0.96, size=number_of_nodes)
r_low = rng.uniform(0.1, 0.9, size=0)
r_list = np.concatenate([r_high, r_low])
rng.shuffle(r_list)
hz_list = rng.uniform(1, 100, size=number_of_nodes)

blocks = [block_discrete(r, f_hz) for r, f_hz in zip(r_list, hz_list)]
A = make_block_diag(blocks)

eigvals_A = np.linalg.eigvals(A)
for i, eig in enumerate(eigvals_A):
    r = np.abs(eig)
    theta = np.angle(eig)
    freq_hz = theta / (2 * np.pi * Ts)

A = random_similarity(A, seed=0)  # 一般形に変換

eigvals_A = np.linalg.eigvals(A)
for i, eig in enumerate(eigvals_A):
    r = np.abs(eig)
    theta = np.angle(eig)
    freq_hz = theta / (2 * np.pi * Ts)
    print(freq_hz, r)
A_both = A
for i, eig in enumerate(eigvals_A):
    x = np.log(eig)
    freq_hz = x.imag / (2 * np.pi * Ts)
# single-input: only one state receives input (change idx as desired)
n = A.shape[0]
B = np.ones((n, 1))
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
    node_color=plt.get_cmap('viridis')(0.2),
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
A_error_dictionary = {}
A_hat_dictionary = {}

######################################
# パッシブ入力（u=0, ws=[0,0]）
######################################
Qx_passive_list = []
A_hat_passive_list = []
B_hat_passive_list = []
eigvals_passive_list = []
A_error_passive_list = []

for trial in range(trial_num):
    Qx_passive, A_hat_passive, B_hat_passive = simulate_cov_for_omega_ws(
        A_both, B_both, [0, 0], T_total, T_burn, seed=trial, white_input=False
    )
    eigvals_passive, eigvecs_passive = np.linalg.eigh(Qx_passive)
    Qx_passive_list.append(Qx_passive)
    A_hat_passive_list.append(A_hat_passive)
    B_hat_passive_list.append(B_hat_passive)
    eigvals_passive_list.append(eigvals_passive)
    A_error_passive_list.append(np.linalg.norm(A - A_hat_passive, ord='fro'))

# 平均値を計算
Qx_passive = np.mean(Qx_passive_list, axis=0)
A_hat_passive = np.mean(A_hat_passive_list, axis=0)
B_hat_passive = np.mean(B_hat_passive_list, axis=0)
eigvals_passive = np.mean(eigvals_passive_list, axis=0)

eigvals_list.append(eigvals_passive)
freqs_list.append(np.angle(np.linalg.eigvals(A_hat_passive)) / (2 * np.pi * Ts))
Qx_list.append(Qx_passive)
A_hat_list.append(A_hat_passive)
B_hat_list.append(B_hat_passive)
inv_sum = np.sum(1.0 / np.linalg.eigvalsh(Qx_passive))
inv_sum_list.append(inv_sum)
inv_sum_dictionary[f"Passive"] = inv_sum
A_error_dictionary[f"Passive"] = A_error_passive_list
A_hat_dictionary["Passive"] = A_hat_passive_list

Qx_passive, A_hat_passive, B_hat_passive = simulate_cov_for_omega_ws(
    A_both, B_both, [0, 0], T_total, T_burn, seed=0, white_input=False, plot_signals=True
)

######################################
# ホワイトノイズ入力（u=0, ws=[0,0]）
######################################
Qx_noise_list = []
A_hat_noise_list = []
B_hat_noise_list = []
eigvals_noise_list = []
A_error_noise_list = []

for trial in range(trial_num):
    Qx_noise, A_hat_noise, B_hat_noise = simulate_cov_for_omega_ws(
        A_both, B_both, [0, 0], T_total, T_burn, seed=trial, white_input=False, uniform_input=True
    )
    eigvals_noise, eigvecs_noise = np.linalg.eigh(Qx_noise)
    Qx_noise_list.append(Qx_noise)
    A_hat_noise_list.append(A_hat_noise)
    B_hat_noise_list.append(B_hat_noise)
    eigvals_noise_list.append(eigvals_noise)
    A_error_noise_list.append(np.linalg.norm(A - A_hat_noise, ord='fro'))

# 平均値を計算
Qx_noise = np.mean(Qx_noise_list, axis=0)
A_hat_noise = np.mean(A_hat_noise_list, axis=0)
B_hat_noise = np.mean(B_hat_noise_list, axis=0)
eigvals_noise = np.mean(eigvals_noise_list, axis=0)

eigvals_list.append(eigvals_noise)
freqs_list.append(np.angle(np.linalg.eigvals(A_hat_noise)) / (2 * np.pi * Ts))
Qx_list.append(Qx_noise)
A_hat_list.append(A_hat_noise)
B_hat_list.append(B_hat_noise)
inv_sum = np.sum(1.0 / np.linalg.eigvalsh(Qx_noise))
inv_sum_list.append(inv_sum)
inv_sum_dictionary[f"Flat"] = inv_sum
A_error_dictionary[f"Flat"] = A_error_noise_list
A_hat_dictionary["Flat"] = A_hat_noise_list


for condition_label, A_hat_current in zip(["passive", "noise"], [A_hat_passive, A_hat_noise]):
    for it in range(num_iterations):
        eigvals_A_hat = np.linalg.eigvals(A_hat_current)
        freqs_A_hat = np.angle(eigvals_A_hat) / (2 * np.pi * Ts)
        freqs_A_hat = freqs_A_hat[freqs_A_hat >= 0]
        print(freqs_A_hat)
        ws_current = 2 * np.pi * freqs_A_hat / fs
        Qx_list_trial = []
        A_hat_list_trial = []
        B_hat_list_trial = []
        A_error_list_trial = []
        for trial in range(trial_num):
            Qx_trial, A_hat_trial, B_hat_trial = simulate_cov_for_omega_ws(
            A_both, B_both, ws_current, T_total, T_burn, seed=trial, white_input=False
            )
            Qx_list_trial.append(Qx_trial)
            A_hat_list_trial.append(A_hat_trial)
            B_hat_list_trial.append(B_hat_trial)
            A_error_list_trial.append(np.linalg.norm(A - A_hat_trial, ord='fro'))
        Qx = np.mean(Qx_list_trial, axis=0)
        A_hat_next = np.mean(A_hat_list_trial, axis=0)
        B_hat_next = np.mean(B_hat_list_trial, axis=0)

        eigvals_list.append(eigvals_A_hat)
        freqs_list.append(freqs_A_hat)
        Qx_list.append(Qx)
        A_hat_list.append(A_hat_next)
        B_hat_list.append(B_hat_next)
        inv_sum = np.sum(1.0 / np.linalg.eigvalsh(Qx))
        inv_sum_list.append(inv_sum)
        A_hat_current = A_hat_next  # 次のイテレーション用
        inv_sum_dictionary[f"{condition_label}-design{it+1}"] = inv_sum
        A_error_dictionary[f"{condition_label}-design{it+1}"] = A_error_list_trial
        A_hat_dictionary[f"{condition_label}-design{it+1}"] = A_hat_list_trial
        Qx_trial, A_hat_trial, B_hat_trial = simulate_cov_for_omega_ws(
            A_both, B_both, ws_current, T_total, T_burn, seed=it+1, white_input=False, plot_signals=True
        )


# 真のAの固有値から周波数を計算し、simulate_cov_for_omega_wsを複数trialで実行して平均を取る

eigvals_A = np.linalg.eigvals(A_both)
freqs_A = np.angle(eigvals_A) / (2 * np.pi * Ts)
# フィルタ: 0以上の周波数のみ残す
freqs_A = freqs_A[freqs_A >= 0]
print(freqs_A)
ws_A = 2 * np.pi * freqs_A / fs

Qx_trueA_list = []
A_hat_trueA_list = []
B_hat_trueA_list = []
A_error_trueA_list = []

for trial in range(trial_num):
    Qx_t, A_hat_t, B_hat_t = simulate_cov_for_omega_ws(
        A_both, B_both, ws_A, T_total, T_burn, seed=trial, white_input=False
    )
    Qx_trueA_list.append(Qx_t)
    A_hat_trueA_list.append(A_hat_t)
    B_hat_trueA_list.append(B_hat_t)
    A_error_trueA_list.append(np.linalg.norm(A - A_hat_t, ord='fro'))

# 平均値を計算
Qx_trueA = np.mean(Qx_trueA_list, axis=0)
A_hat_trueA = np.mean(A_hat_trueA_list, axis=0)
B_hat_trueA = np.mean(B_hat_trueA_list, axis=0)

eigvals_list.append(eigvals_A)
freqs_list.append(freqs_A)
Qx_list.append(Qx_trueA)
A_hat_list.append(A_hat_trueA)
B_hat_list.append(B_hat_trueA)
inv_sum = np.sum(1.0 / np.linalg.eigvalsh(Qx_trueA))
inv_sum_list.append(inv_sum)
inv_sum_dictionary["True"] = inv_sum
# baseline 用には平均誤差をスカラーで格納
A_error_dictionary["True"] = A_error_trueA_list
A_hat_dictionary["True"] = A_hat_trueA


# カラーパレット例（Nature系: muted, pastel, earth tones）
nature_colors = plt.get_cmap('viridis')(np.linspace(0, 1, 3))

# ラベル定義
labels_passive = ["Passive"] + [f"passive-design{i+1}" for i in range(num_iterations)]
labels_noise = ["Flat"] + [f"noise-design{i+1}" for i in range(num_iterations)]

# A_error_dictionaryの各エントリはリスト（trialごとの誤差）になっている想定で平均と標準偏差を計算
def mean_std_from_dict(label):
    v = A_error_dictionary.get(label, np.nan)
    if isinstance(v, (list, np.ndarray)):
        arr = np.asarray(v, dtype=float).ravel()
        if arr.size == 0:
            return np.nan, np.nan
        return np.nanmean(arr), np.nanstd(arr)
    else:
        # scalar or nan
        return np.nan, np.nan

means_passive, stds_passive = zip(*[mean_std_from_dict(lbl) for lbl in labels_passive])
means_noise, stds_noise = zip(*[mean_std_from_dict(lbl) for lbl in labels_noise])

x_ticks = [f"{i+1}" for i in range(len(labels_passive))]
x = np.arange(len(x_ticks))

plt.figure(figsize=(3, 2), dpi=150)
plt.plot(x, means_passive, 'o-', color=nature_colors[0],
         markerfacecolor='white', markeredgecolor=nature_colors[0], linewidth=2, markersize=6, label='Passive')
plt.plot(x, means_noise,  'o-', color=nature_colors[1],
         markerfacecolor='white', markeredgecolor=nature_colors[1], linewidth=2, markersize=6, label='Flat')

plt.xticks(x, x_ticks, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Passive側のみtext表示
for i, lbl in enumerate(labels_passive):
    if i == 0:
        text_label = "Passive"
        plt.text(i+0.24, means_passive[i]-0.05, text_label, fontsize=10, color=nature_colors[0], va='bottom', ha='left')
    else:
        text_label = f"design-{i}"
        plt.text(i, means_passive[i]+0.005, text_label, fontsize=10, color=nature_colors[0], va='bottom', ha='left')

# Flat最初のみラベル表示
plt.text(0, means_noise[0]-0.06, "Flat", fontsize=10, color=nature_colors[1], va='bottom', ha='left')

# ベースライン（True）の平均を横線で描画（存在しない場合は無視）
baseline_mean, baseline_std = mean_std_from_dict("True")
if not np.isnan(baseline_mean):
    plt.axhline(baseline_mean, color='black', linestyle='--', linewidth=1, label='Optimal')
    plt.text(len(x)-0.40, baseline_mean-0.02, f'Optimal', color='black', fontsize=10, va='bottom', ha='right')

plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Estimation error of A', fontsize=12)
plt.legend(loc='upper right', fontsize=10)

filename = f"{program_name}_A_error_dictionary_plot.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format="svg", bbox_inches='tight')
plt.show()

labels_passive = [f"passive-design{i+1}" for i in range(num_iterations)]
labels_noise = [f"noise-design{i+1}" for i in range(num_iterations)]

means_passive, stds_passive = zip(*[mean_std_from_dict(lbl) for lbl in labels_passive])
means_noise, stds_noise = zip(*[mean_std_from_dict(lbl) for lbl in labels_noise])

x_ticks = [f"{i+1}" for i in range(len(labels_passive))]
x = np.arange(len(x_ticks))

plt.figure(figsize=(1.5, 3), dpi=150)
plt.plot(x, means_passive, 'o-', color=nature_colors[0],
         markerfacecolor='white', markeredgecolor=nature_colors[0], linewidth=2, markersize=6, label='Passive')
plt.plot(x, means_noise,  'o-', color=nature_colors[1],
         markerfacecolor='white', markeredgecolor=nature_colors[1], linewidth=2, markersize=6, label='Flat')

# 軸ラベルと目盛
plt.xticks(x, x_ticks, fontsize=12)
plt.yticks(fontsize=12)

# 左右に余白を追加：明示的にxlimを広げる（または plt.margins(x=...) を使う）
pad = 0.5  # 点の間隔に対するパディング（必要に応じて調整）
plt.xlim(x[0] - pad, x[-1] + pad)
# もしくは相対マージンを使う場合: plt.gca().margins(x=0.1)

plt.tight_layout()

# legend不要 -> 削除
# 保存
filename = f"{program_name}_A_error_dictionary_plot_expansion.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format='svg', bbox_inches='tight')
plt.show()
plt.close()

# 上がpassive由来、下がFlat由来を1枚の図で並べる
keys_passive = ["Passive"] + [f"passive-design{i+1}" for i in range(num_iterations)]
keys_noise   = ["Flat"] + [f"noise-design{i+1}"   for i in range(num_iterations)]

cols = len(keys_passive)
rows = 2

# collect diffs to set common vmin/vmax
diffs = []
for key in keys_passive + keys_noise:
    Ahat_entry = A_hat_dictionary.get(key)
    if Ahat_entry is None:
        continue

    # A_hat は trial ごとの list かもしれない -> 平均を取る
    try:
        Ahat_arr = np.asarray(Ahat_entry)
        # handle object-dtype lists of arrays
        if Ahat_arr.dtype == object:
            Ahat_arr = np.stack(list(Ahat_entry))
    except Exception:
        try:
            Ahat_arr = np.stack(list(Ahat_entry))
        except Exception:
            continue

    # 平均を取る（trial軸がある場合）
    if Ahat_arr.ndim == 3:
        Ahat_mean = np.nanmean(Ahat_arr, axis=0)
    elif Ahat_arr.ndim == 2:
        Ahat_mean = Ahat_arr
    else:
        # unexpected shape
        continue

    if Ahat_mean.shape != A_both.shape:
        continue

    diffs.append(np.abs(Ahat_mean - A_both))

if len(diffs) == 0:
    raise RuntimeError("No valid A_hat matrices to compare with A_both.")

#%%
vmin = min(np.nanmin(m) for m in diffs)
vmax = max(np.nanmax(m) for m in diffs) - 0.004
if np.isnan(vmax) or vmax <= vmin:
    vmax = vmin * 10 + 1e-6

# Softer, cleaner white->muted-orange palette (less "dirty" / oversaturated)
cmap_white_red = LinearSegmentedColormap.from_list(
    "nature_error",
    ["#ffffff", "#fbf9f6", "#f7cfa6", "#f08b66", "#b23b2a"],
    N=256
)
fig, axs = plt.subplots(rows, cols, figsize=(2*cols, 4), dpi=150)

# helper to get mean Ahat for plotting
def get_mean_Ahat(key):
    entry = A_hat_dictionary.get(key)
    if entry is None:
        return None
    try:
        arr = np.asarray(entry)
        if arr.dtype == object:
            arr = np.stack(list(entry))
    except Exception:
        try:
            arr = np.stack(list(entry))
        except Exception:
            return None
    if arr.ndim == 3:
        return np.nanmean(arr, axis=0)
    elif arr.ndim == 2:
        return arr
    else:
        return None

# --- Save individual plots for each key: absolute diff and mean A_hat (viridis) ---
all_keys = keys_passive + keys_noise
for key in all_keys:
    Ahat_mean = get_mean_Ahat(key)
    if Ahat_mean is None or Ahat_mean.shape != A_both.shape:
        # create a placeholder image saying missing
        fig_miss, ax_miss = plt.subplots(figsize=(3, 3), dpi=150)
        ax_miss.text(0.5, 0.5, f"missing: {key}", ha='center', va='center', fontsize=12)
        ax_miss.set_xticks([]); ax_miss.set_yticks([])
        fname = f"{program_name}_missing_{key}.svg"
        plt.savefig(os.path.join(results_dir, fname), format='svg', bbox_inches='tight')
        plt.close(fig_miss)
        continue

    # Absolute difference plot (using custom white->orange cmap)
    diff = np.abs(Ahat_mean - A_both)
    fig_d, ax_d = plt.subplots(figsize=(4, 4), dpi=150)
    im_d = ax_d.imshow(diff, cmap=cmap_white_red, vmin=vmin, vmax=vmax, interpolation='nearest', aspect='equal')
    ax_d.set_xticks([]); ax_d.set_yticks([])
    fname = f"{program_name}_absdiff_{key}.svg"
    plt.savefig(os.path.join(results_dir, fname), format='svg', bbox_inches='tight')
    plt.close(fig_d)

    # Mean A_hat plot (viridis)
    fig_m, ax_m = plt.subplots(figsize=(4, 4), dpi=150)
    im_m = ax_m.imshow(Ahat_mean, cmap='viridis', interpolation='nearest', aspect='equal')
    ax_m.set_xticks([]); ax_m.set_yticks([])
    fname = f"{program_name}_Ahat_mean_{key}.svg"
    plt.savefig(os.path.join(results_dir, fname), format='svg', bbox_inches='tight')
    plt.close(fig_m)

# --- Combined grid of passive/noise as before ---
for j, key in enumerate(keys_passive):
    ax = axs[0, j] if rows > 1 else axs[j]
    Ahat_mean = get_mean_Ahat(key)
    if Ahat_mean is None or Ahat_mean.shape != A_both.shape:
        ax.text(0.5, 0.5, "missing", ha='center', va='center', fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        continue
    diff = np.abs(Ahat_mean - A_both)
    im = ax.imshow(diff, cmap=cmap_white_red, vmin=vmin, vmax=vmax, interpolation='nearest', aspect='equal')
    ax.set_title(str(key), fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

for j, key in enumerate(keys_noise):
    ax = axs[1, j]
    Ahat_mean = get_mean_Ahat(key)
    if Ahat_mean is None or Ahat_mean.shape != A_both.shape:
        ax.text(0.5, 0.5, "missing", ha='center', va='center', fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        continue
    diff = np.abs(Ahat_mean - A_both)
    im = ax.imshow(diff, cmap=cmap_white_red, vmin=vmin, vmax=vmax, interpolation='nearest', aspect='equal')
    ax.set_title(str(key), fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

# Leave space on the right for a vertical colorbar so it doesn't overlap plots
fig.subplots_adjust(left=0.05, right=0.90, top=0.95, bottom=0.05)

# Create a dedicated axis for the colorbar to avoid overlapping heatmaps
cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height] in figure coords
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Absolute difference', fontsize=10)

plt.tight_layout(rect=[0, 0, 0.90, 1])  # respect the right margin reserved for the colorbar
filename = f"{program_name}_Ahat_absdiff_passive_vs_noise.svg"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, format='svg', bbox_inches='tight')
plt.show()
plt.close(fig)
# %%
