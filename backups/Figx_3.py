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

r_list = [0.85, 0.9]
hz_list = [10, 40]

# ---- 1. Hz=hz_list[0]のみ ----
r_list_1 = r_list[0:1]
f_hz_list_1 = [hz_list[0]]
blocks_1 = [block_discrete(r, f) for r, f in zip(r_list_1, f_hz_list_1)]
A_1 = make_block_diag(blocks_1)
B_1 = np.eye(A_1.shape[0])

inv_lambda_1 = []
for w in ws:
    Qx, _, _, _ = simulate_cov_for_omega_knownB(A_1, B_1, w, T_total, T_burn)
    eigvals = np.linalg.eigvalsh(Qx)
    inv_lambda_1.append(1.0 / eigvals)
inv_lambda_1 = np.array(inv_lambda_1)  # shape: (num_f, num_eig)

# ---- 2. Hz=hz_list[1]のみ ----
r_list_2 = r_list[1:2]
f_hz_list_2 = [hz_list[1]]
blocks_2 = [block_discrete(r, f) for r, f in zip(r_list_2, f_hz_list_2)]
A_2 = make_block_diag(blocks_2)
B_2 = np.eye(A_2.shape[0])

inv_lambda_2 = []
for w in ws:
    Qx, _, _, _ = simulate_cov_for_omega_knownB(A_2, B_2, w, T_total, T_burn)
    eigvals = np.linalg.eigvalsh(Qx)
    inv_lambda_2.append(1.0 / eigvals)
inv_lambda_2 = np.array(inv_lambda_2)

# ---- 3. Hz=hz_list両方 ----
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

n = A.shape[0]

inv_lambda_both = []
for w in ws:
    Qx, E, _, _ = simulate_cov_for_omega_knownB(A_both, B_both, w, T_total, T_burn)
    eigvals = np.linalg.eigvalsh(Qx)
    inv_lambda_both.append(1.0 / eigvals)
inv_lambda_both = np.array(inv_lambda_both)

# ---- プロット ----
plt.figure(figsize=(8, 10))

plt.subplot(3, 2, 1)
for i in range(inv_lambda_1.shape[1]):
    plt.scatter(freqs, np.sum(inv_lambda_1, axis=1), color='blue', alpha=0.7)
plt.xlabel('Input frequency [Hz]')
plt.ylabel('sum(1/λ)')
plt.title(f'With Hz={hz_list[0]} mode')

plt.subplot(3, 2, 2)
for i in range(inv_lambda_2.shape[1]):
    plt.scatter(freqs, np.sum(inv_lambda_2, axis=1), color='blue', alpha=0.7)
plt.xlabel('Input frequency [Hz]')
plt.ylabel('sum(1/λ)')
plt.title(f'With Hz={hz_list[1]} mode')

plt.subplot(3, 2, 3)
plt.scatter(freqs, np.sum(inv_lambda_1, axis=1) + np.sum(inv_lambda_2, axis=1), color='green', alpha=0.7)
plt.xlabel('Input frequency [Hz]')
plt.ylabel('sum(1/λ)')
plt.title('Simple sum of 1/λ')
plt.yscale('log')

plt.subplot(3, 2, 4)
plt.scatter(freqs, np.sum(inv_lambda_both, axis=1), color='red', alpha=0.7)
plt.xlabel('Input frequency [Hz]')
plt.ylabel('sum(1/λ)')
plt.title(f'With both Hz={hz_list[0]} and Hz={hz_list[1]} modes')
plt.yscale('log')
    
plt.suptitle(f"r={r_list}, freq={hz_list} Hz")
plt.tight_layout()
plt.show()

#%%
# 25Hzの入力で各系のxをプロット
input_freq = 25  # Hz
w_25Hz = 2 * np.pi * input_freq / fs

# シミュレーション
x_1 = np.zeros((A_1.shape[0], T_total))
x_2 = np.zeros((A_2.shape[0], T_total))
x_both = np.zeros((A_both.shape[0], T_total))

rng = np.random.default_rng(0)
noise = rng.normal(scale=1e-2, size=(max(A_1.shape[0], A_2.shape[0], A_both.shape[0]), T_total-1))

for t in range(T_total-1):
    u = np.cos(w_25Hz * t) * np.ones(A_1.shape[0])
    x_1[:, t+1] = A_1 @ x_1[:, t] + B_1 @ u + noise[:A_1.shape[0], t]

    u2 = np.cos(w_25Hz * t) * np.ones(A_2.shape[0])
    x_2[:, t+1] = A_2 @ x_2[:, t] + B_2 @ u2 + noise[A_1.shape[0]:, t]

    ub = np.cos(w_25Hz * t) * np.ones(A_both.shape[0])
    x_both[:, t+1] = A_both @ x_both[:, t] + B_both @ ub + noise[:A_both.shape[0], t]

# プロット: A_1
plt.figure(figsize=(8, 4))
for i in range(x_1.shape[0]):
    plt.subplot(1, x_1.shape[0], i+1)
    plt.plot(np.arange(T_burn, T_total), x_1[i, T_burn:])
    plt.title(f'A_1 dim {i+1}')
    plt.xlabel('Time step')
plt.suptitle('A_1, input 25Hz (after T_burn)')
plt.tight_layout()
plt.show()

# プロット: A_2
plt.figure(figsize=(8, 4))
for i in range(x_2.shape[0]):
    plt.subplot(1, x_2.shape[0], i+1)
    plt.plot(np.arange(T_burn, T_total), x_2[i, T_burn:])
    plt.title(f'A_2 dim {i+1}')
    plt.xlabel('Time step')
plt.suptitle('A_2, input 25Hz (after T_burn)')
plt.tight_layout()
plt.show()

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

# 各 x の共分散行列をプロット（値を重ねて表示）
plt.figure(figsize=(12, 4))

Qx_1 = np.cov(x_1[:, T_burn:])
plt.subplot(1, 3, 1)
im1 = plt.imshow(Qx_1, cmap='viridis')
plt.title('Cov(x_1)')
plt.colorbar(im1)
for i in range(Qx_1.shape[0]):
    for j in range(Qx_1.shape[1]):
        plt.text(j, i, f"{Qx_1[i, j]:.2e}", ha='center', va='center', color='white' if Qx_1[i, j] < Qx_1.max()/2 else 'black', fontsize=8)

Qx_2 = np.cov(x_2[:, T_burn:])
plt.subplot(1, 3, 2)
im2 = plt.imshow(Qx_2, cmap='viridis')
plt.title('Cov(x_2)')
plt.colorbar(im2)
for i in range(Qx_2.shape[0]):
    for j in range(Qx_2.shape[1]):
        plt.text(j, i, f"{Qx_2[i, j]:.2e}", ha='center', va='center', color='white' if Qx_2[i, j] < Qx_2.max()/2 else 'black', fontsize=8)

Qx_both = np.cov(x_both[:, T_burn:])
plt.subplot(1, 3, 3)
im3 = plt.imshow(Qx_both, cmap='viridis')
plt.title('Cov(x_both)')
plt.colorbar(im3)
for i in range(Qx_both.shape[0]):
    for j in range(Qx_both.shape[1]):
        plt.text(j, i, f"{Qx_both[i, j]:.2e}", ha='center', va='center', color='white' if Qx_both[i, j] < Qx_both.max()/2 else 'black', fontsize=8)

plt.suptitle('Covariance matrices of x (input 25Hz, after T_burn)')
plt.tight_layout()
plt.show()


#%%
# 2つの入力周波数 w1, w2 を持つ場合の inv_lambda_both をヒートマップでプロット

num_f = int(fs//2)
freqs = np.arange(1, fs//2 + 1, 1)
ws = 2 * np.pi * freqs / fs

inv_lambda_both_2d = np.zeros((num_f, num_f, n))  # (w1, w2, eig)

for i, w1 in enumerate(ws):
    for j, w2 in enumerate(ws):
        # 2つの入力: u[t] = cos(w1*t) + cos(w2*t)
        def simulate_cov_for_omega2(A, B, w1, w2, T_total, T_burn, seed=0):
            rng = np.random.default_rng(seed)
            n = A.shape[0]
            sigma = 1e-2

            x = np.zeros((n, T_total))
            noise = rng.normal(scale=sigma, size=(n, T_total-1))
            for t in range(T_total-1):
                u = np.cos(w1 * t) + np.cos(w2 * t)
                x[:, t+1] = A @ x[:, t] + B[:,0]*u + noise[:, t]
            X_prev = x[:, T_burn-1:T_total-1]
            Qx = np.cov(X_prev)
            return Qx

        Qx = simulate_cov_for_omega2(A_both, B_both, w1, w2, T_total, T_burn)
        eigvals = np.linalg.eigvalsh(Qx)
        inv_lambda_both_2d[i, j, :] = 1.0 / eigvals

# sum(1/λ) over all eigenvalues
sum_inv_lambda_2d = np.sum(inv_lambda_both_2d, axis=2)  # shape: (num_f, num_f)

#%%
plt.figure(figsize=(8, 6))
plt.imshow(np.log10(sum_inv_lambda_2d), extent=[freqs[0], freqs[-1], freqs[0], freqs[-1]],
           origin='lower', aspect='auto', cmap='viridis')
plt.xlabel('Input frequency 1 [Hz]')
plt.colorbar(label='log10(sum(1/λ))')
plt.ylabel('Input frequency 2 [Hz]')
plt.title(f'r={r_list}, freq={hz_list} Hz \n combination stimuli')
plt.show()

# 最小のインデックスを表示
min_idx = np.unravel_index(np.argmin(sum_inv_lambda_2d), sum_inv_lambda_2d.shape)
print(f"最小値のインデックス: {min_idx}, 周波数: ({freqs[min_idx[0]]:.2f} Hz, {freqs[min_idx[1]]:.2f} Hz)")
# %%
