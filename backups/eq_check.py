#%% x(t)の左辺、右辺の比較
import numpy as np

# パラメータ設定
np.random.seed(0)
n = 3  # 系の次元
L = 2  # 入力の周波数の個数
t = 10

# 安定な行列Aを生成
A = np.random.randn(n, n) * 0.2
eigvals, VL = np.linalg.eig(A)
VR = np.linalg.inv(VL)  # 左固有ベクトル
B = np.random.randn(n, 1)
u0 = np.random.randn(1, 1)
omegas = np.linspace(0.3, 1.0, L)

# --- 左辺: 直接シミュレーション ---
def simulate_lhs(A, B, u0, omegas, t):
    x = np.zeros((A.shape[0], 1))
    for k in range(t):
        u = sum(np.cos(omega * k) * u0 for omega in omegas)
        x = A @ x + B @ u
    return x

lhs = simulate_lhs(A, B, u0, omegas, t)

# --- 右辺: 固有分解式 ---
# 固有ベクトルを正規化して biorthogonal に
eigvals, V = np.linalg.eig(A)
W = np.linalg.inv(V)
rhs = np.zeros_like(lhs, dtype=complex)

for l, omega_l in enumerate(omegas, start=1):
    for d, (lam, v_d) in enumerate(zip(eigvals, V.T)):
        r_d = np.abs(lam)
        theta_d = np.angle(lam)
        num = (
            r_d**(t+1)*np.exp(1j*theta_d*(t-1))
            - r_d**t*np.exp(1j*(theta_d*t - omega_l))
            - r_d*np.exp(1j*(omega_l*t - theta_d))
            + np.exp(1j*omega_l*(t-1))
        )
        den = r_d**2 - 2*r_d*np.cos(theta_d - omega_l) + 1
        rhs += (num/den) * v_d.reshape(-1,1) * (W[d,:] @ B @ u0)

rhs = np.real(rhs)

# --- 比較 ---
print("LHS:\n", np.round(lhs, 6))
print("RHS:\n", np.round(rhs, 6))
print("差分ノルム:", np.linalg.norm(lhs - rhs))

#%% 周波数が異なると打ち消し合う確認
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# ---- Parameters: feel free to tweak ------------------------------
D = 4            # number of d-values
L = 6            # number of l-values (frequencies)
T = 4000         # time steps for time averaging (bigger -> tighter to 0)

rng = np.random.default_rng(42)
r_d      = 0.98 + 0.015 * rng.random(D)          # keep < 1 so it stays bounded-ish
theta_d  = 2*np.pi * rng.random(D)               # phases for d
# make separated frequencies
omega_l  = 2*np.pi * (0.03 + 0.18*rng.random(L)) # [0.03, 0.21]*2π

# ---- Build Psi(d,l,t) exactly as in the user's formula -----------
t = np.arange(1, T+1)[:, None, None]  # (T,1,1)
rd  = r_d[None, :, None]              # (1,D,1)
thd = theta_d[None, :, None]          # (1,D,1)
oml = omega_l[None, None, :]          # (1,1,L)

Psi = (rd**(t+1)) * np.exp(1j*thd*(t-1)) \
    - (rd**t)     * np.exp(1j*(thd*t - oml)) \
    - rd          * np.exp(1j*(oml*t - thd)) \
    + np.exp(1j*oml*(t-1))             # (T,D,L), complex

# ---- Time-averaged inner products: <Psi_{d,l}, Psi_{d',l'}>_t ----
# reshape to (T, D*L)
Psi_flat = Psi.reshape(T, D*L)  # (T, P), P=D*L
# Gram matrix over time average: (1/T) * sum_t Psi(t,:)*Psi(t,:).conj()^T
G = (Psi_flat.conj().T @ Psi_flat) / T  # (P,P), complex

# ---- Aggregate by frequency blocks to test "different freq -> 0" --
# For each pair (l, l'), average over all d,d' the block magnitude
# Block indices for (d,l) in flattened: idx = d*L + l
def block_avg_abs(l1, l2):
    rows = [d*L + l1 for d in range(D)]
    cols = [d*L + l2 for d in range(D)]
    sub = G[np.ix_(rows, cols)]
    return np.mean(np.abs(sub))

freq_block = np.zeros((L, L))
for l1 in range(L):
    for l2 in range(L):
        freq_block[l1, l2] = block_avg_abs(l1, l2)

df = pd.DataFrame(freq_block, 
                  index=[f"l={i}" for i in range(L)],
                  columns=[f"l'={j}" for j in range(L)])

# Key metrics
diag_mean = np.mean(np.diag(freq_block))
off_mean  = (np.sum(freq_block) - np.sum(np.diag(freq_block))) / (L*L - L)
off_max   = np.max(freq_block + np.eye(L)*(-1e9))  # max off-diagonal

summary = pd.DataFrame({
    "metric": ["diag mean |⟨Ψ_l,Ψ_l⟩|", "off-diag mean |⟨Ψ_l,Ψ_l'⟩|", "off-diag max |⟨Ψ_l,Ψ_l'⟩|"],
    "value": [diag_mean, off_mean, off_max]
})

# Also print a few numerical values
print("Diagonal mean (same frequency)      :", diag_mean)
print("Off-diagonal mean (different freq.) :", off_mean)
print("Off-diagonal max (different freq.)  :", off_max)
