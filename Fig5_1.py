#%%
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
from plot_heatmap import plot_heatmap
from set_default_plot_settings import set_default_plot_settings

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import pickle
import numpy as np
import pandas as pd
import scipy
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import os

results_dir = "results"
input_type = "cos"  
fs=100
dt = 1/fs
Trials = 10 # 5000 in the paper
inputTrials = 50 # 50 in the paper
input_values = np.arange(0.1, 50, 1)

#%% 
mode_freq = [10,40]
dim = len(mode_freq)*2

diagB = np.ones(dim)
B_true = np.diag(diagB)

# np.random.seed(0)  # For reproducibility
# B_true = np.random.rand(dim, dim)

desired_eigenvalues = []
for fIndex, f in enumerate(mode_freq):
    desired_eigenvalues.append([-50 -(2*np.pi*f)*1j, -50 +(2*np.pi*f)*1j])

print("Desired eigenvalues:", desired_eigenvalues)

A_continuous = np.zeros([dim, dim])
# Construct the continuous-time matrix A
for index, desired_eigenvalue in enumerate(desired_eigenvalues):
    A_continuous[2*index:2*(index+1), 2*index:2*(index+1)] = np.array([[desired_eigenvalue[0].real, desired_eigenvalue[0].imag],[desired_eigenvalue[1].imag, desired_eigenvalue[1].real]])# Calculate the eigenvalues of the continuous-time matrix
eigenvalues, _ = np.linalg.eig(A_continuous)

# Discretize the continuous-time matrix A
A_discrete = scipy.linalg.expm(A_continuous * dt)

print("Continuous-time matrix A:", A_continuous)
eigenvalues, eigenvectors = np.linalg.eig(A_continuous)

print("Eigenvalues of the continuous-time matrix A:", eigenvalues)
print("eigenvectors of the continuous-time matrix A:", eigenvectors)

print("Discrete-time matrix A:", A_discrete)

# Calculate and print the eigenvalues of the discrete-time matrix A
eigenvalues_discrete, eigenvectors_discrete = np.linalg.eig(A_discrete)
print("Eigenvalues of the discrete-time matrix A:", eigenvalues_discrete)
print("Eigenvectors of the discrete-time matrix A:", eigenvectors_discrete)


#%%
# Plot heatmap of A_discrete
set_default_plot_settings(font_size=24, dpi=200)
plt.figure(figsize=(5, 5))
plt.imshow(A_discrete, cmap='viridis', interpolation='nearest', vmin=-0.8)

# Set ticks at the middle of each element
plt.xticks(ticks=np.arange(A_discrete.shape[1]), labels=range(1, A_discrete.shape[1] + 1))
plt.yticks(ticks=np.arange(A_discrete.shape[0]), labels=range(1, A_discrete.shape[0] + 1))

# Set labels
plt.xlabel('Dimension')
plt.ylabel('Dimension')

# Display the values on the heatmap
for i in range(A_discrete.shape[0]):
    for j in range(A_discrete.shape[1]):
        value = A_discrete[i, j]
        if value == 0:
            plt.text(j, i, f"0.00", ha='center', va='center', color='black')
        elif abs(value) < 1e-2:
            plt.text(j, i, f"{value:.2e}", ha='center', va='center', color='black')
        else:
            plt.text(j, i, f"{value:.2f}", ha='center', va='center', color='black')

plt.tight_layout(pad=0.1)

# Save the heatmap as an SVG file
heatmap_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_A_discrete_heatmap_dim{dim}.svg")
plt.savefig(heatmap_filename, bbox_inches='tight')
print(f"Heatmap saved as {heatmap_filename}")

#%%
A_true = A_discrete
T = 1
sigma = 0.1

xi_trials = np.zeros((Trials, dim, fs*T))
for trial in range(Trials):
    for i in range(1, fs*T):
        xi_trials[trial, :, i-1] = np.random.normal(0, sigma, dim)
        
def compute_metrics(value, Trials, dim, fs, T, dt, A_true, B_true, input_type, test_on = False):
    fros_trial_A = []
    fros_trial_B = []
    fros_trial_Phi = []
    fros_W = []
    trace_inv_X_trial = []
    trace_inv_U_trial = []
    A_optimal_trial = []
    cond_num_trial = []
        
    for trial in range(Trials):
        Xs = []
        Ys = []
        Us = []
        Zs = []
        for trial in range(inputTrials):
            x0 = np.zeros([dim,1])
            x = np.zeros([dim, fs*T])
            xi = np.zeros([dim, fs*T])
            x[:,0] = x0[:,0]
            
            u0 = np.zeros(dim)  # Initial input values
            u0[np.random.choice(dim)] = 1
            if input_type=="impulse": #Impulse
                w = np.zeros(fs*T)
                w[0] = value
                u = np.outer(u0, w)
            elif input_type=="step": #Step
                w = np.ones(fs*T)*value
                u = np.outer(u0, w)
            elif input_type=="cos": #Cosine
                omega = 2*np.pi*value
                u = np.zeros([dim, fs*T])
                w = np.cos(omega * np.arange(0, T, dt))
                u = np.outer(u0, w)        
                    
            for i in range(1, fs*T):
                xi[:,i-1] = np.random.normal(0, sigma, dim)
                x[:,i] = A_true @ x[:,i-1] + B_true @ u[:,i-1] + xi[:,i-1]
                
            X = x[:,:-1]
            U = u[:,:-1]
            Y = x[:,1:]
                    
            Z = np.vstack([X, U])

            Xs.append(X)
            Us.append(U)
            Ys.append(Y)
            Zs.append(Z)
        
        X = np.hstack(Xs)
        U = np.hstack(Us)
        Y = np.hstack(Ys)
        Z = np.hstack(Zs)

        Phi_est = Y@scipy.linalg.pinv(Z)
        
        # Display the condition number of Z
        condition_number_Z = np.linalg.cond(Z)
        cond_num_trial.append(condition_number_Z)

        # Phi_est = W        
        A_est = Phi_est[:,:dim]
        B_est = Phi_est[:,dim:]
        
        W_true = A_true
        W_est = A_est
        
        Phi_true = np.hstack([A_true, B_true])
                        
        frobenius_norm_A = np.linalg.norm(A_true - A_est, 'fro') **2
        frobenius_norm_B = np.linalg.norm(B_true - B_est, 'fro') **2
        frobenius_norm_Phi = np.linalg.norm(Phi_true - Phi_est, 'fro') **2
        frobenius_norm_W = np.linalg.norm(W_true - W_est, 'fro') **2
        fros_trial_A.append(frobenius_norm_A)
        fros_trial_B.append(frobenius_norm_B)
        fros_trial_Phi.append(frobenius_norm_Phi)
        fros_W.append(frobenius_norm_W)

        try:
            A_optimal_trial.append(np.trace(np.linalg.inv(Z@Z.T)))
        except:
            A_optimal_trial.append(np.Inf)
            
        trace_inv_X_trial.append(0)
        trace_inv_U_trial.append(0)
            
    return np.mean(fros_trial_A), np.mean(fros_trial_B), np.mean(fros_trial_Phi), np.mean(A_optimal_trial), np.mean(trace_inv_X_trial), np.mean(trace_inv_U_trial), np.mean(fros_W), np.mean(cond_num_trial)

if True:
    num_cores = multiprocessing.cpu_count() - 3
    results = Parallel(n_jobs=num_cores)(delayed(compute_metrics)(value, Trials, dim, fs, T, dt, A_true, B_true, input_type) for value in tqdm(input_values))

fros_A = []
fros_B = []
fros_Phi = []
A_optimal = []
trace_inv_X = []
trace_inv_U = []
fros_W = []
cond_num = []
for fros_mean_A, fros_mean_B, fros_mean_Phi, A_optimal_mean, trace_inv_X_mean, trace_inv_U_mean, fros_mean_W, cond_num_mean in results:
    fros_A.append(fros_mean_A)
    fros_B.append(fros_mean_B)
    fros_Phi.append(fros_mean_Phi)    
    A_optimal.append(A_optimal_mean)
    trace_inv_X.append(trace_inv_X_mean)
    trace_inv_U.append(trace_inv_U_mean)
    fros_W.append(fros_mean_W)
    cond_num.append(cond_num_mean)


#%%
# Save variables to a pickle file
data_to_save = {
    'fros_A': fros_A,
    'fros_B': fros_B,
    'fros_Phi': fros_Phi,
    'fros_W': fros_W,
    'A_optimal': A_optimal,
    'trace_inv_X': trace_inv_X,
    'trace_inv_U': trace_inv_U,
    'input_values': input_values,
    'input_type': input_type,
    'A_continuous': A_continuous,
    'A_discrete': A_discrete,
    'mode_freq': mode_freq,
    'cond_num': cond_num,
    'input_values': input_values
}

# Ensure the results directory exists
import os
filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_{input_type}_dim{dim}.pkl")
with open(filename, 'wb') as f:
    pickle.dump(data_to_save, f)
    
#%%
filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_{input_type}_dim{dim}.pkl")
with open(filename, 'rb') as f:
    loaded_data = pickle.load(f)

fros_A = loaded_data['fros_A']
fros_B = loaded_data['fros_B']
fros_Phi = loaded_data['fros_Phi']
fros_W = loaded_data['fros_W']
A_optimal = loaded_data['A_optimal']
trace_inv_X = loaded_data['trace_inv_X']
trace_inv_U = loaded_data['trace_inv_U']
input_vectors = loaded_data['input_values']
input_type = loaded_data['input_type']
A_continuous = loaded_data['A_continuous']
A_discrete = loaded_data['A_discrete']
mode_freq = loaded_data['mode_freq']
input_values = loaded_data['input_values']

set_default_plot_settings(font_size=24, dpi=200, line_width=3)
if len(mode_freq) == 1:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
elif len(mode_freq) == 2:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
axs_index = 0
# Plot Frobenius Norm of A
axs[axs_index].plot(input_values, fros_A, label='Frobenius Norm of A', color="blue")
if input_type=="impulse":
    axs[axs_index].set_xlabel('Impulse Strength')
elif input_type=="step":
    axs[axs_index].set_xlabel('Step Strength')
elif input_type=="cos":
    axs[axs_index].set_xlabel('Frequency [Hz]')
axs[axs_index].set_ylabel(r'Error of $\hat{A}$')
axs[axs_index].grid(True)
if input_type=="cos":
    axs[axs_index].set_xticks(np.arange(min(input_values), max(input_values) + 1, 10)-0.1)
    for i, base_f in enumerate(mode_freq):
        axs[axs_index].axvline(x=base_f, color='red', linestyle='--', linewidth=1.5)
        axs[axs_index].text(base_f + mode_freq[0]*4e-1, max(fros_A)-max(fros_A)*0.6e-1, f'$\\theta_{{{i+1}}}$', ha='center', va='bottom', color='black')

# Plot A_optimal
axs_index += 1
axs[axs_index].plot(input_values, A_optimal, label='A_optimal', color="blue")
if input_type == "impulse":
    axs[axs_index].set_xlabel('Impulse Strength')
elif input_type == "step":
    axs[axs_index].set_xlabel('Step Strength')
elif input_type == "cos":
    axs[axs_index].set_xlabel('Frequency [Hz]')
axs[axs_index].set_ylabel(r'tr($\Sigma^{-1}_\mathbf{Z}$)')
axs[axs_index].grid(True)
if input_type == "cos":
    axs[axs_index].set_xticks(np.arange(min(input_values), max(input_values) + 1, 10)-0.1)
    for i, base_f in enumerate(mode_freq):
        axs[axs_index].axvline(x=base_f, color='red', linestyle='--', linewidth=1.5)
        axs[axs_index].text(base_f + mode_freq[0]*4e-1, max(A_optimal)-max(A_optimal)*0.6e-1, f'$\\theta_{{{i+1}}}$', ha='center', va='bottom', color='black')

# Set y-axis to integer values and add 1e-1 at the top
axs[axs_index].yaxis.set_major_formatter(ScalarFormatter())
axs[axs_index].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.tight_layout()
if len(mode_freq)==2:
    plt.subplots_adjust(wspace=0.7)
plt.show()

# Save the figure as an SVG file
svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_{input_type}_A_B_Phi_dim{dim}.svg")
fig.savefig(svg_filename)
print(f"Figure saved as {svg_filename}")
# %%
