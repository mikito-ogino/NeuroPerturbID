#%%

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
from plot_heatmap import plot_heatmap
from set_default_plot_settings import set_default_plot_settings
from rand_matrix_c_specified_number_of_large_eigenvalues import rand_matrix_c_specified_number_of_large_eigenvalues


import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import pickle
import numpy as np
import scipy
from tqdm import tqdm
import networkx as nx
import os
import random
import copy

results_dir = "results"
fs=128
dt = 1/fs
input_type = "cos"
input_strength = 1e5
T=30
sigma = 1e-20
dim = 32
obs_dim = dim
Trials = 10
T_for_gramian = fs

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

base_fs = []
diagB = np.ones(dim)
B_true_full = np.diag(diagB)
A_discrete, A_continuous = rand_matrix_c_specified_number_of_large_eigenvalues(dim, 20, dt)
    
eigenvalues, eigenvectors = np.linalg.eig(A_continuous)

# Display the real and imaginary parts of the eigenvalues
print("Real parts of the eigenvalues:", eigenvalues.real)
print("Imaginary parts of the eigenvalues divided by 2π:", np.round(eigenvalues.imag / (2 * np.pi)).astype(int))

# Calculate and print the eigenvalues of the discrete-time matrix A
eigenvalues_discrete, eigenvectors_discrete = np.linalg.eig(A_discrete)
print("Eigenvalues of the discrete-time matrix A:", eigenvalues_discrete)

#%
#Create a directed graph from the discrete-time matrix A
threshold = 1.5  # Set the threshold for strong connections

# Create a directed graph from the discrete-time matrix A with strong connections only
G = nx.DiGraph()
for i in range(A_discrete.shape[0]):
    for j in range(A_discrete.shape[1]):
        if abs(A_discrete[i, j]) > threshold:
            G.add_edge(j, i, weight=A_discrete[i, j])

# Set the diagonal elements of G to 0
for node in G.nodes():
    if G.has_edge(node, node):
        G.remove_edge(node, node)

# Relabel nodes to start from 1
mapping = {i: i + 1 for i in range(A_discrete.shape[0])}
G = nx.relabel_nodes(G, mapping)

# Plot the graph
set_default_plot_settings(font_size=12)
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=41, k=1)  # positions for all nodes, k controls the distance between nodes

nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='black', node_size=500, font_size=12, font_weight='bold', arrows=True)
svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_network_dim{dim}.svg")
plt.savefig(svg_filename)
print(f"Figure saved as {svg_filename}")
plt.show()
#%%
def plot_time_series(X, results_dir, filename):
    set_default_plot_settings(font_size=36, dpi=200)
    plt.figure(figsize=(9, 6))
    for i in range(dim):
        plt.plot(X[i, :]*3e-6 - i * 10, label=f'State Variable {i+1}', color='blue', alpha=1)
    plt.xlabel('Time Steps')
    plt.ylabel('State Variables')
    plt.yticks([])
    plt.tight_layout(pad=0.1)
    svg_filename = os.path.join(results_dir, filename)
    plt.savefig(svg_filename, bbox_inches='tight')
    print(f"Figure saved as {svg_filename}")
    plt.show()
#%
A_true_full = A_discrete
A_true = A_true_full[:obs_dim,:obs_dim]
B_true = B_true_full[:obs_dim,:obs_dim]

G = copy.deepcopy(A_true)
np.fill_diagonal(G, 0)
weighted_degree_centrality = np.sum(np.abs(G), axis=0)
top_5_indices = np.argsort(weighted_degree_centrality)[-5:][::-1]
print("Top 5 indices with highest weighted degree centrality:", top_5_indices)

xi_trials = np.zeros((Trials, dim, fs*T))
for trial in range(Trials):
    for i in range(1, fs*T):
        xi_trials[trial, :, i-1] = np.random.normal(0, sigma, dim)


#%%Passive
np.random.seed(42)
random.seed(42)

fros_trial = []
A_optimal_trial = []

A_est_trials = []
fros_A_trial = []
fros_W_trial = []
fros_cost_trial = []

for trial in range(Trials):
    x0 = np.ones([dim,1])
    x = np.zeros([dim, fs*T])
    xi = np.zeros([dim, fs*T])
    x[:,0] = x0[:,0]
    
    # オイラー法の実装
    for i in range(1, fs*T):
        xi[:,i-1] = xi_trials[trial, :, i-1]
        x[:,i] = A_true_full @ x[:,i-1] + xi[:,i-1]
        
    X_ = x[:, -fs*T//2:]
        
    X = X_[:obs_dim,:-1]
    Y = X_[:obs_dim,1:]
    
    # 行列 A の推定
    A_est = Y@scipy.linalg.pinv(X)

    if trial==0:
        plot_time_series(X, results_dir, f"{__file__.split('/')[-1].split('.')[0]}_state_variables_passive.svg")
    
    A_est_trials.append(A_est)
    fros_A_trial.append(np.linalg.norm(A_true - A_est, 'fro') **2)

    A_optimal_trial.append(np.trace(scipy.linalg.inv(X@X.T)))        
    
A_est_mean = np.mean(A_est_trials, axis=0)
passive_A_est = A_est_mean
passive_B_est = np.eye(obs_dim)

print("Passive Frobenius Norm:", np.mean(fros_A_trial))
passive_fro = np.mean(fros_A_trial)
print("A_optimal:", np.mean(A_optimal_trial))
passive_A_optimal = np.mean(A_optimal_trial)
passive_fro_W = np.mean(fros_W_trial)
passive_cost = np.mean(fros_cost_trial)


#%% Random Control
np.random.seed(42)
random.seed(42)

value = 20
hub = 6

A_optimal_trial = []
A_est_trials = []
B_est_trials = []
fros_A_trial = []
fros_B_trial = []
fros_W_trial = []
fros_cost_trial = []
fros_phi_trial = []

Ys = []
Zs = []
for trial in range(Trials):
    x0 = np.zeros([dim,1])
    x = np.zeros([dim, fs*T])
    xi = np.zeros([dim, fs*T])
    x[:,0] = x0[:,0]
    
    t = np.arange(0, T, dt)
    u0 = np.zeros(dim)
    
    u0[hub] = 1
    
    if input_type=="impulse": #Impulse
        w = np.zeros(fs*T)
        w[fs*T//2] = input_strength
    elif input_type=="step": #Step
        w = np.ones(fs*T)*value
    elif input_type=="cos": #Cosine
        omega = 2*np.pi*value
        w = input_strength*np.cos(omega * np.arange(0, T, dt))
    u = np.outer(u0, w)
        
    # オイラー法の実装
    for i in range(1, fs*T):
        xi[:,i-1] = xi_trials[trial, :, i-1]
        x[:,i] = A_true_full @ x[:,i-1] + B_true_full @ u[:,i-1] + xi[:,i-1]
        
    X_ = x[:, -fs*T//2:]
    U_ = u[:, -fs*T//2:]
        
    X = X_[:obs_dim,:-1]
    U = U_[:obs_dim,:-1]
    Y = X_[:obs_dim,1:]
            
    Z = np.vstack([X, U])
    
    if trial==0:
        plot_time_series(X_, results_dir, f"{__file__.split('/')[-1].split('.')[0]}_state_variables_random.svg")

    Ys.append(Y)
    Zs.append(Z)

Y = np.hstack(Ys)
Z = np.hstack(Zs)
                                
Phi_est = Y@scipy.linalg.pinv(Z)

A_est = Phi_est[:,:obs_dim]
B_est = Phi_est[:,obs_dim:]

A_est_trials.append(A_est)
B_est_trials.append(B_est)

frobenius_norm_A = np.linalg.norm(A_true - A_est, 'fro') **2
fros_A_trial.append(frobenius_norm_A)

frobenius_norm_B = np.linalg.norm(B_true - B_est, 'fro') **2
fros_B_trial.append(frobenius_norm_B)

frobenius_norm_phi = np.linalg.norm(Phi_est - np.hstack([A_true, B_true]), 'fro') **2
fros_phi_trial.append(frobenius_norm_phi)
            
A_optimal_trial.append(np.trace(scipy.linalg.pinv(Z@Z.T)))

A_est_mean = np.mean(A_est_trials, axis=0)
random_A_est = A_est_mean
B_est_mean = np.mean(B_est_trials, axis=0)
random_B_est = B_est_mean

print("Perturb Frobenius Norm:", np.mean(fros_A_trial))
random_fro = np.mean(fros_A_trial)
print("Perturb Frobenius Norm for B:", np.mean(fros_B_trial))
random_fro_B = np.mean(fros_B_trial)
print("Perturb Frobenius Norm for Phi:", np.mean(fros_phi_trial))
random_fro_phi = np.mean(fros_phi_trial)

print("A_optimal:", np.mean(A_optimal_trial))
random_A_optimal = np.mean(A_optimal_trial)
print("Xi",np.trace(np.cov(xi)))
print("Theory",1/(fs*T)*np.trace(np.cov(xi))*np.trace(scipy.linalg.pinv(Z@Z.T)))

random_freq = value
random_nodes = hub
random_fro_W = np.mean(fros_W_trial)
random_cost = np.mean(fros_cost_trial)


#%% Optimal Control Design
np.random.seed(42)
random.seed(42)

optimal_fros = []
design_A_optimal = []
design_fro_Ws = []
design_costs = []
design_freqs = []
design_nodeses = []
A_ests = []
for design_number in range(1, 3):
    # 入力ベクトル最適化
    import copy
    G = copy.deepcopy(A_est_mean)
    np.fill_diagonal(G, 0)
    weighted_degree_centrality = np.sum(np.abs(G), axis=0)
    # print(weighted_degree_centrality)
    hub = np.argmax(weighted_degree_centrality)
    print("Hub node:", hub)
    hubs = range(hub-5, hub)
    
    # 入力周波数最適化
    eigenvalues, _ = np.linalg.eig(A_est_mean)
    eigenvalues_continuous = np.log(eigenvalues) / dt
        
    # Get the minimum and maximum imaginary parts of the eigenvalues
    min_imaginary_part = 0.1
    max_imaginary_part = np.max(abs(eigenvalues_continuous.imag))
    print("Minimum imaginary part of the eigenvalues:", min_imaginary_part)
    print("Maximum imaginary part of the eigenvalues:", max_imaginary_part)    
    
    if input_type=="cos":
        inv_cov_X = {}
        A_errors = {}
        for value in tqdm(np.arange(1,fs//2)):
            Ys = []
            Zs = []
            Xs = []
            for trial in range(Trials):
                x0 = np.zeros([dim,1])
                x = np.zeros([dim, fs*T])
                xi = np.zeros([dim, fs*T])
                x[:,0] = x0[:,0]
                
                t = np.arange(0, T, dt)
                u0 = np.zeros(dim)
                
                u0[hub] = 1
                
                if input_type=="impulse": #Impulse
                    w = np.zeros(fs*T)
                    w[fs*T//2] = input_strength
                elif input_type=="step": #Step
                    w = np.ones(fs*T)*value
                elif input_type=="cos": #Cosine
                    omega = 2*np.pi*value
                    w = input_strength*np.cos(omega * np.arange(0, T, dt))
                u = np.outer(u0, w)

                # オイラー法の実装
                for i in range(1, fs*T):
                    xi[:,i-1] = 0
                    x[:,i] = A_true_full @ x[:,i-1] + B_true_full @ u[:,i-1] + xi[:,i-1]
                    
                X_ = x[:, -fs*T//2:]
                U_ = u[:, -fs*T//2:]
                    
                X = X_[:obs_dim,:-1]
                U = U_[:obs_dim,:-1]
                Y = X_[:obs_dim,1:]
                        
                Z = np.vstack([X, U])
                
                Ys.append(Y)
                Zs.append(Z)
                Xs.append(X)
                
            Y = np.hstack(Ys)
            Z = np.hstack(Zs)
            X = np.hstack(Xs)
                
            inv_cov_X[value] = np.trace(scipy.linalg.pinv(Z@Z.T))
            Phi_est = Y@scipy.linalg.pinv(Z)
    
            A_est = Phi_est[:,:obs_dim]
            
            A_errors[value] = np.linalg.norm(A_true - A_est, 'fro') **2
        
        value = min(A_errors, key=A_errors.get)

    A_optimal_trial = []
    A_est_trials = []
    B_est_trials = []
    fros_A_trial = []
    fros_B_trial = []
    fros_W_trial = []
    fros_cost_trial = []
    fros_phi_trial = []
    Ys = []
    Zs = []

    for trial in range(Trials):
        x0 = np.zeros([dim,1])
        x = np.zeros([dim, fs*T])
        xi = np.zeros([dim, fs*T])
        x[:,0] = x0[:,0]
        
        t = np.arange(0, T, dt)
        u0 = np.zeros(dim)
        if trial<dim:
            u0[trial % dim] = 1
        else:
            u0[hub] = 1
        if input_type=="impulse": #Impulse
            w = np.zeros(fs*T)
            w[fs*T//2] = input_strength
        elif input_type=="step": #Step
            w = np.ones(fs*T)*value
        elif input_type=="cos": #Cosine
            omega = 2*np.pi*value
            w = input_strength*np.cos(omega * np.arange(0, T, dt))
        u = np.outer(u0, w)
        
        for i in range(1, fs*T):
            xi[:,i-1] = xi_trials[trial, :, i-1]
            x[:,i] = A_true_full @ x[:,i-1] + B_true_full @ u[:,i-1] + xi[:,i-1]
            
        X_ = x[:, -fs*T//2:]
        U_ = u[:, -fs*T//2:]
            
        X = X_[:obs_dim,:-1]
        U = U_[:obs_dim,:-1]
        Y = X_[:obs_dim,1:]
                
        Z = np.vstack([X, U])

        Ys.append(Y)
        Zs.append(Z)
                
        if trial==0:
            plot_time_series(X_, results_dir, f"{__file__.split('/')[-1].split('.')[0]}_state_variables_design_{design_number}.svg")
            
    Y = np.hstack(Ys)
    Z = np.hstack(Zs)

                                
    Phi_est = Y@scipy.linalg.pinv(Z)
    
    A_est = Phi_est[:,:obs_dim]
    B_est = Phi_est[:,obs_dim:]
    
    A_est_trials.append(A_est)
    B_est_trials.append(B_est)
    
    frobenius_norm_A = np.linalg.norm(A_true - A_est, 'fro') **2
    fros_A_trial.append(frobenius_norm_A)
    
    frobenius_norm_B = np.linalg.norm(B_true - B_est, 'fro') **2
    fros_B_trial.append(frobenius_norm_B)
        
    frobenius_norm_phi = np.linalg.norm(Phi_est - np.hstack([A_true, B_true]), 'fro') **2
    fros_phi_trial.append(frobenius_norm_phi)
    
    A_optimal_trial.append(np.trace(np.linalg.pinv(Z@Z.T)))
            
    A_est_mean = np.mean(A_est_trials, axis=0)
    design_A_est = A_est_mean
    B_est_mean = np.mean(B_est_trials, axis=0)
    design_B_est = B_est_mean


    print("Hz:",value,"Perturb Frobenius Norm:", np.mean(fros_A_trial))
    optimal_fro = np.mean(fros_A_trial)
    print("Perturb Frobenius Norm for B:", np.mean(fros_B_trial))
    optimal_fro_B = np.mean(fros_B_trial)
    print("Perturb Frobenius Norm for Phi:", np.mean(fros_phi_trial))
    optimal_fro_phi = np.mean(fros_phi_trial)

    print("A_optimal:", np.mean(A_optimal_trial))
    design_A_optimal.append(np.mean(A_optimal_trial))
    
    print("Perturb Frobenius Norm for W:",fros_W_trial)
    
    design_fro_Ws.append(np.mean(fros_W_trial))
    design_costs.append(np.mean(fros_cost_trial))

    design_freq = value
    design_nodes = hub
    
    design_freqs.append(design_freq)
    design_nodeses.append(design_nodes)

    optimal_fros.append(optimal_fro)
    
    A_ests.append(A_est_mean)
    
#%
print(random_freq, random_nodes)
print(design_freqs, design_nodeses)

#%%
# Save variables to a pickle file
variables_to_save = {
    'A_true_full': A_true_full,
    'A_true': A_true,
    'B_true': B_true,
    'passive_A_est': passive_A_est,
    'passive_B_est': passive_B_est,
    'passive_fro': passive_fro,
    'passive_A_optimal': passive_A_optimal,
    'passive_fro_W': passive_fro_W,
    'passive_cost': passive_cost,
    'random_A_est': random_A_est,
    'random_B_est': random_B_est,
    'random_fro': random_fro,
    'random_fro_B': random_fro_B,
    'random_fro_phi': random_fro_phi,
    'random_A_optimal': random_A_optimal,
    'random_freq': random_freq,
    'random_nodes': random_nodes,
    'random_fro_W': random_fro_W,
    'random_cost': random_cost,
    'design_A_est': design_A_est,
    'design_B_est': design_B_est,
    'optimal_fros': optimal_fros,
    'design_A_optimal': design_A_optimal,
    'design_fro_Ws': design_fro_Ws,
    'design_costs': design_costs,
    'design_freqs': design_freqs,
    'design_nodeses': design_nodeses,
    'A_ests': A_ests,
}

with open(os.path.join(results_dir, 'variables.pkl'), 'wb') as f:
    pickle.dump(variables_to_save, f)

print(f"Variables saved to {os.path.join(results_dir, 'variables.pkl')}")

# Load variables from the pickle file
with open(os.path.join(results_dir, 'variables.pkl'), 'rb') as f:
    loaded_variables = pickle.load(f)


#%%
# Plot True A
vmin_ = -3
vmax_ = 3
set_default_plot_settings(font_size=24, dpi=200)
plt.figure(figsize=(5, 5))
plt.imshow(A_true_full, cmap='viridis', interpolation='nearest', vmin=vmin_, vmax=vmax_)
plt.xticks([])
plt.yticks([])
cbar = plt.colorbar(orientation='vertical', pad=0.05, shrink=0.65)
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_true_A.svg"))
plt.show()

set_default_plot_settings(font_size=24, dpi=200)
plt.figure(figsize=(5, 5))
plt.imshow(passive_A_est, cmap='viridis', interpolation='nearest', vmin=vmin_, vmax=vmax_)
plt.xticks([])
plt.yticks([])
cbar = plt.colorbar(orientation='vertical', pad=0.05, shrink=0.65)
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_passive_A.svg"))
plt.show()

set_default_plot_settings(font_size=24, dpi=200)
plt.figure(figsize=(5, 5))
plt.imshow(random_A_est, cmap='viridis', interpolation='nearest', vmin=vmin_, vmax=vmax_)
plt.xticks([])
plt.yticks([])
cbar = plt.colorbar(orientation='vertical', pad=0.05, shrink=0.65)
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_random_A.svg"))
plt.show()

set_default_plot_settings(font_size=24, dpi=200)
plt.figure(figsize=(5, 5))
plt.imshow(A_ests[0], cmap='viridis', interpolation='nearest', vmin=vmin_, vmax=vmax_)
plt.xticks([])
plt.yticks([])
cbar = plt.colorbar(orientation='vertical', pad=0.05, shrink=0.65)
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_design1_A.svg"))
plt.show()

set_default_plot_settings(font_size=24, dpi=200)
plt.figure(figsize=(5, 5))
plt.imshow(A_ests[1], cmap='viridis', interpolation='nearest', vmin=vmin_, vmax=vmax_)
plt.xticks([])
plt.yticks([])
cbar = plt.colorbar(orientation='vertical', pad=0.05, shrink=0.65)
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_design2_A.svg"))
plt.show()

#% Frobenius Norms Comparison Bar Plot
set_default_plot_settings(font_size=18, dpi=200)
fig, ax = plt.subplots(figsize=(10, 6))

# Frobenius Norms
values_fro = [passive_fro, random_fro] + optimal_fros
labels = ['Passive', 'Initial', 'Design1', 'Design2']
bars = ax.bar(labels, values_fro, color=['#3b0f70', '#8c2981', '#de4968', '#ffa600'])
ax.set_ylabel('Error of A')
ax.set_ylim(min(values_fro) * 0.8, max(values_fro) * 1.1)
for bar, val in zip(bars, values_fro):
    yval = val
    ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2e}', ha='center', va='bottom')

ax.set_yscale('log')

plt.tight_layout()
svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_comparison_bar_plot.svg")
plt.savefig(svg_filename)
print(f"Figure saved as {svg_filename}")
plt.show()

# %%
