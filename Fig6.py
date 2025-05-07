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
import scipy
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import networkx as nx

fs=1000
dt = 1/fs

#%%
base_fs = [15, 25, 35]
dim = len(base_fs)*2+2

# diagB = np.arange(dim+1, 1, -1)
diagB = np.ones(dim)
B_true = np.diag(diagB)

desired_eigenvalues = []
for fIndex, f in enumerate(base_fs):
    desired_eigenvalues.append([-(fIndex*5+1)-(2*np.pi*f)*1j, -(fIndex*5+1)+(2*np.pi*f)*1j])

print("Desired eigenvalues:", desired_eigenvalues)

A_continuous = np.zeros([dim, dim])
# Construct the continuous-time matrix A
for index, desired_eigenvalue in enumerate(desired_eigenvalues):
    A_continuous[2*index:2*(index+1), 2*index:2*(index+1)] = np.array([[desired_eigenvalue[0].real, desired_eigenvalue[0].imag],[desired_eigenvalue[1].imag, desired_eigenvalue[1].real]])# Calculate the eigenvalues of the continuous-time matrix
        
eigenvalues, _ = np.linalg.eig(A_continuous)

# Discretize the continuous-time matrix A
A_discrete = scipy.linalg.expm(A_continuous * dt)

for i in range(dim):
    A_discrete[i,dim-1] = np.random.rand()*0.8
A_discrete[0,dim-2] = np.random.rand()*0.4+0.4
A_discrete[2,dim-2] = np.random.rand()*0.4+0.4
A_discrete[dim-2,dim-2] = 0
A_discrete[dim-1,dim-1] = 0
    

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
A_true = A_discrete
Trials = 200
T = 1
input_type = "impulse"
sigma = 0.01

xi_trials = np.zeros((Trials, dim, fs*T))
for trial in range(Trials):
    for i in range(1, fs*T):
        xi_trials[trial, :, i-1] = np.random.normal(0, sigma, dim)
                
x0 = np.zeros([dim,1])
x = np.zeros([dim, fs*T])
xi = np.zeros([dim, fs*T])
x[:,0] = x0[:,0]

for i in range(1, fs*T):
    xi[:,i-1] = xi_trials[trial, :, i-1]
    x[:,i] = A_true @ x[:,i-1] + xi[:,i-1]
    
X = x[:,:-1]
# Calculate the covariance matrix of X
cov_X = np.cov(X)

# Calculate the eigenvalues and eigenvectors of the covariance matrix
eigenvalues_cov_X, eigenvectors_cov_X = np.linalg.eig(cov_X)

#%%
        
# Print the optimization history
def compute_metrics(vector, Trials, dim, fs, T, dt, A_true, B_true, input_type, test_on=False):
    fros_trial_A = []
    fros_trial_B = []
    fros_trial_Phi = []
    fros_W = []
    trace_inv_X_trial = []
    trace_inv_U_trial = []
    A_optimal_trial = []

    Xs = []
    Us = []
    Ys = []

    for trial in range(Trials):
        x0 = np.zeros([dim,1])
        x = np.zeros([dim, fs*T])
        xi = np.zeros([dim, fs*T])
        x[:,0] = x0[:,0]
        
        u0 = vector
        if input_type=="impulse": #Impulse
            w = np.zeros(fs*T)
            w[0] = np.random.normal(0, 10)
            u = np.outer(u0, w)
        elif input_type=="step": #Step
            w = np.ones(fs*T)*1
            u = np.outer(u0, w)
        elif input_type=="cos": #Cosine
            omega = 2*np.pi*0.1
            u = np.zeros([dim, fs*T])
            w = 1 * np.cos(omega * np.arange(0, T, dt))
            u = np.outer(u0, w)

        for i in range(1, fs*T):
            xi[:,i-1] = xi_trials[trial, :, i-1]
            x[:,i] = A_true @ x[:,i-1] + B_true @ u[:,i-1] + xi[:,i-1]
            
        X = x[:,:-1]
        U = u[:,:-1]
        Y = x[:,1:]
        
        Xs.append(X)
        Us.append(U)
        Ys.append(Y)
        
    X = np.hstack(Xs)
    U = np.hstack(Us)
    Y = np.hstack(Ys)
        
    if test_on:
        # Plot the state variables X in subplots
        fig, axs = plt.subplots(dim, 1, figsize=(10, 2*dim))
        plt.title(np.argmax(u0))
        for i in range(dim):
            axs[i].plot(X[i, :], label=f'State Variable {i+1}')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel(f'X[{i}]')
            axs[i].legend()
            axs[i].grid(True)

        plt.tight_layout()
        plt.show()
                    
    Z = np.vstack([X, U])
                        
    Phi_est = Y@scipy.linalg.pinv(Z)
    
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

    A_optimal_trial.append(np.trace(np.linalg.pinv(Z@Z.T)))
    trace_inv_X_trial.append(np.trace(np.linalg.pinv(X@X.T)))
    trace_inv_U_trial.append(np.trace(np.linalg.pinv(U@U.T)))
    
        
    return np.mean(fros_trial_A), np.mean(fros_trial_B), np.mean(fros_trial_Phi), np.mean(A_optimal_trial), np.mean(trace_inv_X_trial), np.mean(trace_inv_U_trial), np.mean(fros_W)

fros = []
A_optimal = []
# Create a list of vectors rotated by 360 degrees in 2D
input_vectors = []
perturb_node = []
# Generate all possible patterns of one-hot vectors
for i in range(dim):
    vector = np.zeros(dim)
    vector[i] = 1
    input_vectors.append(vector)
    perturb_node.append(i)
    
#並列
if True:
    num_cores = multiprocessing.cpu_count() - 3
    results = Parallel(n_jobs=num_cores)(delayed(compute_metrics)(vector, Trials, dim, fs, T, dt, A_true, B_true, input_type) for vector in tqdm(input_vectors))

fros_A = []
fros_B = []
fros_Phi = []
A_optimal = []
trace_inv_X = []
trace_inv_U = []
fros_W = []
for fros_mean_A, fros_mean_B, fros_mean_Phi, A_optimal_mean, trace_inv_X_mean, trace_inv_U_mean, fros_mean_W in results:
    fros_A.append(fros_mean_A)
    fros_B.append(fros_mean_B)
    fros_Phi.append(fros_mean_Phi)    
    A_optimal.append(A_optimal_mean)
    trace_inv_X.append(trace_inv_X_mean)
    trace_inv_U.append(trace_inv_U_mean)
    fros_W.append(fros_mean_W)

#%%
# Find real eigenvectors from eigenvectors_discrete
real_eigenvectors_discrete = [vec.real for vec in eigenvectors_discrete.T if np.all(np.isreal(vec))]

# Find the closest input vector index for each real eigenvector
closest_indices = []
for real_vec in real_eigenvectors_discrete:
    dot_products = np.dot(input_vectors, real_vec)
    closest_index = np.argmax(abs(dot_products))
    closest_indices.append(closest_index)

print("Indices of the closest input vectors:", closest_indices)

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
    'input_vectors': input_vectors,
    'input_type': input_type,
    'A_continuous': A_continuous,
    'A_discrete': A_discrete,
    'base_fs': base_fs
}

# Ensure the results directory exists
results_dir = "results"
import os

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

filename = os.path.join(results_dir+"/", f"{__file__.split('/')[-1].split('.')[0]}_{input_type}_dim{dim}.pkl")
with open(filename, 'wb') as f:
    pickle.dump(data_to_save, f)

#%%
import os
results_dir = "results"
filename = os.path.join(results_dir+"/", f"{__file__.split('/')[-1].split('.')[0]}_{input_type}_dim{dim}.pkl")
with open(filename, 'rb') as f:
    loaded_data = pickle.load(f)

fros_A = loaded_data['fros_A']
fros_B = loaded_data['fros_B']
fros_Phi = loaded_data['fros_Phi']

fros_W = loaded_data['fros_W']
A_optimal = loaded_data['A_optimal']
trace_inv_X = loaded_data['trace_inv_X']
trace_inv_U = loaded_data['trace_inv_U']
input_vectors = loaded_data['input_vectors']
input_type = loaded_data['input_type']
A_continuous = loaded_data['A_continuous']
A_discrete = loaded_data['A_discrete']
base_fs = loaded_data['base_fs']

#%%
import copy
G = copy.deepcopy(A_discrete)
np.fill_diagonal(G, 0)
weighted_degree_centrality = np.sum(np.abs(G), axis=0)

# Calculate the eigenvectors of A_discrete
eigenvalues_discrete, eigenvectors_discrete = np.linalg.eig(A_discrete)

set_default_plot_settings(font_size=20, dpi=200, line_width=3)

# Plot the first eigenvector as a bar graph
plt.figure(figsize=(5, 2.5))
colors = plt.cm.viridis(np.linspace(0, 1, dim))
for i in range(0, dim-2, 2):
    # Extract the i-th eigenvector
    eigenvector = eigenvectors_discrete[:, i]
    plt.bar(range(1, dim + 1), eigenvector.real, color=colors[i], label=f'{i+1}-{i+2}')

plt.hlines(0, 0, dim, color='black', linewidth=0.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Node Number')
plt.ylabel('Value')
plt.xticks(ticks=range(1, dim + 1))
plt.tight_layout()
svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_eigenvector_dim{dim}.svg")
plt.savefig(svg_filename)
plt.show()

#%%
# Plot the first eigenvector as a bar graph
set_default_plot_settings(font_size=20, dpi=200, line_width=3)

plt.figure(figsize=(5, 2.5))
colors = plt.cm.viridis(np.linspace(0, 1, dim))
for i in [6,7]:
    # Extract the i-th eigenvector
    eigenvector = eigenvectors_discrete[:, i]
    plt.bar(range(1, dim + 1), eigenvector.real, color=colors[i], label=f'{i+1}')
    plt.hlines(0, 0, dim, color='black', linewidth=0.5)
    
    plt.xlabel('Node Number')
    plt.ylabel('Value')
    plt.xticks(ticks=range(1, dim + 1))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout(pad=0.5)
svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_eigenvector_state_dim{dim}.svg")
plt.savefig(svg_filename)
plt.show()

#%%
set_default_plot_settings(font_size=20, dpi=200, line_width=3)

# Plot the absolute values of eigenvalues_discrete
plt.figure(figsize=(5, 2.5))
plt.bar(range(1, len(eigenvalues_discrete) + 1), np.abs(eigenvalues_discrete), color='blue')
plt.xlabel('Node Number')
plt.ylabel('$\|\lambda_A\|$')
plt.yscale('log')
plt.xticks(ticks=range(1, len(eigenvalues_discrete) + 1))
plt.yticks(ticks=[9.9e-1, 9.92e-1, 9.94e-1, 9.96e-1, 9.98e-1,], labels=['9.90', '9.92', '9.94', '9.96', '9.98'])
plt.tight_layout()
svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_absolute_eigenvalues_dim{dim}.svg")
plt.savefig(svg_filename)
plt.show()

#%%
set_default_plot_settings(font_size=20, dpi=200, line_width=3)

# Plot Weighted Degree Centrality
plt.figure(figsize=(5, 2.5))
plt.bar(range(1, dim + 1), weighted_degree_centrality, color='blue')
plt.xlabel('Input Node')
plt.ylabel('$W^{\\text{out}}_i$')
plt.xticks(ticks=range(1, dim + 1))
plt.tight_layout()
svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_centrality_dim{dim}.svg")
plt.savefig(svg_filename)
plt.show()

# Plot Weighted Degree Centrality
plt.figure(figsize=(5, 5))
plt.bar(range(1, dim + 1), A_optimal, color='blue')
plt.xlabel('Input Node')
plt.ylabel(r'tr($\Sigma^{-1}_\mathbf{Z}$)')
plt.xticks(ticks=range(1, dim + 1))
plt.tight_layout()
svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_ZZ_dim{dim}.svg")
plt.savefig(svg_filename)
plt.show()

# Plot Frobenius Norm of A
plt.figure(figsize=(5, 5))
plt.bar(range(1, dim + 1), fros_A, color='blue')
plt.xlabel('Input Node')
plt.ylabel('Error of A')
plt.yscale('log')
plt.xticks(ticks=range(1, dim + 1))

# plt.yticks(ticks=[2e-5, 3e-5, 4e-5, 5e-5, 6e-5], labels=['2.0', '3.0', '4.0', '5.0', '6.0'])
plt.tight_layout()
svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_A_dim{dim}.svg")
plt.savefig(svg_filename)
plt.show()
#%%

# Plot Frobenius Norm of B
plt.figure(figsize=(5, 5))
plt.bar(range(1, dim + 1), fros_B, color='blue')
plt.xlabel('Node Number')
plt.ylabel('Error of B')
plt.xticks(ticks=range(1, dim + 1))
plt.tight_layout()
svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_B_dim{dim}.svg")
plt.savefig(svg_filename)
plt.show()

# Plot Frobenius Norm of W
plt.figure(figsize=(5, 5))
plt.bar(range(1, dim + 1), fros_W, color='blue')
plt.xlabel('Node Number')
plt.ylabel('Error of W')
plt.xticks(ticks=range(1, dim + 1))
plt.tight_layout()
svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_W_dim{dim}.svg")
plt.savefig(svg_filename)
plt.show()

#行列Aをプロット
# Plot heatmap of A_discrete
set_default_plot_settings(font_size=22, dpi=200)
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
        plt.text(j, i, f"{A_discrete[i, j]:.2f}", ha='center', va='center', color='black', fontsize=14)

plt.tight_layout()

# Save the heatmap as an SVG file
heatmap_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_A_discrete_heatmap_dim{dim}.svg")
plt.savefig(heatmap_filename)
print(f"Heatmap saved as {heatmap_filename}")

#行列Bをプロット
# Plot heatmap of B_true
set_default_plot_settings(font_size=22, dpi=200)
plt.figure(figsize=(5, 5))
plt.imshow(B_true, cmap='viridis', interpolation='nearest', vmin=-0.8)

# Set ticks at the middle of each element
plt.xticks(ticks=np.arange(B_true.shape[1]), labels=range(1, B_true.shape[1] + 1))
plt.yticks(ticks=np.arange(B_true.shape[0]), labels=range(1, B_true.shape[0] + 1))

# Set labels
plt.xlabel('Dimension')
plt.ylabel('Dimension')

# Display the values on the heatmap
for i in range(B_true.shape[0]):
    for j in range(B_true.shape[1]):
        plt.text(j, i, f"{B_true[i, j]:.2f}", ha='center', va='center', color='black', fontsize=14)

plt.tight_layout()

# Save the heatmap as an SVG file
heatmap_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_B_discrete_heatmap_dim{dim}.svg")
plt.savefig(heatmap_filename)
print(f"Heatmap saved as {heatmap_filename}")

# Create a directed graph from the discrete-time matrix A
G = nx.DiGraph(A_discrete.T)

# Relabel nodes to start from 1
mapping = {i: i + 1 for i in range(A_discrete.shape[0])}
G = nx.relabel_nodes(G, mapping)

# Plot the graph
set_default_plot_settings(font_size=12)
plt.figure(figsize=(4, 4))
pos = nx.spring_layout(G, seed=42)  # positions for all nodes

# Adjust positions to increase the gap between nodes 7 and 8
pos[7] += np.array([0.0, 0.2])
pos[8] -= np.array([0.0, 0.2])

nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='black', node_size=700, font_size=12, font_weight='bold', arrowsize=20)
svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_{input_type}_network_dim{dim}.svg")
plt.savefig(svg_filename)
print(f"Figure saved as {svg_filename}")
plt.show()

# %%
