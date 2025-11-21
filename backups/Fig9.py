#%%
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
from plot_heatmap import plot_heatmap
from set_default_plot_settings import set_default_plot_settings
from calc_gramian import calc_gramian
from rand_matrix_c_specified_number_of_large_eigenvalues import rand_matrix_c_specified_number_of_large_eigenvalues

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import pickle
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import os
import networkx as nx

results_dir = "results"
fs=100
dt = 1/fs
input_type = "impulse"
inputTrials = 5

#%% 
base_fs = []
dim = 5
B_true = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0]
])

A_discrete, A_continuous = rand_matrix_c_specified_number_of_large_eigenvalues(dim, 0, dt)

A_discrete = np.array([
    [0.2, 0.6, 0.0, 0.0, 0.0],
    [0.6, 0.2, 0.0, 0.0, 0.0],
    [-0.2, 0.0, 0.2, 0.6, 0.3],
    [0.0, -0.2, 0.6, 0.2, 0.6],
    [0.0, 0.0, 0.0, 0.0, 0.2]
])

A_discrete = np.array([
    [0.4, 0.6, 0.0, 0.0, 0.0],
    [0.6, 0.4, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.4, 0.6, 0.3],
    [0.0, 0.0, 0.6, 0.4, 0.6],
    [0.0, 0.0, 0.0, 0.0, 0.4]
])
    

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
        plt.text(j, i, f"{A_discrete[i, j]:.2f}", ha='center', va='center', color='black',fontsize=18)

plt.tight_layout()

# Save the heatmap as an SVG file
heatmap_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_A_discrete_heatmap_dim{dim}.svg")
plt.savefig(heatmap_filename)
print(f"Heatmap saved as {heatmap_filename}")

#%%
# Create a directed graph from the discrete-time matrix A
G = nx.DiGraph(A_discrete.T)
# Set the diagonal elements of G to 0
for node in G.nodes():
    if G.has_edge(node, node):
        G.remove_edge(node, node)
# Relabel nodes to start from 1
mapping = {i: i + 1 for i in range(A_discrete.shape[0])}
G = nx.relabel_nodes(G, mapping)

# Plot the graph
set_default_plot_settings(font_size=12)
plt.figure(figsize=(3, 3))
pos = nx.spring_layout(G, seed=41, k=4)  # positions for all nodes, k controls the distance between nodes

# Draw nodes and edges
nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='black', node_size=500, font_size=12, font_weight='bold', arrows=True, width=2, arrowsize=20)

svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_network_dim{dim}.svg")
plt.savefig(svg_filename)
print(f"Figure saved as {svg_filename}")
plt.show()

#%%

A_true = A_discrete
Trials = 100
T = 1
sigma = 0.01

np.random.seed(45)

xi_trials = np.zeros((Trials, dim, fs*T))
for trial in range(Trials):
    for i in range(1, fs*T):
        xi_trials[trial, :, i-1] = np.random.normal(0, sigma, dim)
        
def compute_metrics(value, Trials, dim, fs, T, dt, A_true, B_true, input_type, test_on = False):
    fros_trial_A = []
    fros_trial_B = []
    fros_trial_Phi = []
    fros_W = []
    error_cost_trial = []
    trace_inv_X_trial = []
    trace_inv_U_trial = []
    A_optimal_trial = []
    
    x0 = np.array([0, 0, 0, 0, 0]).reshape(-1,1)
    for trial in range(Trials):
        Xs = []
        Ys = []
        Zs = []
        Us = []
        
        for trial in range(inputTrials):
            x = np.zeros([dim, fs*T])
            xi = np.zeros([dim, fs*T])
            x[:,0] = x0[:,0]
            
            u0 = np.zeros(dim)
            u0[np.random.randint(0, dim)] = 1
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
                    
            # u += u_xi_trials[trial, :, :]

            # オイラー法の実装
            for i in range(1, fs*T):
                xi[:,i-1] = np.random.normal(0, sigma, dim)
                x[:,i] = A_true @ x[:,i-1] + B_true @ u[:,i-1] + xi[:,i-1]
                
            X = x[:,:-1]
            U = u[:,:-1]
            Y = x[:,1:]
                    
            Z = np.vstack([X, U])
            
            Xs.append(X)
            Ys.append(Y)
            Zs.append(Z)
            Us.append(U)
        
        X = np.hstack(Xs)
        Y = np.hstack(Ys)
        Z = np.hstack(Zs)
        U = np.hstack(Us)
                
        # 
        Phi_est = Y@scipy.linalg.pinv(Z)
        
        A_est = Phi_est[:,:dim]
        B_est = Phi_est[:,dim:]
        
        # Plot heatmap of A_true
        set_default_plot_settings(font_size=18, dpi=200)
        plt.figure(figsize=(3, 3))
        plt.imshow(A_true, cmap='viridis', interpolation='nearest', vmin=-0.6, vmax=0.6)
        plt.xticks(ticks=np.arange(A_true.shape[1]), labels=range(1, A_true.shape[1] + 1))
        plt.yticks(ticks=np.arange(A_true.shape[0]), labels=range(1, A_true.shape[0] + 1))
        for i in range(A_true.shape[0]):
            for j in range(A_true.shape[1]):
                plt.text(j, i, f"{A_true[i, j]:.2f}", ha='center', va='center', color='black', fontsize=10)
        plt.xlabel('Dimension')
        plt.ylabel('Dimension')
        plt.title('$A_{true}$', fontname='cmb10')
        plt.tight_layout()
        svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_A_true_heatmap_{value}.svg")
        plt.savefig(svg_filename)
        print(f"Heatmap saved as {svg_filename}")
        plt.show()

        # Plot heatmap of A_est
        plt.figure(figsize=(3, 3))
        plt.imshow(A_est, cmap='viridis', interpolation='nearest', vmin=-0.6, vmax=0.6)
        plt.xticks(ticks=np.arange(A_est.shape[1]), labels=range(1, A_est.shape[1] + 1))
        plt.yticks(ticks=np.arange(A_est.shape[0]), labels=range(1, A_est.shape[0] + 1))
        for i in range(A_est.shape[0]):
            for j in range(A_est.shape[1]):
                plt.text(j, i, f"{A_est[i, j]:.2f}", ha='center', va='center', color='black', fontsize=10)
        plt.xlabel('Dimension')
        plt.ylabel('Dimension')
        plt.title('$A_{est}$', fontname='cmb10')
        plt.tight_layout()
        svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_A_est_heatmap_{value}.svg")
        plt.savefig(svg_filename)
        print(f"Heatmap saved as {svg_filename}")
        plt.show()

        # Plot heatmap of B_true
        plt.figure(figsize=(3, 3))
        plt.imshow(B_true, cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
        plt.xticks(ticks=np.arange(B_true.shape[1]), labels=range(1, B_true.shape[1] + 1))
        plt.yticks(ticks=np.arange(B_true.shape[0]), labels=range(1, B_true.shape[0] + 1))
        for i in range(B_true.shape[0]):
            for j in range(B_true.shape[1]):
                plt.text(j, i, f"{B_true[i, j]:.2f}", ha='center', va='center', color='black', fontsize=10)
        plt.xlabel('Dimension')
        plt.ylabel('Dimension')
        plt.title('$B_{true}$', fontname='cmb10')
        plt.tight_layout()
        svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_B_true_heatmap_{value}.svg")
        plt.savefig(svg_filename)
        print(f"Heatmap saved as {svg_filename}")
        plt.show()

        # Plot heatmap of B_est
        plt.figure(figsize=(3, 3))
        plt.imshow(B_est, cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
        plt.xticks(ticks=np.arange(B_est.shape[1]), labels=range(1, B_est.shape[1] + 1))
        plt.yticks(ticks=np.arange(B_est.shape[0]), labels=range(1, B_est.shape[0] + 1))
        for i in range(B_est.shape[0]):
            for j in range(B_est.shape[1]):
                plt.text(j, i, f"{B_est[i, j]:.2f}", ha='center', va='center', color='black', fontsize=10)
        plt.xlabel('Dimension')
        plt.ylabel('Dimension')
        plt.title('$B_{est}$', fontname='cmb10')
        plt.tight_layout()
        svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_B_est_heatmap_{value}.svg")
        plt.savefig(svg_filename)
        print(f"Heatmap saved as {svg_filename}")
        plt.show()
                        
        L=int(fs*T*1.0)
        
        W_true = calc_gramian(A_true,B_true,L)
        W_est = calc_gramian(A_est,B_est,L)
        
        cost_true = np.trace(W_true)
        cost_est = np.trace(W_est)
        
        error_cost_trial.append(abs(cost_true - cost_est))
        
        x_T = np.array([0, 0, 50, 50, 0]).reshape(-1,1)
        u_star = np.zeros([dim,0])

        W_inv = scipy.linalg.inv(W_est)
        # 数式の計算
        for i in range(0,L):
            u_star = np.c_[u_star, B_est.T @ np.linalg.matrix_power(A_est.T,L-i-1)@W_inv@(x_T-np.linalg.matrix_power(A_est,L)@x0)]
                             
        # 最適制御
        x_c = np.zeros([dim, L+1])
        x_c[:,0] = x0[:,0]
        for i in range(0, L):
            xi[:,i-1] = np.random.normal(0, sigma, dim)
            # xi[:,i] = np.zeros(dim)
            x_c[:,i+1] = A_true @ x_c[:,i] + B_true @ u_star[:,i] + xi[:,i]
                      

        # Plot time series of x with dimension information
        set_default_plot_settings(font_size=14, dpi=200)
        fig, axs = plt.subplots(5, 1, figsize=(3.5, 2.5), sharex=True)
        for d in range(dim):
            axs[d].plot(np.arange(0, L/fs+dt, dt), x_c[d, :], label=f'$x_{d+1}$', linewidth=2, color='blue')
            axs[d].grid(True)
            axs[d].set_ylim(-5, 70)
            axs[d].set_xlim(0, L/fs+0.1)
            axs[d].set_ylabel(f'$x_{d+1}$', fontname='cmb10')
        for d in range(dim):
            axs[d].scatter(np.arange(0, L/fs+dt, dt)[-1], x_T[d], color='white', edgecolor='black', zorder=1, s=20)  # Mark the final time point
                
        axs[dim-1].set_xlabel('Time (s)')
        
        plt.suptitle("Controlled State Process", y=1.01)
        plt.tight_layout()
        # Add legends to the right of the plots
        # handles, labels = [], []
        # for ax in axs:
        #     h, l = ax.get_legend_handles_labels()
        #     handles.extend(h)
        #     labels.extend(l)
        # fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=2)

        # Save the figure as an SVG file
        svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_optimal_control_test_combined_{value}.svg")
        fig.savefig(svg_filename, bbox_inches='tight')
        print(f"Figure saved as {svg_filename}")
        plt.show()

        fig, axs = plt.subplots(1, 1, figsize=(2.5, 2.5))

        # Plot the squared time integral of u for each dimension
        squared_time_integral_u = np.sum(u_star**2, axis=1)

        axs.bar(range(1, dim + 1), squared_time_integral_u, color="blue", edgecolor="black", linewidth=1.0)
        axs.set_xlabel('$u_i$', fontname='cmb10')
        axs.set_ylabel(r'Integrated Value')
        axs.set_ylim(0, 1.0e2)
        axs.set_xticks(range(1, dim + 1))  # Ensure all xticks are displayed
        
        plt.tight_layout()
        plt.title("Control Input Strength", y=1.01)
        
        # Save the figure as an SVG file
        svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_optimal_control_integrated_strength_{value}.svg")
        fig.savefig(svg_filename, bbox_inches='tight')
        print(f"Figure saved as {svg_filename}")
        plt.show()
                
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
            
    return np.mean(fros_trial_A), np.mean(fros_trial_B), np.mean(fros_trial_Phi), np.mean(A_optimal_trial), np.mean(trace_inv_X_trial), np.mean(trace_inv_U_trial), np.mean(fros_W), np.mean(error_cost_trial)


# Create a list of vectors rotated by 360 degrees in 2D
input_values = np.arange(1, 50, 0.1)
# input_values = np.arange(1,50,1)
# 

# input_values = np.arange(0,10,5)
input_values = [1e-20,1e20]
Trials = 1
results = []
plt.figure(figsize=(10,4))
for value in tqdm(input_values):
    results.append(compute_metrics(value, Trials, dim, fs, T, dt, A_true, B_true, input_type, test_on=True))
plt.show()


fros_A = []
fros_B = []
fros_Phi = []
A_optimal = []
trace_inv_X = []
trace_inv_U = []
fros_W = []
error_cost = []
for fros_mean_A, fros_mean_B, fros_mean_Phi, A_optimal_mean, trace_inv_X_mean, trace_inv_U_mean, fros_mean_W, error_mean_cost in results:
    fros_A.append(fros_mean_A)
    fros_B.append(fros_mean_B)
    fros_Phi.append(fros_mean_Phi)    
    A_optimal.append(A_optimal_mean)
    trace_inv_X.append(trace_inv_X_mean)
    trace_inv_U.append(trace_inv_U_mean)
    fros_W.append(fros_mean_W)
    error_cost.append(error_mean_cost)


#%%
# Save variables to a pickle file
data_to_save = {
    'fros_A': fros_A,
    'fros_B': fros_B,
    'fros_Phi': fros_Phi,
    'fros_W': fros_W,
    'error_cost': error_cost,
    'A_optimal': A_optimal,
    'trace_inv_X': trace_inv_X,
    'trace_inv_U': trace_inv_U,
    'input_values': input_values,
    'input_type': input_type,
    'A_continuous': A_continuous,
    'A_discrete': A_discrete,
    'base_fs': base_fs
}

# Ensure the results directory exists
import os
filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_{input_type}_dim{dim}.pkl")
with open(filename, 'wb') as f:
    pickle.dump(data_to_save, f)

#%%
input_values = np.arange(1, 50, 0.1)
filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_{input_type}_dim{dim}.pkl")
with open(filename, 'rb') as f:
    loaded_data = pickle.load(f)

fros_A = loaded_data['fros_A']
fros_B = loaded_data['fros_B']
fros_Phi = loaded_data['fros_Phi']
fros_W = loaded_data['fros_W']
error_cost = loaded_data['error_cost']
A_optimal = loaded_data['A_optimal']
trace_inv_X = loaded_data['trace_inv_X']
trace_inv_U = loaded_data['trace_inv_U']
input_vectors = loaded_data['input_values']
input_type = loaded_data['input_type']
A_continuous = loaded_data['A_continuous']
A_discrete = loaded_data['A_discrete']
base_fs = loaded_data['base_fs']

#%%
set_default_plot_settings(font_size=14, dpi=200, line_width=3)
# Plot Frobenius Norm of A
plt.figure(figsize=(2.5, 2))
plt.bar(["1e-20", "1e20"], fros_A, color="blue", edgecolor="black", linewidth=1.0)
plt.xlabel('Input Intensity')
plt.ylabel(r'Error of $A$')
plt.yscale('log')
plt.tight_layout()

# Save the figure as an SVG file
svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_{input_type}_fros_A_dim{dim}.svg")
plt.savefig(svg_filename)
print(f"Figure saved as {svg_filename}")
plt.show()

# Plot Frobenius Norm of B
plt.figure(figsize=(2.5, 2))
plt.bar(["1e-20", "1e20"], fros_B, color="blue", edgecolor="black", linewidth=1.0)
plt.xlabel('Input Intensity')
plt.ylabel(r'Error of $B$')
plt.yscale('log')
plt.tight_layout()

# Save the figure as an SVG file
svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_{input_type}_fros_B_dim{dim}.svg")
plt.savefig(svg_filename)
print(f"Figure saved as {svg_filename}")
plt.show()

# Plot Frobenius Norm of W
plt.figure(figsize=(2.5, 2))
plt.bar(["1e-20", "1e20"], fros_W, color="blue", edgecolor="black", linewidth=1.0)
plt.xlabel('Input Intensity')
plt.ylabel(r'Error of $W$')
plt.yscale('log')
plt.tight_layout()

# Save the figure as an SVG file
svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_{input_type}_fros_W_dim{dim}.svg")
plt.savefig(svg_filename)
print(f"Figure saved as {svg_filename}")
plt.show()

# Plot Error of Control Cost
plt.figure(figsize=(2.5, 2))
plt.bar(["1e-20", "1e20"], error_cost, color="blue", edgecolor="black", linewidth=1.0)
plt.xlabel('Input Intensity')
plt.ylabel(r'Error of Cost')
plt.yscale('log')
plt.tight_layout()

# Save the figure as an SVG file
svg_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_{input_type}_error_cost_dim{dim}.svg")
plt.savefig(svg_filename)
print(f"Figure saved as {svg_filename}")
plt.show()

# %%
