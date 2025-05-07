#%%
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
from plot_heatmap import plot_heatmap
from set_default_plot_settings import set_default_plot_settings

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import numpy as np
import scipy
import os
from sklearn.decomposition import PCA
import networkx as nx

results_dir = "results"
fs = 1000
dt = 1/fs

#%%
mode_freq = [10,40]
dim = len(mode_freq)*2

diagB = np.ones(dim)
B_true = np.diag(diagB)

desired_eigenvalues = []
for fIndex, f in enumerate(mode_freq):
    desired_eigenvalues.append([-fIndex*50-(2*np.pi*f)*1j, -fIndex*50+(2*np.pi*f)*1j])

print("Desired eigenvalues:", desired_eigenvalues)

A_continuous = np.zeros([dim, dim])
# Construct the continuous-time matrix A
for index, desired_eigenvalue in enumerate(desired_eigenvalues):
    A_continuous[2*index:2*(index+1), 2*index:2*(index+1)] = np.array([[desired_eigenvalue[0].real, desired_eigenvalue[0].imag],[desired_eigenvalue[1].imag, desired_eigenvalue[1].real]])# Calculate the eigenvalues of the continuous-time matrix
eigenvalues, _ = np.linalg.eig(A_continuous)

A_discrete = scipy.linalg.expm(A_continuous * dt)

A_discrete[0,2] = -0.01
A_discrete[1,3] = 0.01
A_discrete[2,0] = 0.01
A_discrete[3,0] = -0.01
A_discrete[2,1] = 0.01
A_discrete[3,1] = -0.02    

eigenvalues, eigenvectors = np.linalg.eig(A_continuous)

print("Eigenvalues of the continuous-time matrix A:")
print("real:", eigenvalues.real)
print("imag:", eigenvalues.imag/(2*np.pi))

# Calculate and print the eigenvalues of the discrete-time matrix A
eigenvalues_discrete, eigenvectors_discrete = np.linalg.eig(A_discrete)
print("Eigenvalues of the discrete-time matrix A:", eigenvalues_discrete)
# Display the eigenvalues of the discrete-time matrix A in exponential form
magnitudes = np.abs(eigenvalues_discrete)
angles = np.angle(eigenvalues_discrete)

print("Magnitudes of the eigenvalues of the discrete-time matrix A:", magnitudes)
print("Angles of the eigenvalues of the discrete-time matrix A (in radians):", angles)

# Plot heatmap of A_discrete
set_default_plot_settings(font_size=24, dpi=200)
plt.figure(figsize=(5, 5))
plt.imshow(A_discrete, cmap='viridis', interpolation='nearest', vmin=-0.4, vmax=0.4)

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
        if value==0:
            plt.text(j, i, f"0.00", ha='center', va='center', color='black', fontsize=20)
        elif abs(value) < 1e-2:
            plt.text(j, i, f"{value:.0e}", ha='center', va='center', color='black', fontsize=20)
        else:
            plt.text(j, i, f"{value:.2f}", ha='center', va='center', color='black', fontsize=20)

plt.tight_layout()

# Save the heatmap as an SVG file
heatmap_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_A_discrete_heatmap_dim{dim}.svg")
plt.savefig(heatmap_filename)
print(f"Heatmap saved as {heatmap_filename}")

# Create a directed graph from A_discrete
G = nx.DiGraph(A_discrete)

# Relabel nodes to start from 1
mapping = {i: i + 1 for i in range(A_discrete.shape[0])}
G = nx.relabel_nodes(G, mapping)

# Plot the directed graph
plt.figure(figsize=(5, 4.8))
pos = nx.spring_layout(G)  # positions for all nodes

# Draw the nodes and edges
nx.draw_networkx_nodes(G, pos, node_size=600, node_color='skyblue')
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='-|>', arrowsize=20)
nx.draw_networkx_labels(G, pos, font_size=20)

# Add a black border around the plot with thicker lines
plt.gca().spines['top'].set_color('black')
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_color('black')
plt.gca().spines['right'].set_linewidth(2)

# Save the directed graph as an SVG file with minimal padding
graph_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_directed_graph.svg")
plt.savefig(graph_filename, format='svg', bbox_inches='tight', pad_inches=0.05)
print(f"Directed graph saved as {graph_filename}")
plt.show()

#%%
A_true = A_discrete
Trials = 100
T = 10
sigma = 1e-32

for input_type in ["passive", "impulse", "cos"]:
    xi_trials = np.zeros((Trials, dim, fs*T))
    for trial in range(Trials):
        for i in range(1, fs*T):
            xi_trials[trial, :, i-1] = np.random.normal(0, sigma, dim)
            
    x0 = np.ones([dim,1])
    x = np.zeros([dim, fs*T])
    xi = np.zeros([dim, fs*T])
    x[:,0] = x0[:,0]

    u0 = np.array([0.3,0.3,0.6,0.6])
    if input_type=="passive": #Passive
        w = np.zeros(fs*T)
        u = np.outer(u0, w)
    if input_type=="impulse": #Impulse
        w = np.zeros(fs*T)
        w[fs * T - fs//(mode_freq[0])//2] = 1
        u = np.outer(u0, w)
    elif input_type=="step": #Step
        w = np.ones(fs*T)*1
        u = np.outer(u0, w)
    elif input_type=="cos": #Cosine
        omega = 2*np.pi*60
        u = np.zeros([dim, fs*T])
        w = np.cos(omega * np.arange(0, T, dt))*0.7e-1
        u = np.outer(u0, w)        
            
    for i in range(1, fs*T):
        xi[:,i-1] = np.random.normal(0, sigma, dim)
        if i >= fs * T - fs//(mode_freq[0])//2:
            x[:,i] = A_true @ x[:,i-1] + B_true @ u[:,i-1] + xi[:,i-1]
        else:
            x[:,i] = A_true @ x[:,i-1] + xi[:,i-1]
        
    X_ = x[:, -fs//(mode_freq[0]):]
    U_ = u[:, -fs//(mode_freq[0]):]
    
    X = X_[:,:-1]
    U = U_[:,:-1]
    Y = X_[:,1:]
            
    Z = np.vstack([X, U])

    Phi_est = Y@scipy.linalg.pinv(Z)
    A_est = Phi_est[:,:dim]
    B_est = Phi_est[:,dim:]
    
    # Plot time series of X
    set_default_plot_settings(font_size=24, dpi=200)
    fig, axs = plt.subplots(4, 1, figsize=(5, 6), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 1, X.shape[1]))

    lim = 3
    for d in range(dim):
        for t in range(X.shape[1] - 1):
            axs[d % 4].plot(np.arange(t, t + 2) * dt, X[d, t:t + 2], color=colors[t], linewidth=2)
        axs[d % 4].set_ylabel(f'$x_{d+1}$')
        axs[d % 4].grid(True)
        axs[d % 4].set_yticks([])
        axs[d % 4].set_xticks([])
        axs[d % 4].set_ylim(-lim, lim)
        
    axs[-1].set_xlabel('Time')
    plt.tight_layout(pad=0.4, rect=[0, 0, 1, 0.96])

    # Save the time series plot as an SVG file
    time_series_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_time_series_{input_type}.svg")
    plt.savefig(time_series_filename, bbox_inches='tight', pad_inches=0.05)
    print(f"Time series plot saved as {time_series_filename}")
    plt.show()

    set_default_plot_settings(font_size=18, dpi=200)
    # Perform PCA on X
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X.T)

    # Plot the PCA results in 3D
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=np.arange(X_pca.shape[0]), cmap='viridis', marker='o', s=5)
    cbar = plt.colorbar(sc, shrink=0.5, pad=0.2, label='Time Step')  # Adjust the shrink parameter to shorten the colorbar
    cbar.set_ticks([])  # Remove ticks from the colorbar
    ax.set_xlabel('PC 1', labelpad=-12)
    ax.set_ylabel('PC 2', labelpad=-12)
    ax.set_zlabel('PC 3', labelpad=-12)

    lim = 0.9
    
    # Rotate axis labels
    ax.xaxis.label.set_rotation(-18)
    ax.yaxis.label.set_rotation(54)
    ax.zaxis.label.set_rotation(90)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(True)
    plt.tight_layout(pad=0.1, rect=[0, 0, 1, 1])

    # Save the PCA plot as an SVG file
    pca_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_PCA_X_{input_type}_{X.shape[1]}.svg")
    plt.savefig(pca_filename, bbox_inches='tight', pad_inches=0.05)
    print(f"PCA plot saved as {pca_filename}")
    plt.show()
         
    set_default_plot_settings(font_size=24, dpi=200)
    plt.figure(figsize=(5, 5))
    plt.imshow(A_est, cmap='viridis', interpolation='nearest', vmin=-0.4, vmax=0.4)

    # Set ticks at the middle of each element
    plt.xticks(ticks=np.arange(A_est.shape[1]), labels=range(1, A_est.shape[1] + 1))
    plt.yticks(ticks=np.arange(A_est.shape[0]), labels=range(1, A_est.shape[0] + 1))

    # Set labels
    plt.xlabel('Dimension')
    plt.ylabel('Dimension')

    # Display the values on the heatmap
    for i in range(A_est.shape[0]):
        for j in range(A_est.shape[1]):
            value = A_est[i, j]
            if value == 0:
                plt.text(j, i, f"0.00", ha='center', va='center', color='black', fontsize=18)
            elif abs(value) < 1e-2:
                plt.text(j, i, f"0.00", ha='center', va='center', color='black', fontsize=18)
            else:
                plt.text(j, i, f"{value:.2f}", ha='center', va='center', color='black', fontsize=18)

    plt.tight_layout(pad=0.05)

    # Save the heatmap as an SVG file
    heatmap_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_A_est_heatmap_{input_type}_dim{dim}.svg")
    plt.savefig(heatmap_filename, bbox_inches='tight', pad_inches=0.05)
    print(f"Heatmap saved as {heatmap_filename}")

# %%
