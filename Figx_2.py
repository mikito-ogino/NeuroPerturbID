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
from sklearn.manifold import TSNE
import copy
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

set_default_plot_settings(font_size=24, fig_size=[6,6], dpi=250, 
                            line_color='blue',axes_limit=(-2,2),line_width=1)

results_dir = "results"
fs=10
dt = 1/fs
input_type = "impulse"
input_strength = 1e20
T=5
sigma = 1e-2
dim = 8
obs_dim = dim
Trials = 100
T_for_gramian = fs

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def make_matrix(eigenvalue_magnitude, connectivity_direction):
    #State1 LEFT CONNECTION 安定
    A = np.zeros((dim, dim))

    # Strong connectivity for nodes 1-4
    for i in range(4):
        for j in range(4):
            if i != j:
                if connectivity_direction == "left":
                    A[i, j] = np.random.uniform(0.8, 1)
                elif connectivity_direction == "right":
                    A[j, i] = np.random.uniform(0.7, 0.9)
                else:
                    A[i, j] = np.random.uniform(0.8, 0.9)

    # Weak connectivity for nodes 5-8
    for i in range(4, 8):
        for j in range(4, 8):
            if i != j:
                if connectivity_direction == "left":
                    A[i, j] = np.random.uniform(0.7, 0.9)
                elif connectivity_direction == "right":
                    A[j, i] = np.random.uniform(0.8, 1.0)
                else:
                    A[i, j] = np.random.uniform(0.8, 0.9)

    # Cross-group connectivity
    for i in range(4):
        for j in range(4, 8):
            A[i, j] = np.random.uniform(0.5, 0.7)
            A[j, i] = np.random.uniform(0.5, 0.7)

    for i in range(4, 8):
        for j in range(4):
            A[i, j] = np.random.uniform(0.5, 0.7)
            A[j, i] = np.random.uniform(0.5, 0.7)

    # Scale the matrix to achieve the desired eigenvalue magnitude
    eigenvalues, _ = np.linalg.eig(A)
    current_magnitude = max(abs(eigenvalues))
    scaling_factor = eigenvalue_magnitude / current_magnitude
    A *= scaling_factor
    
    return A

A1 = []
for i in range(Trials):
    A1.append(make_matrix(0.9, "left"))
      
A2 = []
for i in range(Trials):
    A2.append(make_matrix(1.0, "left"))#State4 RIGHT CONNECTION 不安定

A3 = []
for i in range(Trials):
    A3.append(make_matrix(0.9, "right"))

A4 = []
for i in range(Trials):
    A4.append(make_matrix(1, "right"))
    
A5 = []
for i in range(Trials):
    A5.append(make_matrix(0.9, "random"))

# Compute the eigenvalues for each matrix in A1 to A5
eigenvalues = {
    "A1": [np.linalg.eigvals(A) for A in A1],
    "A2": [np.linalg.eigvals(A) for A in A2],
    "A3": [np.linalg.eigvals(A) for A in A3],
    "A4": [np.linalg.eigvals(A) for A in A4],
    "A5": [np.linalg.eigvals(A) for A in A5],
}

# Prepare data for the scatter plot
colors = ['red', 'blue', 'green', 'purple', 'orange']
labels = ["A1", "A2", "A3", "A4", "A5"]

# Create the scatter plot
set_default_plot_settings(font_size=24, fig_size=[10, 6], dpi=250)
plt.figure(figsize=(8, 5))

for idx, label in enumerate(labels):
    magnitudes = [abs(eig) for eig_list in eigenvalues[label] for eig in eig_list]
    plt.scatter([idx + 1] * len(magnitudes), magnitudes, alpha=0.5, label=label, color=colors[idx])

plt.xlabel("State")
plt.ylabel("Magnitude")
plt.xticks(range(1, 6), range(1, 6))
plt.grid(alpha=0.3)
plt.tight_layout()

# Save the scatter plot
scatter_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_eigenvalue_magnitude_plot.svg")
plt.savefig(scatter_filename, pad_inches=0.05, bbox_inches='tight')
print(f"Figure saved as {scatter_filename}")
plt.show()


# Plot network graphs for A1 to A4 as subplots
fixed_pos = {}
# Arrange nodes 1-4 in a slightly randomized square
for i in range(4):
    fixed_pos[i] = (i % 2 + np.random.uniform(-0.4, 0.4), i // 2 + np.random.uniform(-0.4, 0.4))
# Arrange nodes 5-8 in another slightly randomized square, slightly offset
for i in range(4, 8):
    fixed_pos[i] = ((i - 4) % 2 + 2 + np.random.uniform(-0.4, 0.4), (i - 4) // 2 + np.random.uniform(-0.4, 0.4))

fig, axes = plt.subplots(1, 5, figsize=(20, 4))  # Create a 1x5 subplot layout

for idx, (A_group, ax) in enumerate(zip([A1, A2, A3, A4, A5], axes)):
    A = A_group[0]  # Compute the average matrix for the group
    G = nx.DiGraph(A)  # Create a directed graph from the adjacency matrix
    
    # Normalize edge weights based on the min and max values in the adjacency matrix
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    min_weight = np.min(A)
    max_weight = np.max(A)
    weights = [((abs(A[u, v]) - min_weight) / (max_weight - min_weight))**4 * 10 for u, v in edges]  # Normalize, square for emphasis, and scale for thickness
    
    nx.draw(
        G,
        fixed_pos,
        with_labels=True,
        node_size=500,
        node_color='gray',  # Node color set to black
        edge_color='gray',  # Edge color set to black
        width=weights,  # Set edge width proportional to scaled weights
        arrows=False,  # Disable arrows
        alpha=1,
        ax=ax  # Draw on the specific subplot axis
    )
    ax.set_title(f"State {idx + 1}")
    ax.axis("off")  # Turn off axis for better visualization

plt.tight_layout(pad=0.1)
graph_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_network_graphs.svg")
plt.savefig(graph_filename)
print(f"Figure saved as {graph_filename}")
plt.show()

#%%
def plot_time_series(X, results_dir, filename):
    set_default_plot_settings(fig_size=[12,4],font_size=28, dpi=200)
    for i in range(dim):
        plt.plot(X[i, :] - i * 1, label=f'State Variable {i+1}', color='blue', alpha=1)
    plt.xlabel('Time Steps')
    plt.ylabel('State Variables')
    plt.xticks([])
    plt.xlim([X.shape[1] // 2, X.shape[1]])
    plt.yticks([])
    plt.tight_layout(pad=0.1)
    svg_filename = os.path.join(results_dir, filename)
    plt.savefig(svg_filename, bbox_inches='tight', pad_inches=0.05)
    print(f"Figure saved as {svg_filename}")
    plt.show()

# Find most excitable node for each A_group
min_indices = []
for idx, A_group in enumerate([A1, A2, A3, A4, A5]):
    Js = []
    for node_index in range(dim):
        u0 = np.zeros(dim)
        u0[node_index] = 10  # Set the excitation to the current node
        A = A_group[0]  # Use the first matrix in the group
        x0 = np.ones([dim, 1])
        x = np.zeros([dim, fs * 100])
        x[:, 0] = x0[:, 0]

        # Euler method to simulate the time series
        for i in range(1, fs * 100):
            if i == fs * 100 // 2 + 3:
                x[:, i] = A @ x[:, i - 1] + u0
            else:
                x[:, i] = A @ x[:, i - 1]

        # Compute the trace of the covariance matrix of x
        cov_matrix = np.cov(x)
        Js.append(np.trace(scipy.linalg.inv(cov_matrix)))

    # Find the index of the smallest element in Js for the current A_group
    min_index = np.argmin(Js)
    min_indices.append(min_index)
    print(f"A_group {idx + 1}: Js = {Js}, min_index = {min_index}")

# Plot time series for A1 to A5 using only the first matrix in each group as subplots
fig, axes = plt.subplots(1, 5, figsize=(16, 4))  # Create a 1x5 subplot layout
colors = ['red', 'blue', 'green', 'purple', 'orange']  # Colors matching PCA

for idx, A_group in enumerate([A1, A2, A3, A4, A5]):
    u0 = np.zeros(dim)
    u0[min_indices[idx]] = 10  # Set the excitation to the current node

    A = A_group[0]  # Use the first matrix in each group
    x0 = np.ones([dim, 1])
    x = np.zeros([dim, fs * 100])
    x[:, 0] = x0[:, 0]

    # Euler method to simulate the time series
    for i in range(1, fs * 100):
        if i == fs * 100 // 2 + 3:
            x[:, i] = A @ x[:, i - 1] + u0 + np.random.normal(0, 1e-1, dim)
        else:
            x[:, i] = A @ x[:, i - 1] + np.random.normal(0, 1e-1, dim)

    # Plot the time series in the corresponding subplot
    axes[idx].set_title(f"Task {idx + 1}", color=colors[idx])
    for j in range(dim):
        axes[idx].plot(x[j, :] - j * 1, label=f'State Variable {j+1}', color=colors[idx], alpha=1)
    axes[idx].set_xlabel('Time Steps')
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])
    axes[idx].grid(alpha=0.3)

plt.tight_layout(pad=0.1)
time_series_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_perturbation_time_series_subplot.svg")
plt.savefig(time_series_filename, bbox_inches='tight', pad_inches=0.05)
print(f"Figure saved as {time_series_filename}")
plt.show()

# Plot time series for A1 to A5 using only the first matrix in each group as subplots
fig, axes = plt.subplots(1, 5, figsize=(16, 4))  # Create a 1x5 subplot layout
colors = ['red', 'blue', 'green', 'purple', 'orange']  # Colors matching PCA

for idx, A_group in enumerate([A1, A2, A3, A4, A5]):
    A = A_group[0]  # Use the first matrix in each group
    x0 = np.ones([dim, 1])
    x = np.zeros([dim, fs * 100])
    x[:, 0] = x0[:, 0]

    # Euler method to simulate the time series
    for i in range(1, fs * 100):
        x[:, i] = A @ x[:, i - 1] + np.random.normal(0, 1e-1, dim)

    # Plot the time series in the corresponding subplot
    axes[idx].set_title(f"Task {idx + 1}", color=colors[idx])
    for j in range(dim):
        axes[idx].plot(x[j, :] - j * 1, label=f'State Variable {j+1}', color=colors[idx], alpha=1)
    axes[idx].set_xlabel('Time Steps')
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])
    axes[idx].grid(alpha=0.3)

plt.tight_layout(pad=0.1)
time_series_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_ppassive_time_series_subplot.svg")
plt.savefig(time_series_filename, bbox_inches='tight', pad_inches=0.05)
print(f"Figure saved as {time_series_filename}")
plt.show()
    
#%%
#Passive
np.random.seed(42)
random.seed(42)

results_passive = {}

for idx, A_group in enumerate([A1, A2, A3, A4, A5]):
    group_label = f"A{idx + 1}"
    fros_trial = []
    A_optimal_trial = []
    A_est_trials = []
    fros_A_trial = []
    fros_W_trial = []
    fros_cost_trial = []

    for trial in range(Trials):
        A = A_group[trial]  # Use the corresponding matrix from the group
        x0 = np.ones([dim, 1])
        x = np.zeros([dim, fs * T])
        xi = np.zeros([dim, fs * T])
        x[:, 0] = x0[:, 0]

        # Euler method to simulate the time series
        for i in range(1, fs * T):
            xi[:, i - 1] = np.random.normal(0, sigma, dim)
            x[:, i] = A @ x[:, i - 1] + xi[:, i - 1]

        X_ = x[:, -fs * T // 2:]

        X = X_[:obs_dim, :-1]
        Y = X_[:obs_dim, 1:]

        # Estimate matrix A
        A_est = Y @ scipy.linalg.pinv(X)

        if trial == 0:
            plot_time_series(X, results_dir, f"{__file__.split('/')[-1].split('.')[0]}_state_variables_{group_label}.svg")

        A_est_trials.append(A_est)
        fros_A_trial.append(np.linalg.norm(A - A_est, 'fro') ** 2)
        A_optimal_trial.append(np.trace(scipy.linalg.inv(X @ X.T)))

    A_est_mean = np.mean(A_est_trials, axis=0)
    results_passive[group_label] = {
        "A_est": A_est_trials,
        "A_est_mean": A_est_mean,
        "fros_A": np.mean(fros_A_trial),
        "A_optimal": np.mean(A_optimal_trial),
        "plot_name": "passive"
    }

    print(f"{group_label} Frobenius Norm:", np.mean(fros_A_trial))
    print(f"{group_label} A_optimal:", np.mean(A_optimal_trial))


#%% Random Control
np.random.seed(42)
random.seed(42)

results_random = {}

for idx, A_group in enumerate([A1, A2, A3, A4, A5]):
    group_label = f"A{idx + 1}"
    value = 20
    hub = 1

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
        x0 = np.zeros([dim, 1])
        x = np.zeros([dim, fs * T])
        xi = np.zeros([dim, fs * T])
        x[:, 0] = x0[:, 0]

        t = np.arange(0, T, dt)
        u0 = np.zeros(dim)

        u0[min_indices[idx]] = 1

        if input_type == "impulse":  # Impulse
            w = np.zeros(fs * T)
            w[-fs * T // 2] = input_strength
        elif input_type == "step":  # Step
            w = np.ones(fs * T) * value
        elif input_type == "cos":  # Cosine
            omega = 2 * np.pi * value
            w = input_strength * np.cos(omega * np.arange(0, T, dt))
        u = np.outer(u0, w)

        xi = np.zeros([dim, fs * T])
        x[:, 0] = x0[:, 0]

        # Euler method implementation
        for i in range(1, fs * T):
            x[:, i] = A_group[trial] @ x[:, i - 1] + np.eye(dim, dim) @ u[:, i - 1] + np.random.normal(0, sigma, dim)

        X_ = x[:, -fs * T // 2:]
        U_ = u[:, -fs * T // 2:]

        X = X_[:obs_dim, :-1]
        U = U_[:obs_dim, :-1]
        Y = X_[:obs_dim, 1:]

        Z = np.vstack([X, U])

        if trial == 0:
            plot_time_series(X_, results_dir, f"{__file__.split('/')[-1].split('.')[0]}_state_variables_random_{group_label}.svg")

        Ys.append(Y)
        Zs.append(Z)

        Phi_est = Y @ scipy.linalg.pinv(Z)

        A_est = Phi_est[:, :obs_dim]
        B_est = Phi_est[:, obs_dim:]

        A_est_trials.append(A_est)
        B_est_trials.append(B_est)

        frobenius_norm_A = np.linalg.norm(A_group[trial] - A_est, 'fro') ** 2
        fros_A_trial.append(frobenius_norm_A)

        frobenius_norm_B = np.linalg.norm(np.eye(dim, dim) - B_est, 'fro') ** 2
        fros_B_trial.append(frobenius_norm_B)

        frobenius_norm_phi = np.linalg.norm(Phi_est - np.hstack([A_group[trial], np.eye(dim, dim)]), 'fro') ** 2
        fros_phi_trial.append(frobenius_norm_phi)

        A_optimal_trial.append(np.trace(scipy.linalg.pinv(Z @ Z.T)))

    results_random[f"{group_label}"] = {
        "A_est": A_est_trials,
        "plot_name": "impulse"
    }

# Perform MDS on A_est from results_passive and results_random

mds_data_passive = []
mds_labels_passive = []

mds_data_random = []
mds_labels_random = []

for group_label in ['A1', 'A2', 'A3', 'A4', 'A5']:
    for A_est in results_passive[group_label]["A_est"]:
        mds_data_passive.append(A_est.flatten())
        mds_labels_passive.append(group_label)
    for A_est in results_random[group_label]["A_est"]:
        mds_data_random.append(A_est.flatten())
        mds_labels_random.append(group_label)

mds_data_passive = np.array(mds_data_passive)
mds_data_random = np.array(mds_data_random)

# Compute pairwise distances and perform MDS
distance_matrix_passive = squareform(pdist(mds_data_passive, metric='euclidean'))
mds_passive = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_results_passive = mds_passive.fit_transform(distance_matrix_passive)

distance_matrix_random = squareform(pdist(mds_data_random, metric='euclidean'))
mds_random = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_results_random = mds_random.fit_transform(distance_matrix_random)

# Plot MDS results side by side
set_default_plot_settings(font_size=24, fig_size=[18, 6], dpi=250, 
                          line_color='blue', axes_limit=(-2, 2), line_width=1)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Passive MDS plot
for label, color in zip(['A1', 'A2', 'A3', 'A4', 'A5'], ['red', 'blue', 'green', 'purple', 'orange']):
    indices = [i for i, l in enumerate(mds_labels_passive) if l == label]
    axes[0].scatter(mds_results_passive[indices, 0], mds_results_passive[indices, 1], label=f"State {int(label[1])}", color=color, alpha=0.6)
axes[0].set_xlabel('MDS 1')
axes[0].set_ylabel('MDS 2')
axes[0].set_title('Passive')
axes[0].grid(alpha=0.3)

# Random MDS plot
for label, color in zip(['A1', 'A2', 'A3', 'A4', 'A5'], ['red', 'blue', 'green', 'purple', 'orange']):
    indices = [i for i, l in enumerate(mds_labels_random) if l == label]
    axes[1].scatter(mds_results_random[indices, 0], mds_results_random[indices, 1], label=f"State {int(label[1])}", color=color, alpha=0.6)
axes[1].set_xlabel('MDS 1')
axes[1].set_ylabel('MDS 2')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1].grid(alpha=0.3)
axes[1].set_title('Perturbation')

plt.tight_layout()

# Save the MDS plot
mds_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_A_est_mds_comparison.svg")
plt.savefig(mds_filename, pad_inches=0.05)
print(f"Figure saved as {mds_filename}")
plt.show()

# Prepare data for classification using results_passive and results_random
data_passive = []
labels_passive = []
data_random = []
labels_random = []

for group_label in ['A1', 'A2', 'A3', 'A4', 'A5']:
    for A_est in results_passive[group_label]["A_est"]:
        data_passive.append(A_est.flatten())
        labels_passive.append(group_label)
    for A_est in results_random[group_label]["A_est"]:
        data_random.append(A_est.flatten())
        labels_random.append(group_label)

data_passive = np.array(data_passive)
labels_passive = np.array(labels_passive)
data_random = np.array(data_random)
labels_random = np.array(labels_random)

# Train classifiers with 10-fold cross-validation for both passive and random
clf_passive = LinearDiscriminantAnalysis()
clf_random = LinearDiscriminantAnalysis()

predicted_labels_passive = []
true_labels_passive = []
predicted_scores_passive = []

predicted_labels_random = []
true_labels_random = []
predicted_scores_random = []

# Perform 10-fold cross-validation for passive
for train_index, test_index in KFold(n_splits=10, shuffle=True, random_state=42).split(data_passive):
    X_train, X_test = data_passive[train_index], data_passive[test_index]
    y_train, y_test = labels_passive[train_index], labels_passive[test_index]

    # Train the classifier
    clf_passive.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf_passive.predict(X_test)
    y_scores = clf_passive.predict_proba(X_test)

    # Collect predictions, scores, and true labels
    predicted_labels_passive.extend(y_pred)
    predicted_scores_passive.extend(y_scores)
    true_labels_passive.extend(y_test)

# Perform 10-fold cross-validation for random
for train_index, test_index in KFold(n_splits=10, shuffle=True, random_state=42).split(data_random):
    X_train, X_test = data_random[train_index], data_random[test_index]
    y_train, y_test = labels_random[train_index], labels_random[test_index]

    # Train the classifier
    clf_random.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf_random.predict(X_test)
    y_scores = clf_random.predict_proba(X_test)

    # Collect predictions, scores, and true labels
    predicted_labels_random.extend(y_pred)
    predicted_scores_random.extend(y_scores)
    true_labels_random.extend(y_test)

# Calculate and print the mean accuracy for both
mean_accuracy_passive = np.mean(np.array(predicted_labels_passive) == np.array(true_labels_passive))
mean_accuracy_random = np.mean(np.array(predicted_labels_random) == np.array(true_labels_random))
print(f"10-Fold Cross-Validation Accuracy (Passive): {mean_accuracy_passive:.2%}")
print(f"10-Fold Cross-Validation Accuracy (Random): {mean_accuracy_random:.2%}")

# Compute confusion matrices
cm_passive = confusion_matrix(true_labels_passive, predicted_labels_passive, labels=['A1', 'A2', 'A3', 'A4', 'A5'])
cm_random = confusion_matrix(true_labels_random, predicted_labels_random, labels=['A1', 'A2', 'A3', 'A4', 'A5'])

# Plot the confusion matrices side by side
set_default_plot_settings(font_size=24, fig_size=[12, 6], dpi=250, 
                          line_color='blue', axes_limit=(-2, 2), line_width=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Passive confusion matrix
disp_passive = ConfusionMatrixDisplay(confusion_matrix=cm_passive, display_labels=['A1', 'A2', 'A3', 'A4', 'A5'])
disp_passive.plot(cmap=plt.cm.Blues, ax=axes[0], colorbar=False)
axes[0].set_title(f"Passive (ACC = {mean_accuracy_passive:.2%})")
axes[0].set_xlabel("Predicted State")
axes[0].set_ylabel("True State")
axes[0].set_xticks(range(5))
axes[0].set_xticklabels([1, 2, 3, 4, 5])
axes[0].set_yticks(range(5))
axes[0].set_yticklabels([1, 2, 3, 4, 5])

# Random confusion matrix
disp_random = ConfusionMatrixDisplay(confusion_matrix=cm_random, display_labels=['A1', 'A2', 'A3', 'A4', 'A5'])
disp_random.plot(cmap=plt.cm.Blues, ax=axes[1], colorbar=False)
axes[1].set_title(f"Perturbation (ACC = {mean_accuracy_random:.2%})")
axes[1].set_xlabel("Predicted State")
axes[1].set_ylabel("True State")
axes[1].set_xticks(range(5))
axes[1].set_xticklabels([1, 2, 3, 4, 5])
axes[1].set_yticks(range(5))
axes[1].set_yticklabels([1, 2, 3, 4, 5])

plt.tight_layout()

# Save the confusion matrix plot
confmat_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_confusion_matrix_comparison.svg")
plt.savefig(confmat_filename, pad_inches=0.05)
print(f"Figure saved as {confmat_filename}")
plt.show()


# Compute AUC and plot ROC curves for each class
# Binarize the labels for multi-class ROC and AUC computation
classes = ['A1', 'A2', 'A3', 'A4', 'A5']
y_true_binarized_passive = label_binarize(true_labels_passive, classes=classes)
y_scores_passive = np.array(predicted_scores_passive)  # Convert scores to numpy array

y_true_binarized_random = label_binarize(true_labels_random, classes=classes)
y_scores_random = np.array(predicted_scores_random)  # Convert scores to numpy array

# Calculate AUC for each class and plot ROC curves for passive and random
def compute_and_plot_roc_side_by_side(y_true_binarized_passive, y_scores_passive, 
                                      y_true_binarized_random, y_scores_random, 
                                      results_dir):
    # Compute ROC and AUC for passive
    auc_scores_passive = {}
    fpr_dict_passive = {}
    tpr_dict_passive = {}
    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_binarized_passive[:, i], y_scores_passive[:, i])
        auc = roc_auc_score(y_true_binarized_passive[:, i], y_scores_passive[:, i])
        auc_scores_passive[class_label] = auc
        fpr_dict_passive[class_label] = fpr
        tpr_dict_passive[class_label] = tpr

    # Compute macro-average ROC curve and AUC for passive
    all_fpr_passive = np.unique(np.concatenate([fpr_dict_passive[class_label] for class_label in classes]))
    mean_tpr_passive = np.zeros_like(all_fpr_passive)
    for class_label in classes:
        mean_tpr_passive += np.interp(all_fpr_passive, fpr_dict_passive[class_label], tpr_dict_passive[class_label])
    mean_tpr_passive /= len(classes)
    macro_avg_auc_passive = roc_auc_score(y_true_binarized_passive, y_scores_passive, average="macro", multi_class="ovr")

    # Compute ROC and AUC for random
    auc_scores_random = {}
    fpr_dict_random = {}
    tpr_dict_random = {}
    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_binarized_random[:, i], y_scores_random[:, i])
        auc = roc_auc_score(y_true_binarized_random[:, i], y_scores_random[:, i])
        auc_scores_random[class_label] = auc
        fpr_dict_random[class_label] = fpr
        tpr_dict_random[class_label] = tpr

    # Compute macro-average ROC curve and AUC for random
    all_fpr_random = np.unique(np.concatenate([fpr_dict_random[class_label] for class_label in classes]))
    mean_tpr_random = np.zeros_like(all_fpr_random)
    for class_label in classes:
        mean_tpr_random += np.interp(all_fpr_random, fpr_dict_random[class_label], tpr_dict_random[class_label])
    mean_tpr_random /= len(classes)
    macro_avg_auc_random = roc_auc_score(y_true_binarized_random, y_scores_random, average="macro", multi_class="ovr")

    # Plot side-by-side ROC curves
    set_default_plot_settings(font_size=24, fig_size=[12, 6], dpi=250, 
                              line_color='blue', axes_limit=(-2, 2), line_width=2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Passive ROC curve
    axes[0].plot(all_fpr_passive, mean_tpr_passive, label=f"Macro-Average (AUC = {macro_avg_auc_passive:.2f})", linestyle="-", color="red")
    axes[0].plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    axes[0].set_title("ROC Curve (Passive)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].grid(alpha=0.3)

    # Random ROC curve
    axes[1].plot(all_fpr_random, mean_tpr_random, label=f"Macro-Average (AUC = {macro_avg_auc_random:.2f})", linestyle="-", color="red")
    axes[1].plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    axes[1].set_title("ROC Curve (Perturbation)")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()

    # Save the ROC plot
    roc_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_roc_curves_comparison.svg")
    plt.savefig(roc_filename, pad_inches=0.05, bbox_inches='tight')
    print(f"Figure saved as {roc_filename}")
    plt.show()

# Plot ROC curves for passive and random side by side
compute_and_plot_roc_side_by_side(y_true_binarized_passive, y_scores_passive, 
                                  y_true_binarized_random, y_scores_random, 
                                  results_dir)


#%% Passive Time Change

#Passive
np.random.seed(42)
random.seed(42)

# Define the range of TotalTime values
# time_range = [5, 1000, 2000]
time_range = [5, 100, 1000, 10000]

timechange_passives_file = os.path.join(results_dir, "timechange_passives.pkl")

# Check if the pickle file exists
if os.path.exists(timechange_passives_file):
    print(f"Loading timechange_passives from {timechange_passives_file}")
    with open(timechange_passives_file, "rb") as f:
        timechange_passives = pickle.load(f)
else:
    print(f"Processing TotalTime and saving to {timechange_passives_file}")
    timechange_passives = []
    for TotalTime in tqdm(time_range, desc="Processing TotalTime"):

        results_passive = {}

        for idx, A_group in enumerate([A1, A2, A3, A4, A5]):
            group_label = f"A{idx + 1}"
            fros_trial = []
            A_optimal_trial = []
            A_est_trials = []
            fros_A_trial = []
            fros_W_trial = []
            fros_cost_trial = []

            for trial in range(Trials):
                A = A_group[trial]  # Use the corresponding matrix from the group
                x0 = np.ones([dim, 1])
                x = np.zeros([dim, fs * TotalTime])
                xi = np.zeros([dim, fs * TotalTime])
                x[:, 0] = x0[:, 0]

                # Euler method to simulate the time series
                for i in range(1, fs * TotalTime):
                    xi[:, i - 1] = np.random.normal(0, sigma, dim)
                    x[:, i] = A @ x[:, i - 1] + xi[:, i - 1]

                X_ = x[:, -fs * TotalTime // 2:]

                X = X_[:obs_dim, :-1]
                Y = X_[:obs_dim, 1:]

                # Estimate matrix A
                A_est = Y @ scipy.linalg.pinv(X)

                A_est_trials.append(A_est)
                fros_A_trial.append(np.linalg.norm(A - A_est, 'fro') ** 2)
                A_optimal_trial.append(np.trace(scipy.linalg.inv(X @ X.T)))

            A_est_mean = np.mean(A_est_trials, axis=0)
            results_passive[group_label] = {
                "A_est": A_est_trials,
                "A_est_mean": A_est_mean,
                "fros_A": np.mean(fros_A_trial),
                "A_optimal": np.mean(A_optimal_trial),
                "plot_name": "passive"
            }

        timechange_passives.append(copy.deepcopy(results_passive))

    # Save the results to a pickle file
    with open(timechange_passives_file, "wb") as f:
        pickle.dump(timechange_passives, f)
    print(f"Saved timechange_passives to {timechange_passives_file}")



# Perform MDS and Classification for each time change in timechange_passives
fig, axes = plt.subplots(1, len(time_range), figsize=(6* len(time_range), 6))

for idx, results_passive in enumerate(tqdm(timechange_passives, desc="Processing Time Changes")):
    # Prepare data for MDS
    mds_data = []
    mds_labels = []
    for group_label in ['A1', 'A2', 'A3', 'A4', 'A5']:
        for A_est in results_passive[group_label]["A_est"]:
            mds_data.append(A_est.flatten())
            mds_labels.append(group_label)
    mds_data = np.array(mds_data)

    # Compute pairwise distances and perform MDS
    distance_matrix = squareform(pdist(mds_data, metric='euclidean'))
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_results = mds.fit_transform(distance_matrix)

    # Prepare data for classification
    data = []
    labels = []
    for group_label in ['A1', 'A2', 'A3', 'A4', 'A5']:
        for A_est in results_passive[group_label]["A_est"]:
            data.append(A_est.flatten())
            labels.append(group_label)
    data = np.array(data)
    labels = np.array(labels)

    # Train classifier with 10-fold cross-validation
    clf = LinearDiscriminantAnalysis()
    predicted_labels = []
    true_labels = []

    for train_index, test_index in KFold(n_splits=10, shuffle=True, random_state=42).split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Train the classifier
        clf.fit(X_train, y_train)

        # Predict on the test set
        y_pred = clf.predict(X_test)

        # Collect predictions and true labels
        predicted_labels.extend(y_pred)
        true_labels.extend(y_test)

    # Calculate the mean accuracy
    mean_accuracy = np.mean(np.array(predicted_labels) == np.array(true_labels))

    # Plot MDS results
    for label, color in zip(['A1', 'A2', 'A3', 'A4', 'A5'], ['red', 'blue', 'green', 'purple', 'orange']):
        indices = [i for i, l in enumerate(mds_labels) if l == label]
        axes[idx].scatter(mds_results[indices, 0], mds_results[indices, 1], label=f"State {int(label[1])}", color=color, alpha=0.6)
    if time_range[idx] == 5:
        axes[idx].set_title(f"T : {time_range[idx]}, ACC : {mean_accuracy_passive:.2%}")
    else:
        axes[idx].set_title(f"T : {time_range[idx]}, ACC : {mean_accuracy:.2%}")        
    axes[idx].set_xlabel("MDS 1")
    axes[idx].set_ylabel("MDS 2")
    axes[idx].grid(alpha=0.3)

plt.tight_layout()

# Save the combined MDS and classification plot
timechange_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_timechange_mds_classification.svg")
plt.savefig(timechange_filename, pad_inches=0.05, bbox_inches='tight')
print(f"Figure saved as {timechange_filename}")
plt.show()
# %%
