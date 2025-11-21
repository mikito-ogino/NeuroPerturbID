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
from matplotlib.animation import FuncAnimation
import ffmpeg

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
set_default_plot_settings(font_size=18, dpi=200)
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
        if value == 0:
            plt.text(j, i, f"0.00", ha='center', va='center', color='black', fontsize=20)
        elif abs(value) < 1e-2:
            plt.text(j, i, f"{value:.0e}", ha='center', va='center', color='black', fontsize=20)
        else:
            plt.text(j, i, f"{value:.2f}", ha='center', va='center', color='black', fontsize=20)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent title cutoff
plt.title("True connectivity matrix", pad=20)  # Add padding to the title

# Save the heatmap as a JPEG file
heatmap_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_A_discrete_heatmap_dim{dim}.jpeg")
plt.savefig(heatmap_filename, format='jpeg')
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
graph_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_directed_graph.png")
plt.savefig(graph_filename, format='png', bbox_inches='tight', pad_inches=0.05)
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
    
    # Create an animation of the time series of X

    set_default_plot_settings(font_size=18, dpi=200)
    fig, axs = plt.subplots(4, 1, figsize=(5, 5), sharex=True)
    fig.suptitle(f"{input_type.capitalize()}", fontsize=18)

    lim = 2
    lines = []
    for d in range(dim):
        line, = axs[d % 4].plot([], [], color='blue', linewidth=2)
        lines.append(line)
        axs[d % 4].set_ylabel(f'$x_{d+1}$')
        axs[d % 4].grid(True)
        axs[d % 4].set_yticks([])
        axs[d % 4].set_xticks([])
        axs[d % 4].set_ylim(-lim, lim)
        axs[d % 4].set_xlim(0, X.shape[1] * dt)

    axs[-1].set_xlabel('Time')

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        for d, line in enumerate(lines):
            if d < dim:
                line.set_data(np.arange(frame) * dt, X[d, :frame])
        return lines

    ani = FuncAnimation(fig, update, frames=X.shape[1]+1, init_func=init, blit=True)

    # Save the animation as an MP4 file
    animation_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_time_series_{input_type}.mp4")
    ani.save(animation_filename, fps=15, extra_args=['-vcodec', 'libx264'])
    print(f"Animation saved as {animation_filename}")

    plt.close(fig)

    set_default_plot_settings(font_size=18, dpi=200)
    # Perform PCA on X
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X.T)

    # Create a 3D animation of the PCA results
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(f"PCA space ({input_type.capitalize()})", fontsize=18)

    sc = ax.scatter([], [], [], c=[], cmap='viridis', marker='o', s=5)
    ax.set_xlabel('PC 1', labelpad=-12)
    ax.set_ylabel('PC 2', labelpad=-12)
    ax.set_zlabel('PC 3', labelpad=-12)

    lim = 0.9
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
    plt.tight_layout()

    def init():
        sc._offsets3d = ([], [], [])
        sc.set_array([])
        return sc,

    def update(frame):
        sc._offsets3d = (X_pca[:frame, 0], X_pca[:frame, 1], X_pca[:frame, 2])
        sc.set_array(np.arange(frame))
        return sc,

    ani = FuncAnimation(fig, update, frames=X_pca.shape[0] + 1, init_func=init, blit=False)

    # Save the PCA animation as an MP4 file
    pca_animation_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_PCA_X_{input_type}_{X.shape[1]}.mp4")
    ani.save(pca_animation_filename, fps=15, extra_args=['-vcodec', 'libx264'])
    print(f"PCA animation saved as {pca_animation_filename}")

    plt.close(fig)
         
    set_default_plot_settings(font_size=18, dpi=200)
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.suptitle(f"Estimated matrix ({input_type.capitalize()})", fontsize=18)

    def init():
        heatmap = ax.imshow(np.zeros_like(A_est), cmap='viridis', interpolation='nearest', vmin=-0.4, vmax=0.4)
        ax.set_xticks(np.arange(A_est.shape[1]))
        ax.set_xticklabels(range(1, A_est.shape[1] + 1))
        ax.set_yticks(np.arange(A_est.shape[0]))
        ax.set_yticklabels(range(1, A_est.shape[0] + 1))
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Dimension')
        return heatmap,

    def update(frame):
        partial_X = X[:, :frame]
        partial_U = U[:, :frame]
        partial_Y = Y[:, :frame]

        partial_Z = np.vstack([partial_X, partial_U])
        try:
            partial_Phi_est = partial_Y @ scipy.linalg.pinv(partial_Z)
        except:
            partial_Phi_est = np.zeros((partial_Y.shape[0], partial_Z.shape[1]))
        partial_A_est = partial_Phi_est[:, :dim]

        heatmap = ax.imshow(partial_A_est, cmap='viridis', interpolation='nearest', vmin=-0.4, vmax=0.4)
        ax.clear()
        ax.imshow(partial_A_est, cmap='viridis', interpolation='nearest', vmin=-0.4, vmax=0.4)

        # Set ticks and labels
        ax.set_xticks(np.arange(A_est.shape[1]))
        ax.set_xticklabels(range(1, A_est.shape[1] + 1))
        ax.set_yticks(np.arange(A_est.shape[0]))
        ax.set_yticklabels(range(1, A_est.shape[0] + 1))

        # Display the values on the heatmap
        for i in range(partial_A_est.shape[0]):
            for j in range(partial_A_est.shape[1]):
                value = partial_A_est[i, j]
                if value == 0:
                    ax.text(j, i, f"0.00", ha='center', va='center', color='black', fontsize=18)
                elif abs(value) < 1e-2:
                    ax.text(j, i, f"0.00", ha='center', va='center', color='black', fontsize=18)
                else:
                    ax.text(j, i, f"{value:.2f}", ha='center', va='center', color='black', fontsize=18)

        return heatmap,

    ani = FuncAnimation(fig, update, frames=X.shape[1] + 1, init_func=init, blit=False)

    # Save the heatmap animation as an MP4 file
    heatmap_animation_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_A_est_heatmap_animation_{input_type}.mp4")
    ani.save(heatmap_animation_filename, fps=15, extra_args=['-vcodec', 'libx264'])
    print(f"Heatmap animation saved as {heatmap_animation_filename}")

    plt.close(fig)
    
    
    # Calculate and animate the trace of the inverse of the covariance of X
    traces = []
    for frame in range(1, X.shape[1] + 1):
        partial_X = X[:, :frame]
        Sigma_X = np.cov(partial_X)
        try:
            trace_value = np.trace(np.linalg.inv(Sigma_X))
            if np.isnan(trace_value) or np.isinf(trace_value) or abs(trace_value) > 1e4:
                trace_value = 1e4
        except np.linalg.LinAlgError:
            trace_value = 1e4
        
        traces.append(trace_value)

    set_default_plot_settings(font_size=18, dpi=200)
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.suptitle(f"Objective function to be minimized ({input_type.capitalize()})", fontsize=18)
    line, = ax.plot([], [], color='blue', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Objective function')
    ax.grid(True)
    ax.set_xlim(0, X.shape[1] * dt)
    ax.set_ylim(0, max(traces) * 1.1)
    plt.tight_layout()

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        line.set_data(np.arange(1, frame + 1) * dt, traces[:frame])
        return line,

    ani = FuncAnimation(fig, update, frames=X.shape[1] + 1, init_func=init, blit=True)

    # Save the animation as an MP4 file
    trace_animation_filename = os.path.join(results_dir, f"{__file__.split('/')[-1].split('.')[0]}_trace_inverse_covariance_{input_type}.mp4")
    ani.save(trace_animation_filename, fps=15, extra_args=['-vcodec', 'libx264'])
    print(f"Trace animation saved as {trace_animation_filename}")

    plt.close(fig)
# %%
import matplotlib.pyplot as plt

# フィギュアを作成（正方形にするためfigsizeを調整）
fig, ax = plt.subplots(figsize=(5, 5))
ax.axis('off')  # 軸は非表示

# テキストを中央に描画（LaTeXスタイルで数式を綺麗に）
text = "Used model\n\n$\\mathbf{x}(t+1) = A\\mathbf{x}(t) + B\\mathbf{u}(t) + \\mathbf{\\xi}(t)$\n\n\n\nObjective function\n\n$\\mathrm{tr}(\\mathbf{\\Sigma}_\\mathbf{X}^{-1})$"
ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=24, fontname='Times New Roman')

# 画像として保存（背景白・枠なし）
plt.savefig("results/used_model.jpeg", bbox_inches='tight', pad_inches=0.5, dpi=300)
plt.show()

# Create an image with the text "Objective function\ntr(\mathbf{\Sigma}_\mathbf{X}^{-1})"
fig, ax = plt.subplots(figsize=(5, 5))
ax.axis('off')  # Hide axes

# Add the text in LaTeX style
text = ""
ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=16)

# Save the image as a JPEG file
objective_function_image = "results/objective_function.jpeg"
plt.savefig(objective_function_image, bbox_inches='tight', pad_inches=0.5, dpi=300)
plt.show()

# Combine 9 videos into a single grid using ffmpeg
import subprocess

# Add JPEG images as static frames to the video grid
jpeg_images = [
    "results/Video1_A_discrete_heatmap_dim4.jpeg",
    "results/used_model.jpeg",
    "results/objective_function.jpeg"
]

# Convert JPEG images to MP4 videos with static frames
converted_figures = []
for jpeg in jpeg_images:
    output_video = jpeg.replace(".jpeg", ".mp4")
    # ffmpegコマンド
    cmd = [
        'ffmpeg',
        '-loop', '1',
        '-t', '5',
        '-i', jpeg,
        "-vf", "scale=512:512",
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',
        output_video
    ]

    # 実行
    subprocess.run(cmd, check=True)
    converted_figures.append(output_video)

# build input
input_videos1 = [
    converted_figures[0],
    "results/Video1_time_series_passive.mp4",
    "results/Video1_PCA_X_passive_99.mp4",
    "results/Video1_trace_inverse_covariance_passive.mp4",
    "results/Video1_A_est_heatmap_animation_passive.mp4",
    converted_figures[1],
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
]

input_videos2 = [
    converted_figures[0],
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
    converted_figures[1],
    "results/Video1_time_series_impulse.mp4",
    "results/Video1_PCA_X_impulse_99.mp4",
    "results/Video1_trace_inverse_covariance_impulse.mp4",
    "results/Video1_A_est_heatmap_animation_impulse.mp4",
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
]

input_videos3 = [
    converted_figures[0],
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
    converted_figures[1],
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
    converted_figures[2],
    "results/Video1_time_series_cos.mp4",
    "results/Video1_PCA_X_cos_99.mp4",
    "results/Video1_trace_inverse_covariance_cos.mp4",
    "results/Video1_A_est_heatmap_animation_cos.mp4"
]

input_videos4 = [
    converted_figures[0],
    "results/Video1_time_series_passive.mp4",
    "results/Video1_PCA_X_passive_99.mp4",
    "results/Video1_trace_inverse_covariance_passive.mp4",
    "results/Video1_A_est_heatmap_animation_passive.mp4",
    converted_figures[1],
    "results/Video1_time_series_impulse.mp4",
    "results/Video1_PCA_X_impulse_99.mp4",
    "results/Video1_trace_inverse_covariance_impulse.mp4",
    "results/Video1_A_est_heatmap_animation_impulse.mp4",
    converted_figures[2],
    "results/Video1_time_series_cos.mp4",
    "results/Video1_PCA_X_cos_99.mp4",
    "results/Video1_trace_inverse_covariance_cos.mp4",
    "results/Video1_A_est_heatmap_animation_cos.mp4"
]

output_combined_video = "results/combined_grid.mp4"

filter_complex = ";".join(
    [f"[{i}:v] setpts=PTS-STARTPTS, scale=640x360 [v{i}]" for i in range(15)]
) + "; " + "".join(
    [f"[v{i}]" for i in range(15)]
) + \
"xstack=inputs=15:layout=0_0|640_0|1280_0|1920_0|2560_0|0_360|640_360|1280_360|1920_360|2560_360|0_720|640_720|1280_720|1920_720|2560_720[tmp];" + \
"[tmp] pad=width=3200:height=1130:x=0:y=50:color=white[tmp2];" + \
"[tmp2] drawbox=x=640:y=410:w=2560:h=1:color=black@1:t=fill[tmp3];" + \
"[tmp3] drawbox=x=640:y=770:w=2560:h=1:color=black@1:t=fill[out]"

for idx, input_videos in enumerate([input_videos1, input_videos2, input_videos3, input_videos4], start=1):
    output_combined_video = f"results/combined_grid_{idx}.mp4"
    
    ffmpeg_command = [
        "ffmpeg",
        *[item for video in input_videos for item in ["-i", video]],
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "veryfast",
        output_combined_video
    ]

    # Remove the output file if it already exists
    if os.path.exists(output_combined_video):
        os.remove(output_combined_video)

    # Run the ffmpeg command
    subprocess.run(ffmpeg_command, check=True)
    print(f"✅ Combined video {idx} saved as {output_combined_video}")

# Concatenate all combined videos into a single video
final_output_video = "results/final_combined_video.mp4"

# List of combined videos to concatenate
combined_videos = [
    "results/combined_grid_1.mp4",
    "results/combined_grid_2.mp4",
    "results/combined_grid_3.mp4",
    "results/combined_grid_4.mp4"
]

# 絶対パスに変換
combined_videos = [os.path.abspath(video) for video in combined_videos]

# Create a text file listing all videos to concatenate with a 1-second pause
concat_file = "results/concat_list.txt"

with open(concat_file, "w") as f:
    for video in combined_videos:
        # 1. 最後のフレームを画像として抽出
        last_frame_image = f"{os.path.splitext(video)[0]}_last_frame.png"
        subprocess.run([
            "ffmpeg",
            "-sseof", "-0.1",  # 最後の直前フレームを狙う（-0.01だと動作しない場合がある）
            "-i", video,
            "-vframes", "1",
            "-q:v", "2",
            last_frame_image
        ], check=True)

        # 2. その画像を1秒の映像にする
        pause_video = f"{os.path.splitext(video)[0]}_pause.mp4"
        subprocess.run([
            "ffmpeg",
            "-loop", "1",
            "-i", last_frame_image,
            "-t", "1",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # x264用に解像度を偶数に
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",
            pause_video
        ], check=True)

        with open(concat_file, "a") as f:
            f.write(f"file '{video}'\n")
            f.write(f"file '{os.path.abspath(pause_video)}'\n")

# Run ffmpeg to concatenate the videos
ffmpeg_concat_command = [
    "ffmpeg",
    "-f", "concat",
    "-safe", "0",
    "-i", concat_file,
    "-c", "copy",
    final_output_video
]

# Remove the output file if it already exists
if os.path.exists(final_output_video):
    os.remove(final_output_video)

# Execute the ffmpeg command
subprocess.run(ffmpeg_concat_command, check=True)
print(f"✅ Final combined video saved as {final_output_video}")