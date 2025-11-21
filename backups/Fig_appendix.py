#%%
# Plot f(ω) = 1/(r^2-2r cos(θ-ω)+1) + 1/(r^2-2r cos(θ+ω)+1)
# for ω over one period [0, 2π], with θ and r fixed.
# You can change r and theta below if you like.

import numpy as np
import os
import matplotlib.pyplot as plt

# --- parameters you can edit ---
r = 0.8           # radius (>=0). try also r=1.0 or r=1.2
theta = np.pi/3   # in radians
# -------------------------------
omega_shifted = np.linspace(-np.pi, np.pi, 2000)

f1_shifted = 1/(r**2 - 2*r*np.cos(theta - omega_shifted) + 1)
f2_shifted = 1/(r**2 - 2*r*np.cos(theta + omega_shifted) + 1)
f_sum_shifted = f1_shifted + f2_shifted

# clip for visualization
f1c_s = np.clip(f1_shifted, -1e3, 1e3)
f2c_s = np.clip(f2_shifted, -1e3, 1e3)
fsc_s = np.clip(f_sum_shifted, -1e3, 1e3)

fig, axs = plt.subplots(1, 3, figsize=(12,4), sharex=True)

axs[0].plot(omega_shifted, f1c_s, color="blue")
axs[0].set_title("f(ω) = 1/(r² - 2r cos(θ-ω) + 1)")
axs[0].grid(True, linestyle="--", alpha=0.5)

axs[1].plot(omega_shifted, f2c_s, color="red")
axs[1].set_title("f(-ω) = 1/(r² - 2r cos(θ+ω) + 1)")
axs[1].grid(True, linestyle="--", alpha=0.5)

axs[2].plot(omega_shifted, fsc_s, color="green")
axs[2].set_title("f(ω) + f(-ω)")
axs[2].set_xlabel("ω (radians)")
axs[2].grid(True, linestyle="--", alpha=0.5)

fig.suptitle(f"Plots of f(ω), f(-ω), and f(ω)+f(-ω) (r={r:.3f}, θ={theta:.3f} rad), ω∈[-π, π]", fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save as PDF in results folder with program name
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
program_name = os.path.splitext(os.path.basename(__file__))[0]
pdf_path = os.path.join(output_dir, f"{program_name}.pdf")
plt.savefig(pdf_path, format="pdf")

plt.show()