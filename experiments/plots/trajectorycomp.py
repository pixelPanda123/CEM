import numpy as np
import matplotlib.pyplot as plt
import os

VIDEO_ID = "2f513ad4ee5e4630"  # pick one good example

BASE = "results/latent_trajectory"

z_no = np.load(f"{BASE}/{VIDEO_ID}_z_no.npy")
z_hard = np.load(f"{BASE}/{VIDEO_ID}_z_hard.npy")
z_soft = np.load(f"{BASE}/{VIDEO_ID}_z_soft.npy")

plt.figure(figsize=(6,6))

plt.plot(z_no[:,0], z_no[:,1], label="No", alpha=0.7)
plt.plot(z_hard[:,0], z_hard[:,1], label="Hard", alpha=0.7)
plt.plot(z_soft[:,0], z_soft[:,1], label="Soft", linewidth=2)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectory Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/figures/trajectory_comparison.png")
plt.show()