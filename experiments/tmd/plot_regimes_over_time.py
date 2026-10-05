import numpy as np
import matplotlib.pyplot as plt
import os 

VIDEO_ID = os.environ.get("VIDEO_ID", "2f513ad4ee5e4630")
# -----------------------
# Load data
# -----------------------
TIME_PATH = f"results/tmd/{VIDEO_ID}_cnn_tmd_times.npy"
OUTPUT_DIR = "results/tmd"

times = np.load(TIME_PATH)

labels_K2 = np.load(f"results/tmd/{VIDEO_ID}_cnn_regime_labels_K2.npy")
labels_K3 = np.load(f"results/tmd/{VIDEO_ID}_cnn_regime_labels_K3.npy")

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(16, 6))

# K = 2
plt.subplot(2, 1, 1)
plt.scatter(times, labels_K2, c=labels_K2, cmap="tab10", s=8)
plt.yticks([0, 1])
plt.ylabel("Regime (K=2)")
plt.title("Temporal Regime Assignment (K=2)")

# K = 3
plt.subplot(2, 1, 2)
plt.scatter(times, labels_K3, c=labels_K3, cmap="tab10", s=8)
plt.yticks([0, 1, 2])
plt.xlabel("Time (frames)")
plt.ylabel("Regime (K=3)")
plt.title("Temporal Regime Assignment (K=3)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/{VIDEO_ID}_plot_regimes_over_time.png")
plt.show()