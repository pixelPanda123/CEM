import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load TMD
# -----------------------
TMD_PATH = "results/tmd/cnn_tmd_raw.npy"
TIME_PATH = "results/tmd/cnn_tmd_times.npy"
TMD_NORM_PATH = "results/tmd/cnn_tmd_norm.npy"

tmd = np.load(TMD_PATH)
tmd_norm = np.load(TMD_NORM_PATH)
times = np.load(TIME_PATH)

mu = tmd[:, 0]
var = tmd[:, 1]
tcs = tmd[:, 2]

# Normalize time for coloring
time_norm = (times - times.min()) / (times.max() - times.min())

# -----------------------
# Plot settings
# -----------------------
plt.figure(figsize=(18, 5))

# μ vs variance
plt.subplot(1, 4, 1)
plt.scatter(mu, var, c=time_norm, cmap="viridis", s=8)
plt.xlabel("Mean embedding change (μ)")
plt.ylabel("Variance of change (σ²)")
plt.title("μ vs Variance")

# μ vs TCS
plt.subplot(1, 4, 2)
plt.scatter(mu, tcs, c=time_norm, cmap="viridis", s=8)
plt.xlabel("Mean embedding change (μ)")
plt.ylabel("Temporal Consistency Score (TCS)")
plt.title("μ vs TCS")

# variance vs TCS
plt.subplot(1, 4, 3)
plt.scatter(var, tcs, c=time_norm, cmap="viridis", s=8)
plt.xlabel("Variance of change (σ²)")
plt.ylabel("Temporal Consistency Score (TCS)")
plt.title("Variance vs TCS")


#TMD space
plt.subplot(1,4,4)
plt.scatter(tmd[:,0], tmd[:,1], s=5)
plt.title("TMD Feature Space")
plt.show()

plt.tight_layout()
plt.show()
