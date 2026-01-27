import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------
# Paths
# -----------------------
EMBED_PATH = "results/vit_embedding/embedding_motion.pkl"
OUTPUT_DIR = "results/temporal_windowing"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Load ViT embedding motion
# -----------------------
with open(EMBED_PATH, "rb") as f:
    embedding_motion = pickle.load(f)

# Convert to ordered arrays
times = np.array(sorted(embedding_motion.keys()))
values = np.array([embedding_motion[t] for t in times])

# -----------------------
# Temporal windowing
# -----------------------
W = 5  # window size in frames

mean_window = []
var_window = []
window_times = []

for i in range(len(values) - W + 1):
    window = values[i:i + W]
    mean_window.append(np.mean(window))
    var_window.append(np.var(window))
    window_times.append(times[i])

mean_window = np.array(mean_window)
var_window = np.array(var_window)
window_times = np.array(window_times)

# -----------------------
# Save outputs
# -----------------------
with open(f"{OUTPUT_DIR}/vit_mean_window_W{W}.pkl", "wb") as f:
    pickle.dump(mean_window, f)

with open(f"{OUTPUT_DIR}/vit_var_window_W{W}.pkl", "wb") as f:
    pickle.dump(var_window, f)

# -----------------------
# Plot
# -----------------------
fps = 2
t_sec = window_times / fps

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t_sec, mean_window)
plt.ylabel("Mean embedding distance")
plt.title(f"ViT Temporal Windowing (W={W})")

plt.subplot(2, 1, 2)
plt.plot(t_sec, var_window)
plt.xlabel("Time (seconds)")
plt.ylabel("Variance of embedding distance")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/vit_temporal_window_W{W}.png")
plt.show()
