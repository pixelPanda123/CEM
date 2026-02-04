import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load data
# -----------------------
EMBED_PATH = "results/cnn_embedding/embedding_motion.pkl"
LABELS_PATH = "results/tmd/cnn_regime_labels_K2.npy"
TIME_PATH = "results/tmd/cnn_tmd_times.npy"
OUTPUT_DIR = "results/tmd"

import pickle
with open(EMBED_PATH, "rb") as f:
    embedding_motion = pickle.load(f)

times = np.array(sorted(embedding_motion.keys()))
values = np.array([embedding_motion[t] for t in times])

labels = np.load(LABELS_PATH)

# Align lengths (TMD windows start later)
values = values[:len(labels)]

# -----------------------
# Identify stable regime
# -----------------------
unique, counts = np.unique(labels, return_counts=True)
stable_regime = unique[np.argmax(counts)]

print("Stable regime:", stable_regime)

# -----------------------
# Compute cumulative motion
# -----------------------
cumulative_naive = np.cumsum(values)

cumulative_regime = np.cumsum(
    values * (labels == stable_regime)
)

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(14, 6))
plt.plot(cumulative_naive, label="Naive cumulative motion", alpha=0.7)
plt.plot(cumulative_regime, label="Regime-aware cumulative motion", linewidth=2)

plt.xlabel("Time (windows)")
plt.ylabel("Cumulative embedding change")
plt.title("Naive vs Regime-Aware Cumulative Motion")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/regime_aware_cumulative_motion.png")
plt.show()
