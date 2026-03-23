import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -----------------------
# Config
# -----------------------
EMBED_PATH = "results/cnn_embedding/embedding_vectors.pkl"
POSTERIOR_PATH = "results/regime_modeling/cnn_hmm/posterior.npy"
OUTPUT_DIR = "results/latent_trajectory"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Load embeddings
# -----------------------
with open(EMBED_PATH, "rb") as f:
    embedding_motion = pickle.load(f)

# Sort by time
times = np.array(sorted(embedding_motion.keys()))
embeddings = np.array([embedding_motion[t] for t in times])

print("Embeddings shape:", embeddings.shape)

# -----------------------
# Compute delta embeddings
# -----------------------
delta_e = embeddings[1:] - embeddings[:-1]

print("Delta shape:", delta_e.shape)

# -----------------------
# PCA projection to 2D
# -----------------------
pca = PCA(n_components=2)
delta_z = pca.fit_transform(delta_e)

print("Projected delta shape:", delta_z.shape)

# -----------------------
# Load regime probabilities
# -----------------------
alpha = np.load(POSTERIOR_PATH)

# Align lengths
min_len = min(len(delta_z), len(alpha))

delta_z = delta_z[:min_len]
alpha = alpha[:min_len]

# Hard threshold (for comparison)
r = (alpha > 0.5).astype(float)

# -----------------------
# Initialize trajectories
# -----------------------
T = min_len

z_no = np.zeros((T, 2))
z_hard = np.zeros((T, 2))
z_soft = np.zeros((T, 2))

# -----------------------
# Integrate trajectories
# -----------------------
for t in range(1, T):
    # No gating
    z_no[t] = z_no[t-1] + delta_z[t]

    # Hard gating
    z_hard[t] = z_hard[t-1] + r[t] * delta_z[t]

    # Soft gating
    z_soft[t] = z_soft[t-1] + alpha[t] * delta_z[t]

# -----------------------
# Save results
# -----------------------
np.save(f"{OUTPUT_DIR}/z_no.npy", z_no)
np.save(f"{OUTPUT_DIR}/z_hard.npy", z_hard)
np.save(f"{OUTPUT_DIR}/z_soft.npy", z_soft)

# -----------------------
# Visualization
# -----------------------
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.plot(z_no[:,0], z_no[:,1])
plt.title("No Gating")

plt.subplot(1,3,2)
plt.plot(z_hard[:,0], z_hard[:,1])
plt.title("Hard Gating")

plt.subplot(1,3,3)
plt.plot(z_soft[:,0], z_soft[:,1])
plt.title("Soft Gating")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/trajectory_comparison.png")
plt.show()

print("Latent trajectory computation complete.")