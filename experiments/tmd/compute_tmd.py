import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# ================================
# Config
# ================================
VIDEO_ID = os.environ.get("VIDEO_ID", "2f513ad4ee5e4630")

EMBED_PATH = f"results/cnn_embedding/{VIDEO_ID}_embedding_motion.pkl"
OUTPUT_DIR = "results/tmd"
WINDOW_SIZE = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nProcessing VIDEO_ID: {VIDEO_ID}")

# ================================
# Load embedding motion
# ================================
with open(EMBED_PATH, "rb") as f:
    embedding_motion = pickle.load(f)

# Sort by time
times = np.array(sorted(embedding_motion.keys()))
values = np.array([embedding_motion[t] for t in times])

print(f"Loaded motion signal of length: {len(values)}")

# ================================
# Compute TMD
# ================================
tmd = []
tmd_times = []

for i in range(len(values) - WINDOW_SIZE + 1):
    window = values[i : i + WINDOW_SIZE]

    mu = np.mean(window)
    var = np.var(window)
    tcs = np.mean(np.abs(np.diff(window)))

    tmd.append([mu, var, tcs])
    tmd_times.append(times[i])

tmd = np.array(tmd)
tmd_times = np.array(tmd_times)

# ================================
# Normalize
# ================================
scaler = StandardScaler()
tmd_norm = scaler.fit_transform(tmd)

# ================================
# Save
# ================================
np.save(f"{OUTPUT_DIR}/{VIDEO_ID}_cnn_tmd_raw.npy", tmd)
np.save(f"{OUTPUT_DIR}/{VIDEO_ID}_cnn_tmd_norm.npy", tmd_norm)
np.save(f"{OUTPUT_DIR}/{VIDEO_ID}_cnn_tmd_times.npy", tmd_times)

print("TMD computation complete.")
print("Raw TMD shape:", tmd.shape)
print("Normalized TMD shape:", tmd_norm.shape)