import cv2
import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt

FRAME_DIR = "Datasets/kvasir-capsule/frames/2f513ad4ee5e4630"
OUTPUT_DIR = "results/optical_flow"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# Load frames
# ================================
frame_paths = sorted(glob.glob(f"{FRAME_DIR}/frame_*.jpg"))

# ================================
# Initialize
# ================================
prev = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)

flow_mag = []
flow_vectors = []

# ================================
# Optical Flow Loop
# ================================
for i in range(1, len(frame_paths)):
    curr = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)

    flow = cv2.calcOpticalFlowFarneback(
        prev, curr,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # --- Magnitude ---
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    flow_mag.append(np.mean(mag))

    # ================================
    # Robust Motion Extraction
    # ================================
    threshold = np.percentile(mag, 70)  # keep top 30% strongest motion
    mask = mag > threshold

    if np.sum(mask) < 50:
        # fallback (too few strong pixels)
        dx = np.median(flow[..., 0])
        dy = np.median(flow[..., 1])
    else:
        dx = np.mean(flow[..., 0][mask])
        dy = np.mean(flow[..., 1][mask])

    flow_vectors.append([dx, dy])

    prev = curr

# ================================
# Convert to numpy
# ================================
flow_mag = np.array(flow_mag)
flow_vectors = np.array(flow_vectors)

# ================================
# Temporal Smoothing (CRITICAL)
# ================================
window = 5
smoothed_vectors = []

for i in range(len(flow_vectors)):
    start = max(0, i - window)
    end = min(len(flow_vectors), i + window)
    smoothed_vectors.append(np.mean(flow_vectors[start:end], axis=0))

flow_vectors = np.array(smoothed_vectors)

# ================================
# Normalize (VERY IMPORTANT)
# ================================
flow_vectors = flow_vectors / (np.std(flow_vectors) + 1e-8)

# ================================
# Debug Stats (IMPORTANT)
# ================================
print("\nFlow Debug Stats:")
print("Shape:", flow_vectors.shape)
print("Mean:", np.mean(flow_vectors))
print("Std:", np.std(flow_vectors))
print("Min:", np.min(flow_vectors), "Max:", np.max(flow_vectors))

# ================================
# Save outputs
# ================================
np.save(f"{OUTPUT_DIR}/flow_vectors.npy", flow_vectors)

with open(f"{OUTPUT_DIR}/flow_mag.pkl", "wb") as f:
    pickle.dump(flow_mag, f)

print("\nSaved flow vectors:", flow_vectors.shape)

# ================================
# Plot magnitude
# ================================
fps = 2
times = np.arange(len(flow_mag)) / fps

plt.figure(figsize=(12, 4))
plt.plot(times, flow_mag)
plt.xlabel("Time (seconds)")
plt.ylabel("Mean optical flow magnitude")
plt.title("Farneback Optical Flow Magnitude Over Time")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/flow_mag.png")
plt.show()