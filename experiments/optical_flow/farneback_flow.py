import cv2
import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt

FRAME_DIR = "Datasets/kvasir-capsule/frames/2f513ad4ee5e4630"   
OUTPUT_DIR = "results/optical_flow"
os.makedirs(OUTPUT_DIR, exist_ok=True)



frame_paths = sorted(glob.glob(f"{FRAME_DIR}/frame_*.jpg"))
flow_mag = {}

#OPTICAL FLOW 

prev = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)

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

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    flow_mag[i - 1] = np.mean(mag)

    prev = curr

with open(f"{OUTPUT_DIR}/flow_mag.pkl", "wb") as f:
    pickle.dump(flow_mag, f)

fps = 2  # extraction FPS
times = np.array(list(flow_mag.keys())) / fps
values = np.array(list(flow_mag.values()))

plt.figure(figsize=(12, 4))
plt.plot(times, values)
plt.xlabel("Time (seconds)")
plt.ylabel("Mean optical flow magnitude")
plt.title("Farneback Optical Flow Magnitude Over Time")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/flow_mag.png")
plt.show()