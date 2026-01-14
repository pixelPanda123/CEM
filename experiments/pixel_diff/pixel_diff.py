import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
import os

FRAME_DIR = "Datasets/kvasir-capsule/frames/2f513ad4ee5e4630"
OUTPUT_DIR = "results/pixel_diff"
os.makedirs(OUTPUT_DIR, exist_ok=True)
frame_paths = sorted(
    glob.glob("Datasets/kvasir-capsule/frames/2f513ad4ee5e4630/frame_*.jpg")
)

pixel_diff = {} 

for i in range(len(frame_paths) - 1):
    img1 = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(frame_paths[i + 1], cv2.IMREAD_GRAYSCALE)

    diff = np.mean(np.abs(img2.astype(np.float32) - img1.astype(np.float32)))
    pixel_diff[i] = diff


with open(f"{OUTPUT_DIR}/pixel_diff.pkl", "wb") as f:
    pickle.dump(pixel_diff, f)


fps = 2  
times = np.array(list(pixel_diff.keys())) / fps
values = np.array(list(pixel_diff.values()))

plt.figure(figsize=(12, 4))
plt.plot(times, values)
plt.xlabel("Time (seconds)")
plt.ylabel("Mean pixel difference")
plt.title("Pixel Difference Over Time")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pixel_diff.png")
plt.show()
