import numpy as np
import matplotlib.pyplot as plt
import os
VIDEO_ID = os.environ.get("VIDEO_ID", "2f513ad4ee5e4630")
alpha = np.load(f"results/regime_modeling/cnn_hmm/{VIDEO_ID}_posterior.npy")

plt.figure(figsize=(12,4))
plt.plot(alpha)
plt.title("HMM Stable Regime Probability Over Time")
plt.ylim(0,1)
plt.show()