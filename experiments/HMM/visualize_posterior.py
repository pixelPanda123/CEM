import numpy as np
import matplotlib.pyplot as plt

alpha = np.load("results/regime_modeling/cnn_hmm/posterior.npy")

plt.figure(figsize=(12,4))
plt.plot(alpha)
plt.title("HMM Stable Regime Probability Over Time")
plt.ylim(0,1)
plt.show()