import numpy as np
import os
import matplotlib.pyplot as plt

RESULT_DIR = "results/final"
K = 20  # main result

methods = ["no_gating", "hard_gating", "soft_gating", "random_gating"]
labels = ["No", "Hard", "Soft", "Random"]

drift = {m: [] for m in methods}
motion = {m: [] for m in methods}

for file in os.listdir(RESULT_DIR):
    if not file.endswith("_metrics.npy"):
        continue

    d = np.load(os.path.join(RESULT_DIR, file), allow_pickle=True).item()

    for m in methods:
        drift[m].append(d[m]["metrics"][K]["drift_mean"])
        motion[m].append(d[m]["metrics"][K]["motion_mean"])

# mean values
drift_mean = {m: np.mean(drift[m]) for m in methods}
motion_mean = {m: np.mean(motion[m]) for m in methods}

# plot
plt.figure(figsize=(6,4))

for m, label in zip(methods, labels):
    plt.scatter(motion_mean[m], drift_mean[m], label=label, s=100)

plt.xlabel("Motion magnitude")
plt.ylabel("Drift")
plt.title("Drift vs Motion (k=20)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/figures/drift_vs_motion.png")
plt.show()