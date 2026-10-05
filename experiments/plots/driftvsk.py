import numpy as np
import os
import matplotlib.pyplot as plt
OUTPUT_DIR = "results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULT_DIR = "results/final"
k_values = [5, 10, 20, 40]

methods = ["no_gating", "hard_gating", "soft_gating", "random_gating"]
labels = ["No", "Hard", "Soft", "Random"]

# collect data
data = {m: {k: [] for k in k_values} for m in methods}

for file in os.listdir(RESULT_DIR):
    if not file.endswith("_metrics.npy"):
        continue

    d = np.load(os.path.join(RESULT_DIR, file), allow_pickle=True).item()

    for m in methods:
        for k in k_values:
            data[m][k].append(d[m]["metrics"][k]["drift_mean"])

# compute mean
means = {m: [np.mean(data[m][k]) for k in k_values] for m in methods}

# plot
plt.figure(figsize=(6,4))

for m, label in zip(methods, labels):
    plt.plot(k_values, means[m], marker='o', label=label)

plt.xlabel("Temporal window (k)")
plt.ylabel("Drift")
plt.title("Drift vs Temporal Scale (CNN)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/figures/drift_vs_k.png")
plt.show()