import numpy as np
from collections import defaultdict

def compute_regime_stats(labels):
    T = len(labels)

    # -----------------------
    # Regime durations
    # -----------------------
    durations = defaultdict(list)

    current_label = labels[0]
    current_length = 1

    for i in range(1, T):
        if labels[i] == current_label:
            current_length += 1
        else:
            durations[current_label].append(current_length)
            current_label = labels[i]
            current_length = 1

    durations[current_label].append(current_length)

    # -----------------------
    # Flicker rate
    # -----------------------
    switches = np.sum(labels[1:] != labels[:-1])
    flicker_rate = switches / T

    # -----------------------
    # Dominance ratio
    # -----------------------
    unique, counts = np.unique(labels, return_counts=True)
    dominance = dict(zip(unique, counts / T))

    return durations, flicker_rate, dominance


# -----------------------
# Load regime labels
# -----------------------
labels_K2 = np.load("results/tmd/cnn_regime_labels_K2.npy")
labels_K3 = np.load("results/tmd/cnn_regime_labels_K3.npy")

print("\n===== K = 2 =====")
dur_K2, flicker_K2, dom_K2 = compute_regime_stats(labels_K2)

for r in dur_K2:
    print(f"Regime {r}: mean duration = {np.mean(dur_K2[r]):.2f}, "
          f"median = {np.median(dur_K2[r]):.2f}")

print(f"Flicker rate: {flicker_K2:.4f}")
print("Dominance:", dom_K2)

print("\n===== K = 3 =====")
dur_K3, flicker_K3, dom_K3 = compute_regime_stats(labels_K3)

for r in dur_K3:
    print(f"Regime {r}: mean duration = {np.mean(dur_K3[r]):.2f}, "
          f"median = {np.median(dur_K3[r]):.2f}")

print(f"Flicker rate: {flicker_K3:.4f}")
print("Dominance:", dom_K3)
