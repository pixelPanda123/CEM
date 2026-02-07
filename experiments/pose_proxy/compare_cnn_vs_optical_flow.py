import numpy as np
import pickle

# -----------------------
# Paths
# -----------------------
CNN_PATH = "results/cnn_embedding/embedding_motion.pkl"
FLOW_PATH = "results/optical_flow/flow_mag.pkl"
LABELS_PATH = "results/tmd/cnn_regime_labels_K2.npy"

# -----------------------
# Load data
# -----------------------
with open(CNN_PATH, "rb") as f:
    cnn_motion = pickle.load(f)

with open(FLOW_PATH, "rb") as f:
    flow_motion = pickle.load(f)

labels = np.load(LABELS_PATH)

# Sort by time and align
times = np.array(sorted(cnn_motion.keys()))
cnn_d = np.array([cnn_motion[t] for t in times])
flow_d = np.array([flow_motion[t] for t in times])

T = len(labels)
cnn_d = cnn_d[:T]
flow_d = flow_d[:T]
labels = labels[:T]

# -----------------------
# Identify stable regime
# -----------------------
unique, counts = np.unique(labels, return_counts=True)
stable_regime = unique[np.argmax(counts)]
print("Stable regime:", stable_regime)

# -----------------------
# Helper: compute metrics
# -----------------------
def compute_metrics(d, labels, stable_regime):
    p_naive = np.cumsum(d)
    p_regime = np.cumsum(d * (labels == stable_regime))

    v_naive = np.diff(p_naive)
    v_regime = np.diff(p_regime)

    naive_updates = v_naive > 0
    regime_updates = v_regime > 0
    stable_mask = labels[:-1] == stable_regime

    metrics = {}

    # Update sparsity
    metrics["update_rate_naive"] = np.mean(naive_updates)
    metrics["update_rate_regime"] = np.mean(regime_updates)

    # Conditional velocity variance
    metrics["var_naive_cond"] = np.var(v_naive[naive_updates])
    metrics["var_regime_cond"] = np.var(v_regime[regime_updates])

    # Drift efficiency
    metrics["eff_naive"] = p_naive[-1] / np.count_nonzero(naive_updates)
    metrics["eff_regime"] = p_regime[-1] / max(np.count_nonzero(regime_updates), 1)

    # Regime alignment
    metrics["align_naive"] = np.mean(stable_mask[naive_updates])
    metrics["align_regime"] = np.mean(stable_mask[regime_updates])

    return metrics

# -----------------------
# Compute metrics
# -----------------------
cnn_metrics = compute_metrics(cnn_d, labels, stable_regime)
flow_metrics = compute_metrics(flow_d, labels, stable_regime)

# -----------------------
# Print comparison
# -----------------------
def print_metrics(name, m):
    print(f"\n--- {name} ---")
    print(f"Update rate (naive / regime): {m['update_rate_naive']:.3f} / {m['update_rate_regime']:.3f}")
    print(f"Cond. var (naive / regime): {m['var_naive_cond']:.2f} / {m['var_regime_cond']:.2f}")
    print(f"Drift eff (naive / regime): {m['eff_naive']:.2f} / {m['eff_regime']:.2f}")
    print(f"Alignment (naive / regime): {m['align_naive']:.3f} / {m['align_regime']:.3f}")

print_metrics("CNN Embedding Proxy", cnn_metrics)
print_metrics("Optical Flow Proxy", flow_metrics)

