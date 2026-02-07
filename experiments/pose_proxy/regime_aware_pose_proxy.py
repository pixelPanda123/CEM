import numpy as np
import pickle

# -----------------------
# Load data
# -----------------------
EMBED_PATH = "results/cnn_embedding/embedding_motion.pkl"
LABELS_PATH = "results/tmd/cnn_regime_labels_K2.npy"

with open(EMBED_PATH, "rb") as f:
    embedding_motion = pickle.load(f)

times = np.array(sorted(embedding_motion.keys()))
d = np.array([embedding_motion[t] for t in times])

labels = np.load(LABELS_PATH)

# Align lengths
T = len(labels)
d = d[:T]
labels = labels[:T]

# -----------------------
# Identify stable regime
# -----------------------
unique, counts = np.unique(labels, return_counts=True)
stable_regime = unique[np.argmax(counts)]

print("Stable regime:", stable_regime)

# -----------------------
# Relative pose proxies
# -----------------------
p_naive = np.cumsum(d)
p_regime = np.cumsum(d * (labels == stable_regime))

# Velocities
v_naive = np.diff(p_naive)
v_regime = np.diff(p_regime)

# Masks
naive_updates = v_naive > 0
regime_updates = v_regime > 0
stable_mask = labels[:-1] == stable_regime

# -----------------------
# Metric 1: Update sparsity
# -----------------------
update_rate_naive = np.mean(naive_updates)
update_rate_regime = np.mean(regime_updates)

# -----------------------
# Metric 2: Conditional velocity variance
# -----------------------
var_naive_cond = np.var(v_naive[naive_updates])
var_regime_cond = np.var(v_regime[regime_updates])

# -----------------------
# Metric 3: Drift efficiency
# -----------------------
eff_naive = p_naive[-1] / np.count_nonzero(naive_updates)
eff_regime = p_regime[-1] / max(np.count_nonzero(regime_updates), 1)

# -----------------------
# Metric 4: Regime alignment
# -----------------------
alignment_naive = np.mean(stable_mask[naive_updates])
alignment_regime = np.mean(stable_mask[regime_updates])

# -----------------------
# Print results
# -----------------------
print("\n=== Update Sparsity ===")
print(f"Naive update rate: {update_rate_naive:.3f}")
print(f"Regime-aware update rate: {update_rate_regime:.3f}")

print("\n=== Conditional Velocity Variance ===")
print(f"Naive (when updating): {var_naive_cond:.2f}")
print(f"Regime-aware (when updating): {var_regime_cond:.2f}")
print(f"Stability improvement factor: {var_naive_cond / max(var_regime_cond, 1e-9):.2f}")

print("\n=== Drift Efficiency ===")
print(f"Naive efficiency: {eff_naive:.2f}")
print(f"Regime-aware efficiency: {eff_regime:.2f}")
print(f"Efficiency gain: {eff_regime / eff_naive:.2f}")

print("\n=== Regime Alignment ===")
print(f"Naive alignment with stable regime: {alignment_naive:.3f}")
print(f"Regime-aware alignment with stable regime: {alignment_regime:.3f}")
