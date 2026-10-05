import numpy as np
import os

RESULT_DIR = "results/final"
K = 20  # we report k=40 in paper

methods = ["no_gating", "hard_gating", "soft_gating", "random_gating"]

cnn_stats = {m: [] for m in methods}
flow_stats = {m: [] for m in methods}

# ================================
# Load all results
# ================================
for file in os.listdir(RESULT_DIR):
    if not file.endswith("_metrics.npy"):
        continue

    path = os.path.join(RESULT_DIR, file)
    data = np.load(path, allow_pickle=True).item()

    print(f"\nProcessing: {file}")

    # CNN
    for m in methods:
        cnn_stats[m].append(data[m]["metrics"][K]["drift_mean"])

    # FLOW
    for m in methods:
        flow_stats[m].append(data["flow_" + m]["metrics"][K]["drift_mean"])


# ================================
# Print aggregated results
# ================================
def print_stats(title, stats):
    print(f"\n===== {title} (k=20) =====")
    for m in methods:
        values = np.array(stats[m])
        print(f"{m:12s} | mean={values.mean():.3f} | std={values.std():.3f}")


print_stats("CNN Drift", cnn_stats)
print_stats("Flow Drift", flow_stats)