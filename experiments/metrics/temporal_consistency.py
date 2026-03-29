import numpy as np
import pickle

# ================================
# Utility
# ================================

def trajectory_to_delta(z):
    return z[1:] - z[:-1]


# ================================
# Metric 1: Smoothness (your old TCS)
# ================================  

def smoothness(signal):
    diffs = np.abs(np.diff(signal, axis=0))
    return np.mean(diffs)


# ================================
# Metric 2: Drift Accumulation
# ================================

def compute_drift(delta_z, k):
    T = len(delta_z)
    drifts = []

    for t in range(0, T - k):
        displacement = np.sum(delta_z[t:t+k], axis=0)
        drift = np.linalg.norm(displacement)
        drifts.append(drift)

    return np.array(drifts)


def compute_drift_multi_scale(delta_z, k_values=[5, 10, 20, 40]):
    results = {}

    for k in k_values:
        drift = compute_drift(delta_z, k)

        results[k] = {
            "mean": np.mean(drift),
            "median": np.median(drift),
            "std": np.std(drift),
        }

    return results

# ================================
# Evaluation Runner
# ================================

def evaluate_trajectory(z, name="method"):
    delta_z = trajectory_to_delta(z)

    print(f"\n===== {name} =====")

    # Smoothness
    sm = smoothness(delta_z)
    print(f"Smoothness: {sm:.6f}")

    # Drift accumulation
    drift_results = compute_drift_multi_scale(delta_z)

    print("\nCDrift accumulation:")
    for k, stats in drift_results.items():
        print(f"k={k} | mean={stats['mean']:.6f} | std={stats['std']:.6f}")

    return {
        "smoothness": sm,
        "cce": drift_results
    }

#Random gating (For trajectory checking) 
def apply_random_gating(delta_z):
    alpha_random = np.random.uniform(0, 1, size=len(delta_z))
    return alpha_random[:, None] * delta_z
# ================================
# Example Usage
# ================================

if __name__ == "__main__":
    # Load trajectories
    z_no = np.load("results/latent_trajectory/z_no.npy")
    z_hard = np.load("results/latent_trajectory/z_hard.npy")
    z_soft = np.load("results/latent_trajectory/z_soft.npy")
    # Random gating (apply on RAW motion ideally)
    delta_random = apply_random_gating(trajectory_to_delta(z_no))

    results = {}

    results["no_gating"] = evaluate_trajectory(z_no, "No Gating")
    results["hard_gating"] = evaluate_trajectory(z_hard, "Hard Gating")
    results["soft_gating"] = evaluate_trajectory(z_soft, "Soft Gating")
    results["Random Gating"] = evaluate_trajectory(np.cumsum(delta_random, axis=0), "Random Gating")

    # ================================
    # Optical Flow Evaluation
    # ================================

    print("\n\n========== OPTICAL FLOW ==========")

    flow = np.load("results/optical_flow/flow_vectors.npy")
    delta_flow = flow  # already (T, 2)
    #No Gating 
    delta_no = delta_flow
    #Hard Gating 
    magnitudes = np.linalg.norm(delta_flow, axis=1)
    threshold = np.percentile(magnitudes, 50)

    mask = (magnitudes > threshold).astype(float)
    delta_hard = delta_flow * mask[:, None]

    #Soft Gating 
    posterior = np.load("results/regime_modeling/cnn_hmm/posterior.npy")
    print("Flow shape:", delta_flow.shape)
    print("Posterior shape:", posterior.shape)
    T = min(len(delta_flow), len(posterior))
    delta_flow_aligned = delta_flow[:T]
    alpha_aligned = posterior[:T]
    delta_soft = delta_flow_aligned * alpha_aligned[:, None]

    #Random Gating 
    alpha_random = np.random.uniform(0, 1, size=len(delta_flow))
    delta_random = delta_flow * alpha_random[:, None]

    def integrate(delta):
        return np.cumsum(delta, axis=0)

    z_no = integrate(delta_no)
    z_hard = integrate(delta_hard)
    z_soft = integrate(delta_soft)
    z_random = integrate(delta_random)

    evaluate_trajectory(z_no, "Flow No Gating")
    evaluate_trajectory(z_hard, "Flow Hard Gating")
    evaluate_trajectory(z_soft, "Flow Soft Gating")
    evaluate_trajectory(z_random, "Flow Random Gating")