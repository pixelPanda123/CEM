import numpy as np
np.random.seed(0)
import pickle
import os

VIDEO_ID = os.environ.get("VIDEO_ID", "2f513ad4ee5e4630")

RESULT_DIR = "results/final"
os.makedirs(RESULT_DIR, exist_ok=True)

# ================================
# Utility
# ================================

def trajectory_to_delta(z):
    return z[1:] - z[:-1]

# ================================
#Temporal Consistency 
# ================================
def compute_total_motion(delta_z, k):
    T = len(delta_z)
    motions = []

    for t in range(0, T - k):
        motion = np.sum(np.linalg.norm(delta_z[t:t+k], axis=1))
        motions.append(motion)

    return np.array(motions)

def compute_metrics_multi_scale(delta_z, k_values=[5, 10, 20, 40]):
    results = {}

    for k in k_values:
        drift = compute_drift(delta_z, k)
        motion = compute_total_motion(delta_z, k)

        efficiency = drift / (motion + 1e-8)

        results[k] = {
            "drift_mean": np.mean(drift),
            "drift_std": np.std(drift),
            "motion_mean": np.mean(motion),
            "efficiency_mean": np.mean(efficiency),
        }

    return results


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

    sm = smoothness(delta_z)
    print(f"Smoothness: {sm:.6f}")

    results = compute_metrics_multi_scale(delta_z)

    print("\nMetrics:")
    for k, stats in results.items():
        print(
            f"k={k} | drift={stats['drift_mean']:.4f} | "
            f"motion={stats['motion_mean']:.4f} | "
            f"eff={stats['efficiency_mean']:.4f}"
        )

    return {
        "smoothness": sm,
        "metrics": results
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
    z_no = np.load(f"results/latent_trajectory/{VIDEO_ID}_z_no.npy")
    z_hard = np.load(f"results/latent_trajectory/{VIDEO_ID}_z_hard.npy")
    z_soft = np.load(f"results/latent_trajectory/{VIDEO_ID}_z_soft.npy")
    # Random gating (apply on RAW motion ideally)
    delta_random = apply_random_gating(trajectory_to_delta(z_no))

    results = {}

    results["no_gating"] = evaluate_trajectory(z_no, "No Gating")
    results["hard_gating"] = evaluate_trajectory(z_hard, "Hard Gating")
    results["soft_gating"] = evaluate_trajectory(z_soft, "Soft Gating")
    results["random_gating"] = evaluate_trajectory(np.cumsum(delta_random, axis=0), "Random Gating")

    # ================================
    # Optical Flow Evaluation
    # ================================

    print("\n\n========== OPTICAL FLOW ==========")

    flow = np.load(f"results/optical_flow/{VIDEO_ID}_flow_vectors.npy")
    delta_flow = flow  # already (T, 2)
    #No Gating 
    delta_no = delta_flow
    #Hard Gating 
    magnitudes = np.linalg.norm(delta_flow, axis=1)
    threshold = np.percentile(magnitudes, 50)

    mask = (magnitudes > threshold).astype(float)
    delta_hard = delta_flow * mask[:, None]

    #Soft Gating 
    posterior = np.load(f"results/regime_modeling/cnn_hmm/{VIDEO_ID}_posterior.npy")
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

    results["flow_no"] =evaluate_trajectory(z_no, "Flow No Gating")
    results["flow_hard"] = evaluate_trajectory(z_hard, "Flow Hard Gating")
    results["flow_soft"] = evaluate_trajectory(z_soft, "Flow Soft Gating")
    results["flow_random"] = evaluate_trajectory(z_random, "Flow Random Gating")

    final_results = {
        "video_id": VIDEO_ID,
        "no_gating": results["no_gating"],
        "hard_gating": results["hard_gating"],
        "soft_gating": results["soft_gating"],
        "random_gating": results["random_gating"],
        "flow_no_gating": results["flow_no"],
        "flow_hard_gating": results["flow_hard"],
        "flow_soft_gating": results["flow_soft"],
        "flow_random_gating": results["flow_random"],
    }
    save_path = os.path.join(RESULT_DIR, f"{VIDEO_ID}_metrics.npy")
    np.save(save_path, final_results)

    print(f"\nSaved results to: {save_path}")