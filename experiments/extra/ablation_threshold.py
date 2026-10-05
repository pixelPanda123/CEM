"""
ablation_threshold.py
----------------------
Hard-gating threshold sweep. Shows that soft gating beats the BEST hard
threshold, not just the single tau=0.5 reported in the paper. Reuses the
CNN latent trajectory (z_no) and the HMM posterior already on disk, so no
frames / no re-embedding is needed.

For each tau in {0.3,...,0.7}:  r_t = 1[alpha_t > tau];  z = cumsum(r_t * dz)
Reports drift and retained motion at k=20, aggregated across videos.

Run from project root:
    python3 experiments/extra/ablation_threshold.py
"""

import os
import numpy as np

VIDEOS = ["4c059b716bf84ea3", "4ed494e20fd349ad", "2f513ad4ee5e4630", "4ab482cc46054935"]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
HEADLINE_K = 20
OUT_DIR = "results/tables"
os.makedirs(OUT_DIR, exist_ok=True)

TRAJ_DIR = "results/latent_trajectory"
POST_DIR = "results/regime_modeling/cnn_hmm"


# ---- identical formulas to experiments/metrics/temporal_consistency.py ----
def trajectory_to_delta(z):
    return z[1:] - z[:-1]

def compute_drift(delta_z, k):
    T = len(delta_z)
    return np.array([np.linalg.norm(np.sum(delta_z[t:t + k], axis=0))
                     for t in range(0, T - k)])

def compute_total_motion(delta_z, k):
    T = len(delta_z)
    return np.array([np.sum(np.linalg.norm(delta_z[t:t + k], axis=1))
                     for t in range(0, T - k)])
# ---------------------------------------------------------------------------


def load_video(vid):
    z_no = np.load(f"{TRAJ_DIR}/{vid}_z_no.npy")      # (M,2)
    alpha = np.load(f"{POST_DIR}/{vid}_posterior.npy")  # (N,)
    dz = trajectory_to_delta(z_no)                     # raw CNN increments (M-1,2)
    T = min(len(dz), len(alpha))
    return dz[:T], alpha[:T]


def sweep():
    drift = {tau: [] for tau in THRESHOLDS}
    motion = {tau: [] for tau in THRESHOLDS}
    soft_drift, soft_motion = [], []
    ref_motion = []            # true no-gating motion per video (=100% retained)

    used = 0
    for vid in VIDEOS:
        try:
            dz, alpha = load_video(vid)
        except FileNotFoundError as e:
            print(f"[skip] {vid}: {e}")
            continue
        used += 1

        ref_motion.append(np.mean(compute_total_motion(dz, HEADLINE_K)))

        z_soft = np.cumsum(alpha[:, None] * dz, axis=0)
        dsz = trajectory_to_delta(z_soft)
        soft_drift.append(np.mean(compute_drift(dsz, HEADLINE_K)))
        soft_motion.append(np.mean(compute_total_motion(dsz, HEADLINE_K)))

        for tau in THRESHOLDS:
            r = (alpha > tau).astype(float)
            z_hard = np.cumsum(r[:, None] * dz, axis=0)
            dhz = trajectory_to_delta(z_hard)
            drift[tau].append(np.mean(compute_drift(dhz, HEADLINE_K)))
            motion[tau].append(np.mean(compute_total_motion(dhz, HEADLINE_K)))

    if used == 0:
        raise SystemExit("No videos found. Check results/latent_trajectory and posteriors.")
    return drift, motion, soft_drift, soft_motion, float(np.mean(ref_motion))


def build_table(drift, motion, soft_drift, soft_motion, ref):
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Hard-gating threshold sweep vs.\ soft gating at $k=20$ (CNN motion, "
        r"mean over videos). Soft gating is reported for reference; it avoids the "
        r"need to choose a threshold.}",
        r"\centering",
        r"\begin{tabular}{|l|c|c|}",
        r"\hline",
        r"\textbf{Gating} & \textbf{Drift $\downarrow$} & \textbf{Motion kept} \\",
        r"\hline",
    ]
    best_tau = min(THRESHOLDS, key=lambda t: np.mean(drift[t]))
    for tau in THRESHOLDS:
        d = np.mean(drift[tau]); mo = np.mean(motion[tau])
        kept = 100.0 * mo / ref if ref else float("nan")
        tag = f"Hard ($\\tau={tau}$)"
        cell = f"{tag} & {d:.2f} & {kept:.0f}\\%"
        if tau == best_tau:
            cell = f"{tag} & \\underline{{{d:.2f}}} & {kept:.0f}\\%"
        lines.append(cell + r" \\")
    lines.append(r"\hline")
    sd = np.mean(soft_drift); sm = np.mean(soft_motion)
    kept = 100.0 * sm / ref if ref else float("nan")
    lines.append(r"\textbf{Soft (Ours)} & \textbf{" + f"{sd:.2f}" + r"} & " + f"{kept:.0f}\\%" + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\label{tab:threshold_sweep}", r"\end{table}"]
    return "\n".join(lines), best_tau, np.mean(drift[best_tau]), sd


if __name__ == "__main__":
    drift, motion, soft_drift, soft_motion, ref = sweep()
    tex, best_tau, best_hard, soft = build_table(drift, motion, soft_drift, soft_motion, ref)
    path = os.path.join(OUT_DIR, "tab_threshold_sweep.tex")
    with open(path, "w") as f:
        f.write(tex + "\n")
    print(tex)
    print(f"\nBest hard threshold: tau={best_tau} -> drift {best_hard:.3f}")
    print(f"Soft gating         -> drift {soft:.3f}")
    verdict = "beats" if soft < best_hard else "does NOT beat"
    print(f"Soft {verdict} the best hard threshold.")
    print("Saved", path)