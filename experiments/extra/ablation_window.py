"""
ablation_window.py
-------------------
Ablation over the TMD window size W (paper fixes W=5 with no justification).
For each W it recomputes the TMD, retrains the HMM, and re-integrates the soft
trajectory, reporting drift / retained motion at k=20 averaged over videos.

Reuses the project's own GaussianHMM so the model is identical to the main run.
CNN motion increments (PCA-2D) are recomputed from the embedding vectors, as in
experiments/pose_proxy/regime_aware_latent_trajectory.py.

Run from project root:
    python3 experiments/extra/ablation_window.py
"""

import os, sys, pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# make the project's HMM importable regardless of CWD
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "HMM"))
from hmm_model import GaussianHMM            # noqa: E402

VIDEOS = ["4c059b716bf84ea3", "4ed494e20fd349ad", "2f513ad4ee5e4630", "4ab482cc46054935"]
WINDOWS = [3, 5, 7, 10]
HEADLINE_K = 20
N_ITER = 25
SEED = 0
OUT_DIR = "results/tables"
os.makedirs(OUT_DIR, exist_ok=True)

EMB_DIR = "results/cnn_embedding"


def trajectory_to_delta(z): return z[1:] - z[:-1]

def compute_drift(dz, k):
    T = len(dz); return np.array([np.linalg.norm(np.sum(dz[t:t+k], axis=0)) for t in range(0, T-k)])

def compute_total_motion(dz, k):
    T = len(dz); return np.array([np.sum(np.linalg.norm(dz[t:t+k], axis=1)) for t in range(0, T-k)])


def tmd_from_motion(values, W):
    feats = []
    for i in range(len(values) - W + 1):
        w = values[i:i+W]
        feats.append([np.mean(w), np.var(w), np.mean(np.abs(np.diff(w)))])
    return np.array(feats)


def posterior_for_W(values, W):
    X = tmd_from_motion(values, W)
    X = StandardScaler().fit_transform(X)
    np.random.seed(SEED)                      # GaussianHMM.initialize uses np.random
    hmm = GaussianHMM(n_states=2)
    gamma = hmm.fit(X, n_iter=N_ITER)
    stable = int(np.argmin(hmm.means[:, 1]))  # lowest variance-feature mean = stable
    return gamma[:, stable]


def load_cnn(vid):
    with open(f"{EMB_DIR}/{vid}_embedding_vectors.pkl", "rb") as f:
        vecs = pickle.load(f)
    with open(f"{EMB_DIR}/{vid}_embedding_motion.pkl", "rb") as f:
        motion = pickle.load(f)
    t = np.array(sorted(vecs.keys()))
    E = np.array([vecs[i] for i in t])
    dz = PCA(n_components=2).fit_transform(E[1:] - E[:-1])      # (F-1, 2)
    mt = np.array([motion[i] for i in sorted(motion.keys())])   # (F-1,)
    return dz, mt


def run():
    drift = {W: [] for W in WINDOWS}
    motion = {W: [] for W in WINDOWS}
    ref_motion = []
    used = 0
    for vid in VIDEOS:
        try:
            dz, mt = load_cnn(vid)
        except FileNotFoundError as e:
            print(f"[skip] {vid}: {e}")
            continue
        used += 1
        ref_motion.append(np.mean(compute_total_motion(dz, HEADLINE_K)))
        for W in WINDOWS:
            alpha = posterior_for_W(mt, W)
            T = min(len(dz), len(alpha))
            z_soft = np.cumsum(alpha[:T, None] * dz[:T], axis=0)
            dsz = trajectory_to_delta(z_soft)
            drift[W].append(np.mean(compute_drift(dsz, HEADLINE_K)))
            motion[W].append(np.mean(compute_total_motion(dsz, HEADLINE_K)))
            print(f"{vid[:8]} W={W:2d} drift={drift[W][-1]:.3f}")
    if used == 0:
        raise SystemExit("No embedding vectors found in results/cnn_embedding.")
    return drift, motion, float(np.mean(ref_motion))


def table(drift, motion, ref):
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Effect of TMD window size $W$ on soft gating at $k=20$ "
        r"(mean over videos). $W=5$ is the setting used in the main results.}",
        r"\centering",
        r"\begin{tabular}{|c|c|c|}",
        r"\hline",
        r"\textbf{Window $W$} & \textbf{Drift $\downarrow$} & \textbf{Motion kept} \\",
        r"\hline",
    ]
    best = min(WINDOWS, key=lambda W: np.mean(drift[W]))
    for W in WINDOWS:
        d = np.mean(drift[W]); kept = 100.0 * np.mean(motion[W]) / ref
        cell = f"{W} & {d:.2f} & {kept:.0f}\\%"
        if W == best:
            cell = f"{W} & \\textbf{{{d:.2f}}} & {kept:.0f}\\%"
        lines.append(cell + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\label{tab:window_ablation}", r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    drift, motion, ref = run()
    tex = table(drift, motion, ref)
    with open(f"{OUT_DIR}/tab_window_ablation.tex", "w") as f:
        f.write(tex + "\n")
    print("\n" + tex)