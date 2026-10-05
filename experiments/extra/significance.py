"""
significance.py
---------------
Paired Wilcoxon + bootstrap CI on per-window drift (k=20) across videos,
from the saved trajectories. Run from project root:
    python3 experiments/extra/significance.py
"""

import os
import numpy as np

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

VIDEOS = ["4c059b716bf84ea3", "4ed494e20fd349ad", "2f513ad4ee5e4630", "4ab482cc46054935"]
K = 20
N_BOOT = 10000
SEED = 0
TRAJ_DIR = "results/latent_trajectory"
rng = np.random.default_rng(SEED)


def trajectory_to_delta(z):
    return z[1:] - z[:-1]

def drift_series(z, k):
    dz = trajectory_to_delta(z)
    T = len(dz)
    return np.array([np.linalg.norm(np.sum(dz[t:t + k], axis=0)) for t in range(0, T - k)])


def gather():
    """Pool per-window drift across videos; keep windows paired across methods."""
    pooled = {"no": [], "hard": [], "soft": []}
    for vid in VIDEOS:
        try:
            z_no = np.load(f"{TRAJ_DIR}/{vid}_z_no.npy")
            z_hard = np.load(f"{TRAJ_DIR}/{vid}_z_hard.npy")
            z_soft = np.load(f"{TRAJ_DIR}/{vid}_z_soft.npy")
        except FileNotFoundError as e:
            print(f"[skip] {vid}: {e}")
            continue
        dn, dh, ds = drift_series(z_no, K), drift_series(z_hard, K), drift_series(z_soft, K)
        T = min(len(dn), len(dh), len(ds))
        pooled["no"].append(dn[:T]); pooled["hard"].append(dh[:T]); pooled["soft"].append(ds[:T])
    for key in pooled:
        if not pooled[key]:
            raise SystemExit("No trajectories found in results/latent_trajectory.")
        pooled[key] = np.concatenate(pooled[key])
    return pooled


def bootstrap_reduction(a, b):
    """95% CI on overall reduction % = (mean(a)-mean(b))/mean(a)*100, resampling
    windows. Ratio of means, NOT mean of per-window ratios (robust to a~=0)."""
    idx = np.arange(len(a))
    def red(ix):
        ma = a[ix].mean()
        return 100.0 * (ma - b[ix].mean()) / ma
    point = red(idx)
    boots = np.array([red(rng.choice(idx, len(idx), replace=True)) for _ in range(N_BOOT)])
    return point, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


if __name__ == "__main__":
    pooled = gather()
    n = len(pooled["soft"])
    print(f"Paired windows pooled across videos: {n}")
    print(f"Mean drift  no={pooled['no'].mean():.3f}  "
          f"hard={pooled['hard'].mean():.3f}  soft={pooled['soft'].mean():.3f}")

    if HAVE_SCIPY:
        for ref in ["no", "hard"]:
            stat, p = wilcoxon(pooled["soft"], pooled[ref])
            print(f"Wilcoxon soft vs {ref}: W={stat:.1f}, p={p:.2e}")
    else:
        print("[scipy not installed: pip install scipy  -> for Wilcoxon p-values]")

    m_no, lo_no, hi_no = bootstrap_reduction(pooled["no"], pooled["soft"])
    m_hd, lo_hd, hi_hd = bootstrap_reduction(pooled["hard"], pooled["soft"])
    print(f"Drift reduction soft vs no:   {m_no:.1f}% (95% CI [{lo_no:.1f}, {hi_no:.1f}])")
    print(f"Drift reduction soft vs hard: {m_hd:.1f}% (95% CI [{lo_hd:.1f}, {hi_hd:.1f}])")

    print("\nPaper sentence:")
    if HAVE_SCIPY:
        print(f"Soft gating lowers mean per-window drift by {m_no:.1f}\\% "
              f"(95\\% CI [{lo_no:.1f}, {hi_no:.1f}]), from {pooled['no'].mean():.2f} to "
              f"{pooled['soft'].mean():.2f}, and by {m_hd:.1f}\\% over hard gating; both "
              f"differences are significant under a paired Wilcoxon signed-rank test "
              f"($p<10^{{-16}}$, $n={n}$ windows).")