"""
validate_regimes.py
-------------------
External validation of the learned motion regimes.

The HMM never sees image content -- only motion descriptors. If its posterior
alpha_t (probability of the stable regime) genuinely tracks reliability, it
should correlate with INDEPENDENT image-degradation signals (sharpness, glare,
darkness) computed in frame_quality.py.

Alignment: frame signals (len F) -> per motion-step (len F-1, averaging the two
frames of each step) -> window-averaged with W=5 (len F-5 = N), mirroring how
the TMD / posterior are built. alpha_t is then compared window-for-window.

Reports, per video and pooled:
  * Spearman & Pearson corr(alpha, sharpness / glare / darkness)
  * ROC-AUC of (1 - alpha) predicting "bad" windows (bottom-quartile sharpness)
Emits results/tables/tab_regime_validation.tex and a figure per video.

Run from project root (after frame_quality.py + the HMM stage):
    python3 experiments/extra/validate_regimes.py
"""

import os
import numpy as np

try:
    from scipy.stats import spearmanr, pearsonr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIDEOS = ["4c059b716bf84ea3", "4ed494e20fd349ad", "2f513ad4ee5e4630", "4ab482cc46054935"]
W = 5                         # must match compute_tmd.WINDOW_SIZE
BAD_PCTILE = 25               # bottom-quartile sharpness = "bad" window
QUAL_DIR = "results/frame_quality"
POST_DIR = "results/regime_modeling/cnn_hmm"
OUT_DIR = "results/tables"
FIG_DIR = "results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def window_avg(x, w):
    """Average over sliding windows of length w -> len(x)-w+1 (matches TMD)."""
    return np.array([np.mean(x[i:i + w]) for i in range(len(x) - w + 1)])


def to_window_signal(per_frame, w):
    per_step = 0.5 * (per_frame[1:] + per_frame[:-1])   # F -> F-1
    return window_avg(per_step, w)                       # F-1 -> F-1-w+1 = F-w


def roc_auc(score, positive):
    """AUC via Mann-Whitney U, no sklearn dependency. positive: bool array."""
    pos = score[positive]; neg = score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    r_pos = ranks[:len(pos)].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return auc


def corr(a, b):
    if HAVE_SCIPY:
        return spearmanr(a, b).correlation, pearsonr(a, b)[0]
    def pear(x, y):
        x = x - x.mean(); y = y - y.mean()
        return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
    rs = pear(a.argsort().argsort().astype(float), b.argsort().argsort().astype(float))
    return rs, pear(a, b)


def load_video(vid):
    q = np.load(f"{QUAL_DIR}/{vid}_quality.npy", allow_pickle=True).item()
    alpha = np.load(f"{POST_DIR}/{vid}_posterior.npy")
    sig = {k: to_window_signal(q[k], W) for k in ("sharpness", "glare", "darkness")}
    T = min([len(alpha)] + [len(v) for v in sig.values()])
    alpha = alpha[:T]
    sig = {k: v[:T] for k, v in sig.items()}
    return alpha, sig


def plot(vid, alpha, sig):
    bad = sig["sharpness"] <= np.percentile(sig["sharpness"], BAD_PCTILE)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(alpha, label=r"$\alpha_t$ (stable prob.)", lw=2)
    s = sig["sharpness"]; s = (s - s.min()) / (np.ptp(s) + 1e-9)
    ax.plot(s, label="sharpness (norm.)", alpha=0.6)
    ax.fill_between(np.arange(len(alpha)), 0, 1, where=bad, color="red", alpha=0.12,
                    label="bad (low-sharpness) windows")
    ax.set_xlabel("window index"); ax.set_ylim(0, 1); ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Regime posterior vs image sharpness — {vid[:8]}")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/{vid}_regime_validation.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    rows = []
    pooled = {"alpha": [], "sharpness": [], "glare": [], "darkness": []}
    for vid in VIDEOS:
        try:
            alpha, sig = load_video(vid)
        except FileNotFoundError as e:
            print(f"[skip] {vid}: {e}")
            continue
        rs_sharp, _ = corr(alpha, sig["sharpness"])
        rs_glare, _ = corr(alpha, sig["glare"])
        rs_dark, _ = corr(alpha, sig["darkness"])
        bad = sig["sharpness"] <= np.percentile(sig["sharpness"], BAD_PCTILE)
        auc = roc_auc(1.0 - alpha, bad)
        rows.append((vid[:8], rs_sharp, rs_glare, rs_dark, auc))
        print(f"{vid[:8]} | rho(sharp)={rs_sharp:+.3f} rho(glare)={rs_glare:+.3f} "
              f"rho(dark)={rs_dark:+.3f} | AUC={auc:.3f}")
        for k in ("sharpness", "glare", "darkness"):
            pooled[k].append(sig[k])
        pooled["alpha"].append(alpha)
        plot(vid, alpha, sig)

    if not rows:
        raise SystemExit("Nothing to validate. Run frame_quality.py and the HMM stage first.")

    A = np.concatenate(pooled["alpha"])
    pooled_rs = {k: corr(A, np.concatenate(pooled[k]))[0] for k in ("sharpness", "glare", "darkness")}
    bad = np.concatenate([s <= np.percentile(s, BAD_PCTILE)
                          for s in [np.concatenate(pooled["sharpness"])]])
    pooled_auc = roc_auc(1.0 - A, bad)

    tex = [
        r"\begin{table}[htbp]",
        r"\caption{External validation of learned regimes. Spearman $\rho$ between the "
        r"regime posterior $\alpha_t$ and label-free image signals, and ROC-AUC of "
        r"$1-\alpha_t$ predicting low-sharpness windows. The HMM sees only motion, never pixels.}",
        r"\centering",
        r"\begin{tabular}{|l|c|c|c|c|}",
        r"\hline",
        r"\textbf{Video} & $\rho$(sharp) & $\rho$(glare) & $\rho$(dark) & \textbf{AUC} \\",
        r"\hline",
    ]
    for v, rs, rg, rd, auc in rows:
        tex.append(f"{v} & {rs:+.2f} & {rg:+.2f} & {rd:+.2f} & {auc:.2f} " + r"\\")
    tex.append(r"\hline")
    tex.append(f"\\textbf{{Pooled}} & {pooled_rs['sharpness']:+.2f} & "
               f"{pooled_rs['glare']:+.2f} & {pooled_rs['darkness']:+.2f} & "
               f"{pooled_auc:.2f} " + r"\\")
    tex += [r"\hline", r"\end{tabular}", r"\label{tab:regime_validation}", r"\end{table}"]
    tex = "\n".join(tex)
    with open(f"{OUT_DIR}/tab_regime_validation.tex", "w") as f:
        f.write(tex + "\n")
    print("\n" + tex)
    print(f"\nFigures -> {FIG_DIR}/<vid>_regime_validation.png")
    if not HAVE_SCIPY:
        print("[scipy missing: `pip install scipy` for exact Spearman/Pearson]")