"""
aggregate_full.py
-----------------
Reads every results/final/<VIDEO_ID>_metrics.npy produced by
experiments/metrics/temporal_consistency.py and emits the FULL result grid
that the paper currently promises but never reports:

  (1) Drift vs k, all methods, mean +/- std across videos     -> Table (CNN + Flow)
  (2) Motion preservation / efficiency / smoothness at k=20    -> Table  <-- the missing defence
  (3) Per-video drift breakdown at k=20                        -> Table

Nothing is re-run: this only aggregates numbers already saved on disk.
LaTeX is written to results/tables/*.tex and also printed.

Run from project root:
    python3 experiments/extra/aggregate_full.py
"""

import os
import numpy as np

RESULT_DIR = "results/final"
OUT_DIR = "results/tables"
os.makedirs(OUT_DIR, exist_ok=True)

K_VALUES = [5, 10, 20, 40]
HEADLINE_K = 20

CNN_METHODS = ["no_gating", "hard_gating", "soft_gating", "random_gating"]
FLOW_METHODS = ["flow_no_gating", "flow_hard_gating", "flow_soft_gating", "flow_random_gating"]
PRETTY = {
    "no_gating": "No gating",
    "hard_gating": "Hard gating",
    "soft_gating": "Soft gating (Ours)",
    "random_gating": "Random gating",
    "flow_no_gating": "No gating",
    "flow_hard_gating": "Hard gating",
    "flow_soft_gating": "Soft gating (Ours)",
    "flow_random_gating": "Random gating",
}


# ------------------------------------------------------------------
# Load every per-video metrics file
# ------------------------------------------------------------------
def load_all():
    records = {}
    for f in sorted(os.listdir(RESULT_DIR)):
        if not f.endswith("_metrics.npy"):
            continue
        data = np.load(os.path.join(RESULT_DIR, f), allow_pickle=True).item()
        vid = data.get("video_id", f.replace("_metrics.npy", ""))
        records[vid] = data
    if not records:
        raise SystemExit(f"No *_metrics.npy found in {RESULT_DIR}. Run the metrics stage first.")
    print(f"Loaded {len(records)} videos: {list(records.keys())}")
    return records


def collect(records, method, field, k=None):
    """Return a per-video array of one field for one method (optionally at scale k)."""
    out = []
    for vid, data in records.items():
        if method not in data:
            continue
        entry = data[method]
        if k is None:
            out.append(entry[field])                  # scalar field e.g. smoothness
        else:
            out.append(entry["metrics"][k][field])    # scale-dependent field
    return np.array(out, dtype=float)


def ms(arr):
    """mean, std helper that tolerates empties."""
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


# ------------------------------------------------------------------
# Table 1: drift vs k (mean +/- std across videos)
# ------------------------------------------------------------------
def table_drift_vs_k(records, methods, caption, label):
    header_k = " & ".join([f"$k={k}$" for k in K_VALUES])
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{" + caption + r"}",
        r"\centering",
        r"\begin{tabular}{|l|" + "c|" * len(K_VALUES) + r"}",
        r"\hline",
        r"\textbf{Method} & " + header_k + r" \\",
        r"\hline",
    ]
    for m in methods:
        cells = []
        for k in K_VALUES:
            mean, std = ms(collect(records, m, "drift_mean", k))
            txt = f"{mean:.2f} $\\pm$ {std:.2f}"
            if m.endswith("soft_gating"):
                txt = r"\textbf{" + txt + r"}"
            cells.append(txt)
        lines.append(PRETTY[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\label{" + label + r"}", r"\end{table}"]
    return "\n".join(lines)


# ------------------------------------------------------------------
# Table 2: the missing defence -- drift vs motion retained vs smoothness
# ------------------------------------------------------------------
def table_tradeoff(records, methods, caption, label):
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{" + caption + r"}",
        r"\centering",
        r"\begin{tabular}{|l|c|c|c|}",
        r"\hline",
        r"\textbf{Method} & \textbf{Drift $\downarrow$} & \textbf{Motion kept} & "
        r"\textbf{Smoothness $\downarrow$} \\",
        r"\hline",
    ]
    ref_key = methods[0]
    ref_motion, _ = ms(collect(records, ref_key, "motion_mean", HEADLINE_K))
    for m in methods:
        d_mean, d_std = ms(collect(records, m, "drift_mean", HEADLINE_K))
        mo_mean, _ = ms(collect(records, m, "motion_mean", HEADLINE_K))
        sm_mean, _ = ms(collect(records, m, "smoothness"))
        kept = 100.0 * mo_mean / ref_motion if ref_motion else float("nan")
        drift_txt = f"{d_mean:.2f} $\\pm$ {d_std:.2f}"
        name = r"\textbf{" + PRETTY[m] + r"}" if m.endswith("soft_gating") else PRETTY[m]
        lines.append(f"{name} & {drift_txt} & {kept:.0f}\\% & {sm_mean:.4f}" + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\label{" + label + r"}", r"\end{table}"]
    return "\n".join(lines)


# ------------------------------------------------------------------
# Table 3: per-video drift at headline k
# ------------------------------------------------------------------
def table_per_video(records, methods, caption, label):
    vids = list(records.keys())
    short = [v[:8] for v in vids]
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{" + caption + r"}",
        r"\centering",
        r"\begin{tabular}{|l|" + "c|" * len(vids) + r"}",
        r"\hline",
        r"\textbf{Method} & " + " & ".join(short) + r" \\",
        r"\hline",
    ]
    for m in methods:
        cells = []
        for vid in vids:
            try:
                cells.append(f"{records[vid][m]['metrics'][HEADLINE_K]['drift_mean']:.2f}")
            except KeyError:
                cells.append("--")
        lines.append(PRETTY[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\label{" + label + r"}", r"\end{table}"]
    return "\n".join(lines)


# ------------------------------------------------------------------
# Sanity check for the known flow-path bug in temporal_consistency.py
# ------------------------------------------------------------------
def warn_if_flow_invariant(records):
    d20 = collect(records, "flow_no_gating", "drift_mean", HEADLINE_K)
    if d20.size >= 2 and np.std(d20) < 1e-9:
        print("\n[WARNING] Flow drift is IDENTICAL across all videos "
              "(std=0 at k=20).")
        print("          temporal_consistency.py line ~144 loads "
              "'results/optical_flow/flow_vectors.npy' WITHOUT the VIDEO_ID "
              "prefix, so every video reuses one flow file.")
        print("          Fix: flow = np.load(f'results/optical_flow/"
              "{VIDEO_ID}_flow_vectors.npy')  then re-run the metrics stage.\n")


# ------------------------------------------------------------------
def write(name, tex):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        f.write(tex + "\n")
    print(f"\n% ---- {name} ----\n{tex}\n")
    return path


if __name__ == "__main__":
    records = load_all()
    warn_if_flow_invariant(records)

    write("tab_cnn_drift_vs_k.tex",
          table_drift_vs_k(records, CNN_METHODS,
                           "Drift across temporal scales (CNN motion). Mean $\\pm$ std over videos.",
                           "tab:cnn_drift_k"))
    write("tab_flow_drift_vs_k.tex",
          table_drift_vs_k(records, FLOW_METHODS,
                           "Drift across temporal scales (optical flow). Mean $\\pm$ std over videos.",
                           "tab:flow_drift_k"))
    write("tab_cnn_tradeoff.tex",
          table_tradeoff(records, CNN_METHODS,
                         "Drift vs.\\ motion preservation at $k=20$ (CNN). "
                         "Lower drift with higher retained motion is the goal; "
                         "Drift/Motion normalises away trivial suppression.",
                         "tab:cnn_tradeoff"))
    write("tab_flow_tradeoff.tex",
          table_tradeoff(records, FLOW_METHODS,
                         "Drift vs.\\ motion preservation at $k=20$ (optical flow).",
                         "tab:flow_tradeoff"))
    write("tab_per_video.tex",
          table_per_video(records, CNN_METHODS,
                          "Per-video drift at $k=20$ (CNN motion).",
                          "tab:per_video"))
    print("All tables written to", OUT_DIR)