"""
runtime_benchmark.py
--------------------
Quantifies the "lightweight" claim: wall-clock cost per stage. Times each stage
on one video (default 60 frames) and reports milliseconds per frame plus the
one-off cost of regime modeling. CPU timings unless CUDA is present.

Stages: ResNet-18 embedding, Farneback optical flow, TMD, HMM fit, integration.
Each stage is guarded, so a missing dependency degrades gracefully to "n/a".

Run from project root:
    VIDEO_ID=4c059b716bf84ea3 N_FRAMES=60 python3 experiments/extra/runtime_benchmark.py
"""

import os, sys, glob, time, pickle
import numpy as np

VIDEO_ID = os.environ.get("VIDEO_ID", "2f513ad4ee5e4630")
N_FRAMES = int(os.environ.get("N_FRAMES", "60"))
N_ITER = 25
FRAME_DIR = f"Datasets/kvasir-capsule/frames/{VIDEO_ID}"
OUT_DIR = "results/tables"
os.makedirs(OUT_DIR, exist_ok=True)

rows = []   # (stage, time_string, note)

def add(stage, ms, note, per_frame=False):
    if ms is None:
        rows.append((stage, "n/a", note)); return
    unit = "ms/frame" if per_frame else "ms"
    tstr = f"{ms:.2f} {unit}" if ms >= 1 else f"{ms:.3f} {unit}"
    rows.append((stage, tstr, note))


def frames():
    p = sorted(glob.glob(f"{FRAME_DIR}/frame_*.jpg"))[:N_FRAMES]
    if not p:
        raise SystemExit(f"No frames in {FRAME_DIR}")
    return p


def time_cnn(paths):
    try:
        import torch, cv2
        import torchvision.transforms as T
        import torchvision.models as models
    except Exception as e:
        add("ResNet-18 embedding", None, f"skipped ({e.__class__.__name__})"); return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); m.fc = torch.nn.Identity()
    m = m.to(dev).eval()
    tf = T.Compose([T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():                    # warmup
        x = tf(cv2.cvtColor(cv2.imread(paths[0]), cv2.COLOR_BGR2RGB)).unsqueeze(0).to(dev); m(x)
    t0 = time.perf_counter()
    with torch.no_grad():
        for p in paths:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            m(tf(img).unsqueeze(0).to(dev))
    add("ResNet-18 embedding", 1e3 * (time.perf_counter() - t0) / len(paths), str(dev), per_frame=True)


def time_flow(paths):
    try:
        import cv2
    except Exception as e:
        add("Farneback optical flow", None, f"skipped ({e.__class__.__name__})"); return
    prev = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
    t0 = time.perf_counter()
    for p in paths[1:]:
        curr = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev = curr
    add("Farneback optical flow", 1e3 * (time.perf_counter() - t0) / (len(paths) - 1), "CPU", per_frame=True)


def time_regime(n):
    from sklearn.preprocessing import StandardScaler
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "HMM"))
    vals = np.abs(np.random.default_rng(0).normal(size=n))
    t0 = time.perf_counter()
    W = 5
    tmd = np.array([[np.mean(vals[i:i+W]), np.var(vals[i:i+W]),
                     np.mean(np.abs(np.diff(vals[i:i+W])))] for i in range(len(vals)-W+1)])
    add("TMD (W=5)", 1e3 * (time.perf_counter() - t0), f"ms total, T={len(tmd)}")
    X = StandardScaler().fit_transform(tmd)
    try:
        from hmm_model import GaussianHMM
        np.random.seed(0)
        t0 = time.perf_counter()
        GaussianHMM(n_states=2).fit(X, n_iter=N_ITER)
        add("HMM fit (full video)", 1e3 * (time.perf_counter() - t0), f"ms total, {N_ITER} iters, T={len(X)}")
    except Exception as e:
        add("HMM fit (full video)", None, f"skipped ({e.__class__.__name__})")
    dz = np.random.default_rng(1).normal(size=(len(X), 2)); alpha = np.random.default_rng(2).random(len(X))
    t0 = time.perf_counter()
    np.cumsum(alpha[:, None] * dz, axis=0)
    add("Soft integration (full video)", 1e3 * (time.perf_counter() - t0), "ms total")


def latex():
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Per-stage runtime. Frame-wise stages report ms/frame; regime "
        r"modeling is a one-off per video.}",
        r"\centering",
        r"\begin{tabular}{|l|c|l|}",
        r"\hline",
        r"\textbf{Stage} & \textbf{Time} & \textbf{Note} \\",
        r"\hline",
    ]
    for stage, tstr, note in rows:
        lines.append(f"{stage} & {tstr} & {note} " + r"\\")
    lines += [r"\hline", r"\end{tabular}", r"\label{tab:runtime}", r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    paths = frames()
    print(f"Benchmarking on {len(paths)} frames of {VIDEO_ID}")
    time_cnn(paths)
    time_flow(paths)
    time_regime(len(paths))
    for s, tstr, n in rows:
        print(f"  {s:32s} {tstr:>16s}  {n}")
    tex = latex()
    with open(f"{OUT_DIR}/tab_runtime.tex", "w") as f:
        f.write(tex + "\n")
    print("\n" + tex)