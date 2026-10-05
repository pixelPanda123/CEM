"""
frame_quality.py
----------------
Computes LABEL-FREE image-degradation signals per frame, used later to test
whether the HMM's learned "stable" regime actually corresponds to usable
imagery (external validation, no pose labels needed).

Per frame:
  * sharpness  = variance of the Laplacian      (low  => motion blur / defocus)
  * glare      = fraction of near-saturated px   (high => specular reflection)
  * darkness   = 1 - mean intensity              (high => dark, uninformative)

Saves results/frame_quality/<VIDEO_ID>_quality.npy  (dict of 1-D arrays, len F).

Run per video from project root:
    VIDEO_ID=4c059b716bf84ea3 python3 experiments/extra/frame_quality.py
"""

import os, glob
import numpy as np
import cv2

VIDEO_ID = os.environ.get("VIDEO_ID", "2f513ad4ee5e4630")
FRAME_DIR = f"Datasets/kvasir-capsule/frames/{VIDEO_ID}"
OUT_DIR = "results/frame_quality"
os.makedirs(OUT_DIR, exist_ok=True)

GLARE_LEVEL = 240     # 0-255; pixels brighter than this count as glare
DARK_IS_1MINUS = True


def frame_signals(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    glare = float(np.mean(gray >= GLARE_LEVEL))
    mean_i = float(np.mean(gray)) / 255.0
    darkness = (1.0 - mean_i) if DARK_IS_1MINUS else mean_i
    return sharpness, glare, darkness


if __name__ == "__main__":
    paths = sorted(glob.glob(f"{FRAME_DIR}/frame_*.jpg"))
    if not paths:
        raise SystemExit(f"No frames in {FRAME_DIR}")
    sharp, glare, dark = [], [], []
    for p in paths:
        s, g, d = frame_signals(p)
        sharp.append(s); glare.append(g); dark.append(d)

    out = {
        "sharpness": np.array(sharp),
        "glare": np.array(glare),
        "darkness": np.array(dark),
    }
    np.save(f"{OUT_DIR}/{VIDEO_ID}_quality.npy", out)
    print(f"{VIDEO_ID}: {len(paths)} frames | "
          f"sharpness {np.mean(sharp):.1f} | glare {np.mean(glare):.4f} | "
          f"darkness {np.mean(dark):.3f}")