import pickle
import numpy as np


CNN_WIN = "results/temporal_windowing/cnn_mean_window_W5.pkl"
VIT_WIN = "results/temporal_windowing/vit_mean_window_W5.pkl"


def temporal_consistency(signal):
    diffs = np.abs(np.diff(signal))
    return np.mean(diffs)


with open(CNN_WIN, "rb") as f:
    cnn_signal = pickle.load(f)

with open(VIT_WIN, "rb") as f:
    vit_signal = pickle.load(f)

cnn_signal = np.array(cnn_signal)
vit_signal = np.array(vit_signal)


cnn_tcs = temporal_consistency(cnn_signal)
vit_tcs = temporal_consistency(vit_signal)


print("Temporal Consistency Score")
print("-----------------------------------------------")
print(f"CNN  TCS = {cnn_tcs:.4f}")
print(f"ViT  TCS = {vit_tcs:.4f}")
