import pickle
import numpy as np


CNN_RAW = "results/cnn_embedding/embedding_motion.pkl"
CNN_WIN = "results/temporal_windowing/cnn_mean_window_W5.pkl"

VIT_RAW = "results/vit_embedding/embedding_motion.pkl"
VIT_WIN = "results/temporal_windowing/vit_mean_window_W5.pkl"


def load_raw_variance(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    values = np.array(list(data.values()))
    return np.var(values)

def load_windowed_variance(path):
    with open(path, "rb") as f:
        values = pickle.load(f)
    return np.var(values)


cnn_raw_var = load_raw_variance(CNN_RAW)
cnn_win_var = load_windowed_variance(CNN_WIN)
cnn_vrr = cnn_raw_var / cnn_win_var

vit_raw_var = load_raw_variance(VIT_RAW)
vit_win_var = load_windowed_variance(VIT_WIN)
vit_vrr = vit_raw_var / vit_win_var


print("Variance Reduction Ratio (VRR)")
print(f"CNN  | Raw Var = {cnn_raw_var:.4f}, Windowed Var = {cnn_win_var:.4f}, VRR = {cnn_vrr:.2f}")
print(f"ViT  | Raw Var = {vit_raw_var:.4f}, Windowed Var = {vit_win_var:.4f}, VRR = {vit_vrr:.2f}")
