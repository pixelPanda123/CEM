import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from hmm_model import GaussianHMM

# -----------------------
# Config
# -----------------------
TMD_PATH = "results/tmd/cnn_tmd_norm.npy"
OUTPUT_DIR = "results/regime_modeling/cnn_hmm"
N_ITER = 25

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Load TMD Features
# -----------------------
if not os.path.exists(TMD_PATH):
    raise FileNotFoundError(f"TMD file not found at {TMD_PATH}")

X = np.load(TMD_PATH)

print("Loaded TMD shape:", X.shape)

# -----------------------
# Normalize (extra safety)
# -----------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------
# Train HMM
# -----------------------
hmm = GaussianHMM(n_states=2)

gamma = hmm.fit(X, n_iter=N_ITER)

# Posterior probability of stable regime
alpha_t = gamma[:, 1]

# -----------------------
# Save artifacts
# -----------------------
np.save(f"{OUTPUT_DIR}/posterior.npy", alpha_t)
np.save(f"{OUTPUT_DIR}/transition_matrix.npy", hmm.A)
np.save(f"{OUTPUT_DIR}/means.npy", hmm.means)
np.save(f"{OUTPUT_DIR}/covariances.npy", hmm.covs)

with open(f"{OUTPUT_DIR}/params.pkl", "wb") as f:
    pickle.dump({
        "pi": hmm.pi,
        "A": hmm.A,
        "means": hmm.means,
        "covs": hmm.covs
    }, f)

print("HMM training complete.")
print("Transition matrix:\n", hmm.A)
print("Posterior shape:", alpha_t.shape)
print("Means:\n", hmm.means)