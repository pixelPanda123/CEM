import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------
# Load normalized TMD
# -----------------------
TMD_PATH = "results/tmd/cnn_tmd_norm.npy"
TIME_PATH = "results/tmd/cnn_tmd_times.npy"

tmd = np.load(TMD_PATH)
times = np.load(TIME_PATH)

print("TMD shape:", tmd.shape)

# -----------------------
# Run K-means for K=2,3
# -----------------------
results = {}

for K in [2, 3]:
    print(f"\nRunning K-means with K={K}")

    kmeans = KMeans(
        n_clusters=K,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(tmd)

    sil = silhouette_score(tmd, labels)

    results[K] = {
        "labels": labels,
        "silhouette": sil
    }

    print(f"Silhouette score (K={K}): {sil:.4f}")

# -----------------------
# Save regime labels
# -----------------------
for K in results:
    np.save(
        f"results/tmd/cnn_regime_labels_K{K}.npy",
        results[K]["labels"]
    )

print("\nRegime separation complete.")
