import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.pyplot as plt


# ================================
# Video config
# ================================
VIDEO_ID = os.environ.get("VIDEO_ID", "2f513ad4ee5e4630")

FRAME_DIR = f"Datasets/kvasir-capsule/frames/{VIDEO_ID}"
OUTPUT_DIR = "results/cnn_embedding"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================
# Device
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================
# Model
# ================================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()


# ================================
# Transform
# ================================
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================================
# Load frames
# ================================
frame_paths = sorted(glob.glob(f"{FRAME_DIR}/frame_*.jpg"))

print(f"\nProcessing VIDEO_ID: {VIDEO_ID}")
print(f"Total frames: {len(frame_paths)}")

embeddings = []

# ================================
# Extract embeddings
# ================================
with torch.no_grad():
    for path in frame_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0).to(device)

        z = model(img)
        z = z.squeeze(0).cpu().numpy()
        embeddings.append(z)

embeddings = np.stack(embeddings)


# ================================
# Compute motion (scalar)
# ================================
embedding_motion = {}

for t in range(len(embeddings) - 1):
    diff = np.linalg.norm(embeddings[t + 1] - embeddings[t])
    embedding_motion[t] = diff


# ================================
# Store full vectors
# ================================
embedding_vectors = {}

for t in range(len(embeddings)):
    embedding_vectors[t] = embeddings[t]


# ================================
# SAVE (FIXED: per video)
# ================================
with open(f"{OUTPUT_DIR}/{VIDEO_ID}_embedding_motion.pkl", "wb") as f:
    pickle.dump(embedding_motion, f)

with open(f"{OUTPUT_DIR}/{VIDEO_ID}_embedding_vectors.pkl", "wb") as f:
    pickle.dump(embedding_vectors, f)

np.save(f"{OUTPUT_DIR}/{VIDEO_ID}_embeddings.npy", embeddings)


# ================================
# Plot
# ================================
fps = 2
times = np.array(list(embedding_motion.keys())) / fps
values = np.array(list(embedding_motion.values()))

plt.figure(figsize=(12, 4))
plt.plot(times, values)
plt.xlabel("Time (seconds)")
plt.ylabel("Embedding distance ||z(t+1) - z(t)||")
plt.title(f"CNN Embedding Motion ({VIDEO_ID})")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/{VIDEO_ID}_embedding_motion.png")
plt.close()


print(f"Saved outputs for {VIDEO_ID}")