import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.pyplot as plt


FRAME_DIR = "Datasets/kvasir-capsule/frames/2f513ad4ee5e4630"
OUTPUT_DIR = "results/vit_embedding"
os.makedirs(OUTPUT_DIR, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
model.heads = torch.nn.Identity()  # remove classification head
model = model.to(device)
model.eval()


transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


frame_paths = sorted(glob.glob(f"{FRAME_DIR}/frame_*.jpg"))

if len(frame_paths) == 0:
    raise RuntimeError(f"No frames found in {FRAME_DIR}")

embeddings = []

with torch.no_grad():
    for path in frame_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0).to(device)

        z = model(img)            # (1, 768)
        z = z.squeeze(0).cpu().numpy()
        embeddings.append(z)

embeddings = np.stack(embeddings)  # (T, 768)


embedding_motion = {}

for t in range(len(embeddings) - 1):
    diff = np.linalg.norm(embeddings[t + 1] - embeddings[t])
    embedding_motion[t] = diff


with open(f"{OUTPUT_DIR}/embedding_motion.pkl", "wb") as f:
    pickle.dump(embedding_motion, f)


fps = 2
times = np.array(list(embedding_motion.keys())) / fps
values = np.array(list(embedding_motion.values()))

plt.figure(figsize=(12, 4))
plt.plot(times, values)
plt.xlabel("Time (seconds)")
plt.ylabel("Embedding distance ||z(t+1) - z(t)||")
plt.title("ViT Embedding Motion Signal (ViT-B/16)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/embedding_motion.png")
plt.show()
