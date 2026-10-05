import cv2
import os

VIDEO_ID = "4ed494e20fd349ad"  # change this
VIDEO_PATH = f"Datasets/kvasir-capsule/videos-raw/unlabelled_videos/{VIDEO_ID}.mp4"
OUTPUT_DIR = f"Datasets/kvasir-capsule/frames/{VIDEO_ID}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = 2
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
interval = max(1, frame_rate // fps)

count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % interval == 0:
        frame = cv2.resize(frame, (224, 224))
        cv2.imwrite(f"{OUTPUT_DIR}/frame_{saved:05d}.jpg", frame)
        saved += 1

    count += 1

cap.release()

print(f"Saved {saved} frames to {OUTPUT_DIR}")