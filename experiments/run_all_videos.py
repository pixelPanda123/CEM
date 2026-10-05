import os

VIDEOS = [
    "4c059b716bf84ea3",
    "4ed494e20fd349ad",
    "2f513ad4ee5e4630",
    "4ab482cc46054935"

]

for vid in VIDEOS:
    print(f"\n===== Processing {vid} =====")

    os.environ["VIDEO_ID"] = vid
    os.system("python3 experiments/metrics/temporal_consistency.py")