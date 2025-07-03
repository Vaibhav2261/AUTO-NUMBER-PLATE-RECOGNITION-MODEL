import os
import shutil
import random

RAW_DIR = "data/raw"
OUTPUT_DIR = "data"
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}

# Create output dirs
for split in SPLIT_RATIOS:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# Collect image files
images = [f for f in os.listdir(RAW_DIR) if f.endswith(('.jpg', '.png'))]
random.shuffle(images)

# Calculate splits
total = len(images)
train_cut = int(SPLIT_RATIOS["train"] * total)
val_cut = train_cut + int(SPLIT_RATIOS["val"] * total)

# Split and copy files
for i, img in enumerate(images):
    label = os.path.splitext(img)[0] + ".txt"
    if i < train_cut:
        split = "train"
    elif i < val_cut:
        split = "val"
    else:
        split = "test"

    for f in [img, label]:
        src = os.path.join(RAW_DIR, f)
        dst = os.path.join(OUTPUT_DIR, split, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)

print("✅ Dataset split complete!")
