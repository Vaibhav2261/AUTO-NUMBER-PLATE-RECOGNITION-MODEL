import os
import shutil
import random

RAW_DIR = "data/raw"
IMAGE_DIR = "data/images"
LABEL_DIR = "data/labels"

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Create destination folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(IMAGE_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(LABEL_DIR, split), exist_ok=True)

# Get all image files
images = [f for f in os.listdir(RAW_DIR) if f.endswith(".png")]
random.shuffle(images)

total = len(images)
train_count = int(total * train_ratio)
val_count = int(total * val_ratio)

splits = {
    "train": images[:train_count],
    "val": images[train_count:train_count + val_count],
    "test": images[train_count + val_count:]
}

for split, img_list in splits.items():
    for img_file in img_list:
        base = os.path.splitext(img_file)[0]
        txt_file = base + ".txt"

        src_img = os.path.join(RAW_DIR, img_file)
        src_txt = os.path.join(RAW_DIR, txt_file)

        dst_img = os.path.join(IMAGE_DIR, split, img_file)
        dst_txt = os.path.join(LABEL_DIR, split, txt_file)

        shutil.copyfile(src_img, dst_img)
        if os.path.exists(src_txt):
            shutil.copyfile(src_txt, dst_txt)

print("✅ Dataset split into train/val/test completed.")
