import os
import random
import shutil

# Set paths
RAW_DIR = "data/raw"
DEST_DIR = "data"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
LABEL_EXTENSION = ".txt"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Create destination folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(DEST_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, split, "labels"), exist_ok=True)

# Collect all image files (matching only those that also have corresponding .txt label)
all_images = []
for file in os.listdir(RAW_DIR):
    if any(file.endswith(ext) for ext in IMAGE_EXTENSIONS):
        label_file = os.path.splitext(file)[0] + LABEL_EXTENSION
        if os.path.exists(os.path.join(RAW_DIR, label_file)):
            all_images.append(file)

print(f"Found {len(all_images)} annotated image-label pairs.")

# Shuffle and split
random.shuffle(all_images)
total = len(all_images)
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_files = all_images[:train_end]
val_files = all_images[train_end:val_end]
test_files = all_images[val_end:]

splits = [("train", train_files), ("val", val_files), ("test", test_files)]

# Copy files
for split_name, files in splits:
    for img_file in files:
        label_file = os.path.splitext(img_file)[0] + LABEL_EXTENSION

        # Copy image
        shutil.copy2(
            os.path.join(RAW_DIR, img_file),
            os.path.join(DEST_DIR, split_name, "images", img_file),
        )
        # Copy label
        shutil.copy2(
            os.path.join(RAW_DIR, label_file),
            os.path.join(DEST_DIR, split_name, "labels", label_file),
        )

    print(f"✅ {split_name.upper()}: {len(files)} samples copied.")

print("🎯 Dataset split complete!")
