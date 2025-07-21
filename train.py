import os
import argparse

def train_yolo():
    import torch

    # Clone YOLOv5 repo if not already present
    if not os.path.exists("yolov5"):
        os.system("git clone https://github.com/ultralytics/yolov5.git")

    # Install YOLOv5 dependencies
    os.system("pip install -r yolov5/requirements.txt")

    # Train command
    os.system(
        "python yolov5/train.py "
        "--img 640 "
        "--batch 16 "
        "--epochs 50 "
        "--data anpr.yaml "
        "--weights yolov5s.pt "
        "--name plate_detector"
    )

if __name__ == "__main__":
    train_yolo()