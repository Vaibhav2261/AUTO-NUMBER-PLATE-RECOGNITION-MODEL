from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/train3/weights/best.pt')

# Run validation on the dataset
metrics = model.val(data='data.yaml')

# Print mAP@0.5 (main accuracy metric for detection)
print(f"\n✅ Detection Accuracy (mAP@0.5): {metrics.box.map50:.4f}\n")
