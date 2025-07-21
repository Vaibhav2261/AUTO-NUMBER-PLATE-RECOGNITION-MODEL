import torch
import cv2
import os
import pytesseract
from pathlib import Path

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Image source folder
SOURCE_DIR = 'test_images'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each image
for image_name in os.listdir(SOURCE_DIR):
    img_path = os.path.join(SOURCE_DIR, image_name)
    img = cv2.imread(img_path)

    # Run detection
    results = model(img)
    detections = results.xyxy[0].numpy()

    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        plate = img[y1:y2, x1:x2]

        # OCR using Tesseract
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--psm 7').strip()

        # Annotate image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Save text to file
        with open(os.path.join(OUTPUT_DIR, f"{Path(image_name).stem}_text.txt"), "w") as f:
            f.write(text)

    # Save annotated image
    out_path = os.path.join(OUTPUT_DIR, image_name)
    cv2.imwrite(out_path, img)

print("✅ Detection + OCR complete. Check 'outputs/' folder.")
