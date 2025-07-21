# inference/run.py

import cv2
import torch
from ultralytics import YOLO
import pytesseract
import numpy as np
from utils.ocr_utils import preprocess_plate
from utils.file_utils import save_output

model = YOLO("runs/detect/train/weights/best.pt")  # Load trained model

def infer_on_image(img_path):
    img = cv2.imread(img_path)
    results = model(img)
    
    annotated_img = img.copy()
    texts = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate = img[y1:y2, x1:x2]
        processed = preprocess_plate(plate)
        text = pytesseract.image_to_string(processed, config='--psm 7')
        texts.append(text.strip())

        # Draw results
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(annotated_img, text.strip(), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    save_output(img_path, annotated_img, texts)
    return annotated_img, texts
