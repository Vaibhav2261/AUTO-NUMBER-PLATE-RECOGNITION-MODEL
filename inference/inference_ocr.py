# inference/inference_ocr.py
import cv2
from ultralytics import YOLO
import pytesseract
import os

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

def extract_text(crop):
    config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return pytesseract.image_to_string(preprocess(crop), config=config).strip()

def run_ocr_on_folder(model_path, input_folder, output_folder="results"):
    model = YOLO(model_path)
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if not file.lower().endswith((".jpg", ".png")): continue
        img_path = os.path.join(input_folder, file)
        image = cv2.imread(img_path)
        results = model.predict(img_path, conf=0.25, verbose=False)
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]
            text = extract_text(crop)
            print(f"{file}: {text}")
            cv2.imwrite(f"{output_folder}/{file}", crop)

if __name__ == "__main__":
    run_ocr_on_folder("model/best.pt", "data/test")