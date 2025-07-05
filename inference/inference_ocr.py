# inference_ocr.py
import cv2
import pytesseract
from ultralytics import YOLO

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def main(weights="runs/detect/train/weights/best.pt", source="test.jpg", save_crops="plates/"):
    model = YOLO(weights)
    results = model.predict(source, imgsz=640, conf=0.25, verbose=False)

    for i, res in enumerate(results):
        img = cv2.imread(source) if isinstance(source, str) else source[i]
        for j, box in enumerate(res.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            pre = preprocess_for_ocr(crop)
            txt = pytesseract.image_to_string(pre, 
                config=r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            plate = txt.strip()
            print(f"[Plate {i}-{j}] {plate}")
            cv2.imwrite(f"{save_crops}plate_{i}_{j}.png", crop)

if __name__ == "__main__":
    import sys, os
    os.makedirs("plates", exist_ok=True)
    src = sys.argv[1] if len(sys.argv)>1 else "test.jpg"
    main(source=src)
