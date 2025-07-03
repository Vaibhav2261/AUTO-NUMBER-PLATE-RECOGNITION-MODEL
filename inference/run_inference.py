# Inference script: detection + OCR
from ultralytics import YOLO
import cv2
from ocr.tesseract_reader import extract_text

model = YOLO("model/best.pt")
frame = cv2.imread("data/test/sample.jpg")

results = model(frame)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = frame[y1:y2, x1:x2]
        text = extract_text(plate_crop)
        print("Detected Plate:", text)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

cv2.imshow('Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()