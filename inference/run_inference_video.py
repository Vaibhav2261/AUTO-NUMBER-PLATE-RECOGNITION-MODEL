from ultralytics import YOLO
import cv2
from ocr.tesseract_reader import extract_text

model = YOLO("model/best.pt")
cap = cv2.VideoCapture("data/raw/car_video.mp4")  # or 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = frame[y1:y2, x1:x2]
            text = extract_text(plate_crop)
            print("Detected Plate:", text)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Video ANPR", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ocr/tesseract_reader.py
import pytesseract
import cv2

def extract_text(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray, config='--psm 7').strip()
