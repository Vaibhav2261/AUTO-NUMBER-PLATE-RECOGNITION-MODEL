# Inference script: detection + OCR
from ultralytics import YOLO
import cv2
from ocr.tesseract_reader import extract_text

model = YOLO("model/best.pt")
img = cv2.imread("data/test/sample.jpg")

results = model(img)
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = img[y1:y2, x1:x2]
        text = extract_text(plate_crop)
        print("Detected Plate:", text)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow("Image ANPR", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


from utils.ocr_utils import extract_text_from_plate

# Suppose `plate_image` is cropped using YOLO box
for detection in detections:
    x1, y1, x2, y2 = detection_box
    plate_crop = image[y1:y2, x1:x2]
    
    # OCR
    plate_text = extract_text_from_plate(plate_crop)
    print("Detected Plate:", plate_text)

    # Optional: draw on image
    cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(image, plate_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
