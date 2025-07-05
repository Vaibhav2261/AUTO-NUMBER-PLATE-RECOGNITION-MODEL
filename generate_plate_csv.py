import os
import cv2
import pytesseract
from ultralytics import YOLO

# ⛔ Update this path if Tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ✅ Load trained YOLO model
model = YOLO('model/best.pt')

# 📁 Folder containing the 400+ raw images
image_folder = 'data/raw/'
output_csv = 'number_plate_predictions.csv'

# 📄 Open output CSV and write header
with open(output_csv, 'w') as f:
    f.write("Image,NumberPlate\n")

    # 🔁 Process each image
    for img_name in os.listdir(image_folder):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_name}")
            continue

        results = model(img)
        number = ""

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = img[y1:y2, x1:x2]

                # 👁️ Preprocess for OCR
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 11, 17, 17)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

                text = pytesseract.image_to_string(thresh, config='--psm 7')
                number = text.strip().replace("\n", "").replace(" ", "")
                break  # First plate per image only

        if number == "":
            number = "NOT_DETECTED"

        print(f"{img_name}: {number}")
        f.write(f"{img_name},{number}\n")
