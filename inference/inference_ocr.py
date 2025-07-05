from ultralytics import YOLO
import cv2
import pytesseract
import os

# Set Tesseract path (update this path if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the trained YOLO model
model = YOLO("runs/detect/train3/weights/best.pt")

# Folder containing test images
source_folder = "data/test/images"
output_file = "ocr_predictions.csv"
results = []

# Loop through images
for img_name in os.listdir(source_folder):
    img_path = os.path.join(source_folder, img_name)
    image = cv2.imread(img_path)

    # Run YOLO detection
    detections = model(img_path)[0]
    
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]

        # Apply OCR on cropped plate
        plate_number = pytesseract.image_to_string(cropped, config='--psm 7').strip()
        print(f"{img_name}: {plate_number}")
        results.append((img_name, plate_number))

        # Optional: save cropped image
        cv2.imwrite(f"runs/crops/{img_name}", cropped)

# Save all predictions
with open(output_file, "w") as f:
    for img, plate in results:
        f.write(f"{img},{plate}\n")
