import cv2
import os
from model.load_model import load_yolo_model
from ocr.ocr_reader import read_plate_text
from utils.file_utils import save_image_with_text

def detect_and_read(image_path, model_path="model/yolov8n.pt", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    model = load_yolo_model(model_path)
    results = model(image_path)[0]
    image = cv2.imread(image_path)

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        crop = image[y1:y2, x1:x2]
        plate_text = read_plate_text(crop)
        save_image_with_text(image.copy(), f"{output_dir}/{os.path.basename(image_path)}", plate_text, (x1, y1, x2, y2))
        print(f"Detected Text: {plate_text}")

if __name__ == "__main__":
    detect_and_read("test_images/car1.jpg")
