import pytesseract

from utils.ocr_utils import preprocess_plate

config = (
    "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

...
for idx, box in enumerate(results[0].boxes):
    ...
    processed = preprocess_plate(plate)
    cv2.imwrite(f"debug_crops/plate_{idx}.png", processed)
    text = pytesseract.image_to_string(processed, config=config)
