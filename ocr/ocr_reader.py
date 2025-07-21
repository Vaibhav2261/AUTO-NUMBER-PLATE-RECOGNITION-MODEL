import pytesseract
import cv2

def read_plate_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)

    config = "--psm 7"
    text = pytesseract.image_to_string(thresh, config=config)
    return text.strip()
