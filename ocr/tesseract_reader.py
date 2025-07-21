# Tesseract OCR reader
import cv2
import pytesseract


def extract_text(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray, config="--psm 7").strip()
