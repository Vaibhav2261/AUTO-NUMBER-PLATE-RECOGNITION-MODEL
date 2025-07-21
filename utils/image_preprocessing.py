import cv2

def enhance_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    return clahe.apply(gray)

def threshold_plate(image):
    blur = cv2.bilateralFilter(image, 11, 17, 17)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    return thresh