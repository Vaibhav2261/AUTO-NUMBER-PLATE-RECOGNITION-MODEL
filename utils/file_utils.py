import os

import cv2


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_image_with_text(image, save_path, text=None, box=None):
    if text and box:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
        )
    cv2.imwrite(save_path, image)


def read_images_from_folder(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
