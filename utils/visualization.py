import cv2
import matplotlib.pyplot as plt


def draw_boxes(image, boxes, labels=None, color=(0, 255, 0)):
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if labels:
            cv2.putText(
                image, labels[i], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
    return image


def show_image(image, title="Image"):
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()
