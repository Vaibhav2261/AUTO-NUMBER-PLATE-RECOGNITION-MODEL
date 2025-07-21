import os
import cv2

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image_with_text(image, save_path, text=None, box=None):
    if text and box:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.imwrite(save_path, image)

def read_images_from_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def save_output(input_path, annotated_img, texts, output_dir='outputs'):
    """ Save the annotated image and extracted text """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save image
    base = os.path.basename(input_path)
    name, _ = os.path.splitext(base)
    output_img_path = os.path.join(output_dir, f"{name}_output.jpg")
    cv2.imwrite(output_img_path, annotated_img)

    # Save text
    output_txt_path = os.path.join(output_dir, f"{name}_text.txt")
    with open(output_txt_path, 'w') as f:
        for i, txt in enumerate(texts):
            f.write(f"Plate {i+1}: {txt}\n")

    print(f"[INFO] Results saved to {output_img_path} and {output_txt_path}")
