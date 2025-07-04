import os
import xml.etree.ElementTree as ET
from PIL import Image

ANNOT_DIR = "data/raw/"
CLASS_NAME = "plate"  # Change if your class name differs

for filename in os.listdir(ANNOT_DIR):
    if not filename.endswith(".xml"):
        continue

    xml_path = os.path.join(ANNOT_DIR, filename)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_filename = root.find("filename").text
    image_path = os.path.join(ANNOT_DIR, image_filename)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    image = Image.open(image_path)
    w, h = image.size

    txt_filename = os.path.splitext(image_filename)[0] + ".txt"
    txt_path = os.path.join(ANNOT_DIR, txt_filename)

    with open(txt_path, "w") as f:
        for obj in root.findall("object"):
            name = obj.find("name").text.lower()
            if name != CLASS_NAME:
                continue
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymin = int(float(bndbox.find("ymin").text))
            ymax = int(float(bndbox.find("ymax").text))

            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            box_width = (xmax - xmin) / w
            box_height = (ymax - ymin) / h

            f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

print("✅ XML to YOLO TXT conversion completed.")