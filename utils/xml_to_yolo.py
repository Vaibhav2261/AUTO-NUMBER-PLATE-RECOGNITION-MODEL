import os
import xml.etree.ElementTree as ET
from PIL import Image

ANNOT_DIR = "data/raw/"
OUTPUT_DIR = "data/raw/"
CLASS_NAME = "plate"  # Update this if your class label is different

for filename in os.listdir(ANNOT_DIR):
    if not filename.endswith(".xml"):
        continue

    xml_path = os.path.join(ANNOT_DIR, filename)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_filename = root.find("filename").text
    image_path = os.path.join(ANNOT_DIR, image_filename)
    if not os.path.exists(image_path):
        continue

    image = Image.open(image_path)
    w, h = image.size

    txt_filename = os.path.splitext(image_filename)[0] + ".txt"
    with open(os.path.join(OUTPUT_DIR, txt_filename), "w") as f:
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name.lower() != CLASS_NAME:
                continue
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            xmax = int(bndbox.find("xmax").text)
            ymin = int(bndbox.find("ymin").text)
            ymax = int(bndbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            box_width = (xmax - xmin) / w
            box_height = (ymax - ymin) / h

            f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")