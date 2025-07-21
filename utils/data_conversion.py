import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(xml_file, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    yolo_labels = []

    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))

        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_labels.append(f"0 {x_center} {y_center} {width} {height}")
    
    return yolo_labels

def batch_convert_voc_folder(xml_folder, output_folder, img_shape_lookup):
    os.makedirs(output_folder, exist_ok=True)
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue
        name = os.path.splitext(xml_file)[0]
        if name not in img_shape_lookup:
            continue
        h, w = img_shape_lookup[name]
        yolo_data = convert_voc_to_yolo(os.path.join(xml_folder, xml_file), w, h)
        with open(os.path.join(output_folder, name + '.txt'), 'w') as f:
            for line in yolo_data:
                f.write(line + '\n')
