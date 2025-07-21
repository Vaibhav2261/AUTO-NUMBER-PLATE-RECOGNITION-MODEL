from ultralytics import YOLO


def load_yolo_model(model_path="model/yolov8n.pt"):
    model = YOLO(model_path)
    return model
