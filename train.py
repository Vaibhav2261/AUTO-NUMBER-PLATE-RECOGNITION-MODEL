# Training script entrypoint
from ultralytics import YOLO

def train_yolo():
    model = YOLO('model/yolov8n.pt')  
    model.train(
        data='anpr.yaml',
        epochs=50,
        imgsz=640,
        project='runs/train',
        name='anpr_yolo',
        batch=16,
        workers=4
    )

if __name__ == '__main__':
    train_yolo()