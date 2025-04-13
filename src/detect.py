import cv2
import numpy as np
from ultralytics import YOLO

def detect_objects(image, model_path='model/best.pt'):
    model = YOLO(model_path)
    results = model(image)
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            damage_type = model.names[cls]
            detections.append({
                "damage_type": damage_type,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
    return detections