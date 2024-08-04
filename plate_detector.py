import torch
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path, conf_threshold=0.25, device=None):
        self.conf_threshold = conf_threshold
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, image):
        results = self.model(image)[0]
        detections = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > self.conf_threshold:
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'score': score
                })

        return detections