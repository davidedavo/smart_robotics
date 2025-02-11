import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, lower_color=np.array([30, 150, 50]), upper_color=np.array([255, 255, 180])):
        # Soglie colore in HSV (modificale in base agli oggetti)
        self.lower_color = lower_color
        self.upper_color = upper_color

    def detect_box(self, image):
        # Converti l'immagine in HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)

        # Trova i contorni
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x + w // 2, y + h // 2, w, h))  # Centro della BBox
        return boxes



class ObjectDetector_YOLO:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect_box(self, image):
        results = self.model(image)
        boxes = []
        for result in results:
            for box in result.boxes.xywh.numpy():  # Bounding box formato (x, y, w, h)
                x, y, w, h = box
                boxes.append((int(x), int(y), int(w), int(h)))
        return boxes
