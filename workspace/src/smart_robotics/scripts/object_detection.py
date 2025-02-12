from abc import ABC, abstractmethod
import cv2
import numpy as np

class ObjectDetector(ABC):
    @abstractmethod
    def detect_box(self, rgb_image):
        """
        This method returns a numpy of np.int32 array of shape (N_DETS, 4), where the four bbox params are [xc, yc, w, h].

        TODO: Condier also bbox rotation?
        """
        pass

class HardCodedObjectDetector(ObjectDetector):
    def detect_box(self, rgb_image) -> np.ndarray:
        # cube
        x0 = 196
        y0 = 192
        w = 32
        h = 32
        xc = x0 + w/2
        yc = y0 + h/2
        bboxes=[[xc, yc, w, h]]

        # cylinder
        x0 = 333
        y0 = 189
        w = 38
        h = 39
        xc = x0 + w/2
        yc = y0 + h/2
        bboxes += [[xc, yc, w, h]]
        return np.array(bboxes, dtype=np.int32)
    

class ContourObjectDetector(ObjectDetector):
    def __init__(self, template_path):
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    def preprocess(self, image):
        """Applica preprocessing per migliorare il rilevamento dei contorni"""
        if len(image.shape) == 3:  # Se ha tre canali, convertilo in scala di grigi
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # **Converti l'immagine in uint8**
        if image.dtype != np.uint8:
            image = cv2.convertScaleAbs(image, alpha=255.0)
            
        img_eq = cv2.equalizeHist(image)
        img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
        _, img_thresh = cv2.threshold(img_blur, 120, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        return img_morph

    def detect_box(self, rgb_image):
        """Rileva i contorni e restituisce le bounding box"""
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        print(f"Gray image dtype: {gray.dtype}, shape: {gray.shape}")
        img_test_thresh = self.preprocess(gray)

        contours, _ = cv2.findContours(img_test_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)

            aspect_ratio = w / float(h)
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * (radius ** 2)

            if len(approx) >= 4 and area > 300 and 0.8 < aspect_ratio < 1.2 and (area / circle_area) < 0.85:
                xc, yc = x + w // 2, y + h // 2
                bboxes.append([xc, yc, w, h])

        return np.array(bboxes, dtype=np.int32)

