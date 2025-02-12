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
    

# Detector basato su SIFT
class SIFTObjectDetector(ObjectDetector):
    def __init__(self, template_path):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()
        self.template = cv2.imread(template_path, 0)  # Template in scala di grigi
        self.kp_template, self.des_template = self.sift.detectAndCompute(self.template, None)

    def detect_box(self, rgb_image):
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        kp_image, des_image = self.sift.detectAndCompute(gray, None)

        matches = self.bf.knnMatch(self.des_template, des_image, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) > 5:
            src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            h, w = self.template.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)

            x, y, w, h = cv2.boundingRect(dst)
            xc, yc = x + w // 2, y + h // 2
            return np.array([[xc, yc, w, h]])  # Restituisce la bounding box

        return np.empty((0, 4))  # Nessun oggetto rilevato

# Detector basato su Template Matching
class TemplateMatchingObjectDetector(ObjectDetector):
    def __init__(self, template_path, threshold=0.8):
        self.template = cv2.imread(template_path, 0)  # Carica il template in scala di grigi
        self.w, self.h = self.template.shape[::-1]  # Ottieni larghezza e altezza del template
        self.threshold = threshold  # Soglia di corrispondenza

    def detect_box(self, rgb_image):
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= self.threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + self.w, top_left[1] + self.h)

            # Calcola il centro e restituisci bounding box [xc, yc, w, h]
            xc, yc = top_left[0] + self.w // 2, top_left[1] + self.h // 2
            return np.array([[xc, yc, self.w, self.h]])

        return np.empty((0, 4))  # Nessun oggetto rilevato