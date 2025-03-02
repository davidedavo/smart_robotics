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
        img_test_thresh = self.preprocess(gray)

        contours, _ = cv2.findContours(img_test_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []

        '''for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)

            aspect_ratio = w / float(h)
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * (radius ** 2)

            if len(approx) >= 4 and area > 300 and 0.8 < aspect_ratio < 1.2 and (area / circle_area) < 0.85:
                xc, yc = x + w // 2, y + h // 2
                bboxes.append([xc, yc, w, h])

        return np.array(bboxes, dtype=np.int32)'''

        classes_ids = [] # 0: Quadrato,   1: Rettangolo,   2: Cerchio 
        for cnt in contours:
            # Approssima il contorno per identificare la forma
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            area = cv2.contourArea(cnt)
            
            # Calcola la bounding box del contorno
            x, y, w, h = cv2.boundingRect(cnt)

            # Calcola il cerchio minimo che contiene il contorno
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * (radius ** 2)
            
            # Calcola la circularità
            circularity = area / circle_area if circle_area != 0 else 0

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            w, h = rect[1]

            # Verifica se il contorno ha 4 lati (forma poligonale) e valuta l'aspect ratio
            if len(approx) == 4:
                aspect_ratio = w / float(h)  # rapporto tra larghezza e altezza

                if 0.9 <= aspect_ratio <= 1.1:  # Forma quasi quadrata
                    # print(f"Quadrato rilevato: {(x, y, w, h)}")
                    #cv2.drawContours(img_test, [approx], -1, (0, 255, 0), 2)  # Disegna in verde
                    class_id = 0
                else:
                    # print(f"Rettangolo rilevato: {(x, y, w, h)}")
                    #cv2.drawContours(img_test, [approx], -1, (255, 255, 0), 2)  # Disegna in azzurro
                    class_id = 1

                # Aggiungiamo la bbox trovata
                # bboxes.append([x + w // 2, y + h // 2, w, h])
                bboxes.append(rect)
                classes_ids += [class_id]

            # Se non ha 4 lati e la circularità è elevata, lo consideriamo un cilindro
            elif circularity > 0.8:
                # print(f"Cilindro rilevato: {(x, y, w, h)}")
                #cv2.drawContours(img_test, [approx], -1, (0, 0, 255), 2)  # Disegna in rosso
                # bboxes.append([x + w // 2, y + h // 2, w, h])
                bboxes.append(rect)
                class_id = 2
                classes_ids += [class_id]

        # Se desideri visualizzare l'immagine con i disegni per debug, puoi usare:
        # cv2.imshow("Contorni rilevati", img_test)
        # cv2.waitKey(1)

        # Ritorna le bbox come array numpy e la lista di flag
        # return np.array(bboxes, dtype=np.int32), np.array(classes_ids, dtype=np.int32)[:, None]
        return bboxes, classes_ids

