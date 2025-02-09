from abc import ABC, abstractmethod

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