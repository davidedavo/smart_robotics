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
        x0 = 262
        y0 = 218
        w = 14
        h = 14
        xc = x0 + w/2
        yc = y0 + h/2
        return np.array([[xc, yc, w, h]], dtype=np.int32)