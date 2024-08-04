import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cv2
import numpy as np
from paddleocr import PaddleOCR

class OCREngine:
    def __init__(self):
        self.char_whitelist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, 
                             show_log=False, log_level='error', rec_char_type='ch',
                             rec_char_whitelist=self.char_whitelist)

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return thresh
    
    def calculateArea(self, bbox):
        return np.prod(np.ptp(bbox, axis=0))
    
    def recognize(self, image):
        preprocessed = self.preprocess(image)
        results = self.ocr.ocr(preprocessed, cls=True)

        if not results or len(results)==0:
            return "NO_TEXT_DETECTED"

        largest_bbox = None
        largest_area = 0
        largest_text = ""

        for result in results:
            if result is None:
                continue
            for line in result:
                if len(line) < 2:
                    continue
                bbox, (text, confidence) = line
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                area = self.calculateArea(bbox)
                
                if area > largest_area:
                    largest_area = area
                    largest_bbox = bbox
                    largest_text = text

        return largest_text.strip()