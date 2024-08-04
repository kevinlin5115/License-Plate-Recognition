import cv2
import os
import numpy as np
from pathlib import Path
from plate_detector import PlateDetector
from ocr_engine import OCREngine

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def process_image(image_path, plate_detector, ocr_engine):
    image = cv2.imread(str(image_path))
    detections = plate_detector.detect(image)
    
    results = []
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        plate_image = image[y1:y2, x1:x2]
        plate_text = ocr_engine.recognize(plate_image)
        results.append({
            'bbox': detection['bbox'],
            'score': detection['score'],
            'text': plate_text
        })
    
    return results

def visualize_results(image, results):
    vis_image = image.copy()
    for result in results:
        x1, y1, x2, y2 = result['bbox']
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{result['text']}"
        cv2.putText(vis_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return vis_image

def main():
    model_path = './model.pt'
    plate_detector = PlateDetector(model_path)
    ocr_engine = OCREngine()

    img_dir = Path('./test')
    img_files = list(img_dir.glob('*.jpg'))

    for img_file in img_files:
        print(f"Processing {img_file}")
        image = cv2.imread(str(img_file))
        results = process_image(img_file, plate_detector, ocr_engine)

        vis_image = visualize_results(image, results)

        for result in results:
            print(f"Detected: {result['text']} (confidence: {result['score']:.2f})")

        cv2.imshow('Result', vis_image)
        key=cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()