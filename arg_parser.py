import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for license plate detection")
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Initial weights path')
    parser.add_argument('--device', type=str, default='', help='Device to run on')
    return parser.parse_args()