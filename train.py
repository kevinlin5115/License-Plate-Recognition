from ultralytics import YOLO
import yaml
import torch
from pathlib import Path
from arg_parser import parse_args

def validate_data_yaml(data_path):
    with open(data_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            required_keys = ['train', 'val', 'test', 'nc', 'names']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key '{key}' in data.yaml")
                print("data.yaml validation successful")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing data.yaml: {e}")
        
def train_yolov8(args):
    # Validate data.yaml
    validate_data_yaml(args.data)

    # Create model
    model = YOLO(args.weights)

    device = args.device if args.device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    if device.startswith('cuda'):
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Train the model
    results = model.train(
        data=args.data,
        epochs = args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=device
    )

def main():
    args = parse_args()
    train_yolov8(args)

if __name__ == "__main__":
    main()