import sys
import os
import torch
import argparse
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
os.environ["YOLO_VERBOSE"] = "False"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import SimpleYOLO, MobileNetYOLO, RoomYOLODataset

def calculate_iou(boxA, boxB):
    xA1 = boxA[0] - (boxA[2] / 2)
    yA1 = boxA[1] - (boxA[3] / 2)
    xA2 = boxA[0] + (boxA[2] / 2)
    yA2 = boxA[1] + (boxA[3] / 2)
    
    xB1 = boxB[0] - (boxB[2] / 2)
    yB1 = boxB[1] - (boxB[3] / 2)
    xB2 = boxB[0] + (boxB[2] / 2)
    yB2 = boxB[1] + (boxB[3] / 2)
    
    inter_x1 = max(xA1, xB1)
    inter_y1 = max(yA1, yB1)
    inter_x2 = min(xA2, xB2)
    inter_y2 = min(yA2, yB2)
    
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    
    union_area = boxA_area + boxB_area - inter_area
    if union_area == 0: return 0.0
    return inter_area / union_area


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple', 
                        choices=['simple', 'mobilenet', 'yolo26', 'yolo12'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating model: {args.model.upper()}")

    correct_classifications = 0
    total_iou = 0.0
    true_positives = 0
    total_images = 0

    # ==========================================
    # TRACK 1: SOTA SDK EVALUATION
    # ==========================================
    if args.model in ['yolo26', 'yolo12']:
        from ultralytics import YOLO
        weight_name = 'yolo26n.pt' if args.model == 'yolo26' else 'yolo12n.pt'
        sota_model = YOLO(weight_name)

        test_frames_dir = os.path.join(os.path.dirname(__file__), '..', 'test_frames')
        test_labels_dir = os.path.join(os.path.dirname(__file__), '..', 'test_labels')
        valid_extensions = ('.jpg', '.jpeg', '.png')
        
        image_files = [f for f in os.listdir(test_frames_dir) if f.lower().endswith(valid_extensions)]
        total_images = len(image_files)

        for img_name in image_files:
            img_path = os.path.join(test_frames_dir, img_name)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(test_labels_dir, label_name)

            # Read Ground Truth Target
            target_conf = 0.0
            target_box = None
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if lines and len(lines[0].strip()) > 0:
                        parts = lines[0].strip().split()
                        target_conf = 1.0
                        target_box = [float(x) for x in parts[1:5]]

            # Read Image and Run Inference
            img = cv2.imread(img_path)
            h_orig, w_orig = img.shape[:2]

            pred_conf = 0.0
            pred_box = None

            results = sota_model(img, classes=[0], verbose=False)
            
            # Ultralytics Parsing
            if len(results[0].boxes) > 0:
                raw_conf = float(results[0].boxes.conf[0])
                if raw_conf >= 0.5:
                    pred_conf = 1.0
                    xyxy = results[0].boxes.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    pred_box = [
                        ((x1 + x2) / 2) / w_orig,
                        ((y1 + y2) / 2) / h_orig,
                        (x2 - x1) / w_orig,
                        (y2 - y1) / h_orig
                    ]

            # Tally Metrics
            pred_class = 1.0 if pred_conf >= 0.5 else 0.0
            if pred_class == target_conf:
                correct_classifications += 1
            
            if pred_class == 1.0 and target_conf == 1.0:
                iou = calculate_iou(pred_box, target_box)
                total_iou += iou
                true_positives += 1

    # ==========================================
    # TRACK 2: CUSTOM PYTORCH EVALUATION
    # ==========================================
    else:
        if args.model == 'mobilenet':
            model = MobileNetYOLO().to(device)
            weights_filename = 'room_yolo_weights_mobilenet.pth'
            my_transforms = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Resize((256, 256)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            model = SimpleYOLO().to(device)
            weights_filename = 'room_yolo_weights_simple.pth'
            my_transforms = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Resize((256, 256))
            ])
        
        test_dataset = RoomYOLODataset(img_dir='test_frames', label_dir='test_labels', transform=my_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        total_images = len(test_dataset)

        weights_path = os.path.join(os.path.dirname(__file__), '..', weights_filename)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.cpu()[0]
                
                predictions = model(images).cpu()[0]
                pred_conf = torch.sigmoid(predictions[0]).item()
                
                target_conf = targets[0].item()
                pred_class = 1.0 if pred_conf > 0.5 else 0.0
                
                if pred_class == target_conf:
                    correct_classifications += 1
                    
                if pred_class == 1.0 and target_conf == 1.0:
                    pred_box = predictions[1:5].numpy()
                    target_box = targets[1:5].numpy()
                    iou = calculate_iou(pred_box, target_box)
                    total_iou += iou
                    true_positives += 1

    class_acc = (correct_classifications / total_images) * 100 if total_images > 0 else 0
    avg_iou = (total_iou / true_positives) if true_positives > 0 else 0.0

    print(f"--- EVALUATION REPORT ---")
    print(f"Accuracy: {class_acc:.2f}% | Avg IoU: {avg_iou:.4f}")

if __name__ == "__main__":
    main()