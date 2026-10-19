import sys
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import RoomYOLODataset
from src.model import SimpleYOLO

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
    
    test_dataset = RoomYOLODataset(img_dir='test_frames', label_dir='test_labels', transform=my_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = SimpleYOLO().to(device)
    weights_path = os.path.join(os.path.dirname(__file__), '..', 'room_yolo_weights.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    correct_classifications = 0
    total_iou = 0.0
    true_positives = 0

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

    total_images = len(test_dataset)
    class_acc = (correct_classifications / total_images) * 100
    avg_iou = (total_iou / true_positives) if true_positives > 0 else 0.0

    print(f"--- EVALUATION REPORT ---")
    print(f"Accuracy: {class_acc:.2f}% | Avg IoU: {avg_iou:.4f}")

if __name__ == "__main__":
    main()