import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import RoomYOLODataset, SimpleYOLO, YOLOLoss, MobileNetYOLO

import argparse
from src.model import SimpleYOLO, MobileNetYOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'mobilenet'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the chosen model
    if args.model == 'mobilenet':
        print("Initializing MobileNetV2 Backbone...")
        model = MobileNetYOLO().to(device)
        weights_filename = 'room_yolo_weights_mobilenet.pth'
    else:
        print("Initializing Custom SimpleYOLO...")
        model = SimpleYOLO().to(device)
        weights_filename = 'room_yolo_weights_simple.pth'

    # Build the transform list dynamically
    transform_list = [
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        #transforms.RandomAffine(degrees=5, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.RandomErasing(p=0.5, scale=(0.05, 0.2), ratio=(0.5, 2.0), value=0)
    ]
    
    # MobileNet requires strict ImageNet color normalization
    if args.model == 'mobilenet':
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    my_transforms = transforms.Compose(transform_list)

    dataset = RoomYOLODataset(img_dir='dataset_frames', label_dir='dataset_labels', transform=my_transforms)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

    if args.model == 'mobilenet':
        model = MobileNetYOLO().to(device)
    else:
        model = SimpleYOLO().to(device)
    criterion = YOLOLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 200

    for epoch in range(epochs):
        model.train()
        total_batch_loss = 0
        
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()
            total_batch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_batch_loss / len(dataloader):.4f}")

    # Save model weights to root directory
    save_path = os.path.join(os.path.dirname(__file__), '..', weights_filename)
    torch.save(model.state_dict(), save_path)
    print(f"Training Complete. Weights saved to {save_path}")

if __name__ == "__main__":
    main()