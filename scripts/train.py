import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import RoomYOLODataset, SimpleYOLO, YOLOLoss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    my_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])

    dataset = RoomYOLODataset(img_dir='dataset_frames', label_dir='dataset_labels', transform=my_transforms)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

    model = SimpleYOLO().to(device)
    criterion = YOLOLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 500

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
    save_path = os.path.join(os.path.dirname(__file__), '..', 'room_yolo_weights.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Training Complete. Weights saved to {save_path}")

if __name__ == "__main__":
    main()