# Create a custom Dataset class for YOLO
import os
import cv2
import torch
from torch.utils.data import Dataset

class RoomYOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.cache = {}
        
        # Get a list of all images in the folder
        self.imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Check if the item is already in cache
        if idx in self.cache:
            return self.cache[idx]
        
        # Load the Image
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # OpenCV loads images in BGR format, but PyTorch expects RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find the matching label file
        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(self.label_dir, label_name)
        
        # Parse the Label (Handling Negative Samples)
        # Target format: [is_person, x_center, y_center, width, height]
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                # YOLO format: class_id x_center y_center width height
                data = f.readline().strip().split()
                is_person = 1.0
                target = [is_person, float(data[1]), float(data[2]), float(data[3]), float(data[4])]
        else:
            # is_person = 0.0
            target = [0.0, 0.0, 0.0, 0.0, 0.0]
            
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Save the final tensors to the cache before returning them
        self.cache[idx] = (image, target_tensor)

        return image, target_tensor