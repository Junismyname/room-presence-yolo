import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Define the Simple YOLO Model
class SimpleYOLO(nn.Module):
    def __init__(self):
        super(SimpleYOLO, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # Apply Linear layers for Confidence and Bounding Box Predictions
        self.head_conf = nn.Linear(256, 1)
        self.head_box = nn.Linear(256, 4)
        
        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.leaky_relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(self.leaky_relu(self.batch_norm3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        
        conf_logits = self.head_conf(x)               
        box_coords = torch.sigmoid(self.head_box(x)) 
        
        return torch.cat((conf_logits, box_coords), dim=1)

# Pre-trained MobileNetV2-based YOLO Model
class MobileNetYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone[-4:].parameters():
            param.requires_grad = True
        self.flatten = nn.Flatten()
        self.shared_fc = nn.Sequential(
            nn.Linear(1280 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.conf_head = nn.Linear(256, 1)
        
        self.box_head = nn.Sequential(
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.shared_fc(x)
        
        conf = self.conf_head(x)
        box = self.box_head(x)

        return torch.cat((conf, box), dim=1)