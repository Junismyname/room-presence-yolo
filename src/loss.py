import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou_loss

# Define the YOLO Loss Function
class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        
        # Force the model to care 5x more about the "Person" class
        self.bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([2.0]))

    def forward(self, predictions, targets):
        pred_conf = predictions[:, 0]
        target_conf = targets[:, 0]
        pred_boxes = predictions[:, 1:5]
        target_boxes = targets[:, 1:5]

        # Confidence Loss
        loss_conf = self.bce(pred_conf, target_conf)

        # Box Loss
        has_person_mask = (target_conf == 1.0)
        
        loss_box = torch.tensor(0.0, device=predictions.device)
        
        if has_person_mask.sum() > 0:
            # Filter to only the positive frames
            pred_box_pos = pred_boxes[has_person_mask]
            target_box_pos = target_boxes[has_person_mask]

            # Convert [x_center, y_center, w, h] to [x1, y1, x2, y2] for GIoU
            pred_x1y1 = pred_box_pos[:, :2] - (pred_box_pos[:, 2:] / 2)
            pred_x2y2 = pred_box_pos[:, :2] + (pred_box_pos[:, 2:] / 2)
            pred_xyxy = torch.cat([pred_x1y1, pred_x2y2], dim=1)

            target_x1y1 = target_box_pos[:, :2] - (target_box_pos[:, 2:] / 2)
            target_x2y2 = target_box_pos[:, :2] + (target_box_pos[:, 2:] / 2)
            target_xyxy = torch.cat([target_x1y1, target_x2y2], dim=1)

            # 3. Calculate GIoU Loss (Directly optimizes overlap and shape alignment)
            loss_box = generalized_box_iou_loss(pred_xyxy, target_xyxy, reduction='mean')

        # Combine losses (Box loss gets a heavier multiplier to force tight bounds)
        total_loss = loss_conf + (5.0 * loss_box)
        return total_loss