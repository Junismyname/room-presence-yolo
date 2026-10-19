import torch
import torch.nn as nn

# Define the YOLO Loss Function
class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        
        # Force the model to care 5x more about the "Person" class
        self.bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([5.0]))

    def forward(self, predictions, targets):
        pred_conf = predictions[:, 0]
        target_conf = targets[:, 0]
        pred_boxes = predictions[:, 1:5]
        target_boxes = targets[:, 1:5]

        # Confidence Loss
        loss_conf = self.bce(pred_conf, target_conf)

        # Box Loss
        mask = target_conf.unsqueeze(1)
        raw_box_error = self.mse(pred_boxes, target_boxes) * mask
        
        num_active_boxes = target_conf.sum()
        
        if num_active_boxes > 0:
            loss_box = raw_box_error.sum() / num_active_boxes
        else:
            loss_box = torch.tensor(0.0, device=predictions.device)

        # Combine 
        total_loss = loss_conf + (loss_box * 5.0)
        
        return total_loss