import torch
import torch.nn as nn
from torch.nn import functional as F

class HybridLoss(nn.Module):
    def __init__(self, dice_weight=0.5, reduction='mean'):
        super(HybridLoss, self).__init__()
        self.dice_weight = dice_weight
        self.reduction = reduction

    def forward(self, input, target):
        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)

        # Dice Loss
        smooth = 1e-5
        input_sigmoid = torch.sigmoid(input)
        intersection = (input_sigmoid * target).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (input_sigmoid.sum() + target.sum() + smooth)

        total_loss = bce_loss + self.dice_weight * dice_loss
        return total_loss