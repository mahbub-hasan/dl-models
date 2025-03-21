import torch
from torch import nn
from project_configuration import ProjectConfiguration


class DiceAccuracy(nn.Module):
    def __init__(self, config: ProjectConfiguration):
        super(DiceAccuracy, self).__init__()
        self.config = config

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        pred_value = y_pred.clone()

        pred_value[pred_value > 0] = 1
        pred_value[pred_value <= 0] = 0

        intersection = abs(torch.sum(pred_value * y_true))
        union = abs(torch.sum(pred_value) + torch.sum(y_true))
        dice = (2. * intersection + self.config.eps) / (union + self.config.eps)

        return dice
