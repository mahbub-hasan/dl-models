import torch
import torch.nn as nn
from project_config import ProjectConfig
from vit_encoder_block import ViTEncoderBlock


class Encoder(nn.Module):
    def __init__(self, config: ProjectConfig):
        super(Encoder, self).__init__()
        self.config = config
        self.encoder_layer = nn.ModuleList([ViTEncoderBlock(config=config) for _ in range(config.attention_layer)])

    def forward(self, x):
        attention_scores = []
        for encoder_layer in self.encoder_layer:
            x, attention_score = encoder_layer(x)
            attention_scores.append(attention_score)
        return x, attention_scores