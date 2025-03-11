import torch
import torch.nn as nn
from project_config import ProjectConfig
from vit_encoder import Encoder
from image_patching import ImageLinearProjection


class ViTModel(nn.Module):
    def __init__(self, config: ProjectConfig, number_of_class: int):
        super(ViTModel, self).__init__()
        self.config = config
        self.image_patching = ImageLinearProjection(config=config)
        self.encoder = Encoder(config=config)
        self.classification = nn.Sequential(
            nn.LayerNorm(normalized_shape=config.embedding_dim, eps=config.eps),
            nn.Linear(in_features=config.embedding_dim, out_features=number_of_class)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.image_patching(x)
        x, attention_scores = self.encoder(x)
        x = self.classification(x[:, 0, :]) # only cls_token hold the classification
        x = self.softmax(x)

        return x, attention_scores