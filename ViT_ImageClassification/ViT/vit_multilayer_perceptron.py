import torch
import torch.nn as nn
from project_config import ProjectConfig


class ViTMultilayerPerceptron(nn.Module):
    def __init__(self, config: ProjectConfig):
        super(ViTMultilayerPerceptron, self).__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(in_features=config.embedding_dim, out_features=config.hidden_layer),
            nn.GELU(),
            nn.Linear(in_features=config.hidden_layer, out_features=config.embedding_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.mlp(x)