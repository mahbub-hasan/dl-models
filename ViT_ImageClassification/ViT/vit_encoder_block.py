import torch
import torch.nn as nn
from project_config import ProjectConfig
from vit_attention import ViTMultiHeadAttention
from vit_multilayer_perceptron import ViTMultilayerPerceptron


class ViTEncoderBlock(nn.Module):
    def __init__(self, config: ProjectConfig):
        super(ViTEncoderBlock, self).__init__()
        self.config = config
        self.mha = ViTMultiHeadAttention(config=config)
        self.mlp = ViTMultilayerPerceptron(config=config)
        self.layer_norm = nn.LayerNorm(normalized_shape=config.embedding_dim, eps=config.eps)

    def forward(self, x):
        res_connection = x
        x, attention_score = self.mha(self.layer_norm(x))
        x = x + res_connection
        res_connection = x
        x = self.mlp(self.layer_norm(x))
        x = x + res_connection
        return x, attention_score