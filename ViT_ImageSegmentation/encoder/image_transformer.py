import math

import torch
import torch.nn as nn
from project_configuration import ProjectConfiguration


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ProjectConfiguration):
        super(MultiHeadAttention, self).__init__()
        self.config = config

        self.query = nn.Linear(in_features=config.image_embedding, out_features=config.image_embedding)
        self.key = nn.Linear(in_features=config.image_embedding, out_features=config.image_embedding)
        self.value = nn.Linear(in_features=config.image_embedding, out_features=config.image_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()

        assert self.config.image_embedding % self.config.attention_head == 0

        query_state = self.query(x).view(batch_size, seq_len, self.config.attention_head,
                                         self.config.image_embedding//self.config.attention_head).transpose(1, 2)
        key_state = self.key(x).view(batch_size, seq_len, self.config.attention_head,
                                     self.config.image_embedding//self.config.attention_head).transpose(1, 2)
        value_state = self.value(x).view(batch_size, seq_len, self.config.attention_head,
                                         self.config.image_embedding//self.config.attention_head).transpose(1, 2)

        attention_score = torch.softmax(torch.matmul(query_state, key_state.transpose(2, 3))/math.sqrt(self.config.image_embedding), dim=-1)

        attention_output = torch.matmul(value_state, attention_score).reshape(batch_size, seq_len, embed_dim).contiguous()

        return attention_output


class MultiLayerPerceptron(nn.Module):
    def __init__(self, config: ProjectConfiguration):
        super(MultiLayerPerceptron, self).__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(in_features=config.image_embedding, out_features=config.hidden_layer),
            nn.GELU(),
            nn.Linear(in_features=config.hidden_layer, out_features=config.image_embedding),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.mlp(x)


class ViTImageTransformer(nn.Module):
    def __init__(self, config: ProjectConfiguration):
        super(ViTImageTransformer, self).__init__()
        self.config = config
        self.layer_norm = nn.LayerNorm(normalized_shape=config.image_embedding, eps=config.eps)
        self.multihead_attention = MultiHeadAttention(config=config)
        self.multilayer_perceptron = MultiLayerPerceptron(config=config)

    def forward(self, image):
        res_conn = image
        image = self.multihead_attention(self.layer_norm(image))
        image = res_conn + image

        res_conn = image
        image = self.multilayer_perceptron(self.layer_norm(image))
        return res_conn + image


class TransformerEncoder(nn.Module):
    def __init__(self, config: ProjectConfiguration):
        super(TransformerEncoder, self).__init__()
        self.config = config

        self.encoder_layers = nn.ModuleList([ViTImageTransformer(config=config) for _ in range(config.attention_layer)])

    def forward(self, image):
        for encoder_layer in self.encoder_layers:
            image = encoder_layer(image)

        return image
