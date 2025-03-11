import torch
import torch.nn as nn
from encoder.image_transformer import TransformerEncoder
from encoder.image_embedding import ImageEmbedding
from project_configuration import ProjectConfiguration


class ImageEncoder(nn.Module):
    def __init__(self, config: ProjectConfiguration):
        super(ImageEncoder, self).__init__()
        self.config = config
        self.img_embed = ImageEmbedding(config=config)
        self.transformer_encoder = TransformerEncoder(config=config)
        self.segment_encoder = nn.Sequential(
            nn.LayerNorm(normalized_shape=config.image_embedding, eps=config.eps)
        )

    def forward(self, images):
        images = self.img_embed(images)
        transformer_encoded_image = self.transformer_encoder(images)
        segment_encoded_image = self.segment_encoder(transformer_encoded_image[:, 1:, :])
        return segment_encoded_image
