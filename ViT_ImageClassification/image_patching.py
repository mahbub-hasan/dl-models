import torch
import torch.nn as nn
from project_config import ProjectConfig


class ImagePatch(nn.Module):
    def __init__(self, config: ProjectConfig):
        super(ImagePatch, self).__init__()
        self.config = config
        self.img_patch = nn.Conv2d(in_channels=config.image_channel, out_channels=config.embedding_dim,
                                   kernel_size=config.image_patch_size, stride=config.image_patch_size,
                                   padding='valid')
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x = self.flatten(self.img_patch(x))
        x = x.transpose(1, 2)  # [batch_size, (image_H/patch * image_w/patch), embedding]
        return x


class ImageEmbedding(nn.Module):
    def __init__(self, config: ProjectConfig):
        super(ImageEmbedding, self).__init__()
        self.config = config
        self.grid_size = (config.image_size // config.image_patch_size) ** 2
        # create an empty class token
        self.class_token = nn.Parameter(torch.randn(1, 1, config.embedding_dim), requires_grad=True)
        # create positional embedding
        self.positional_embedding = nn.Parameter(torch.randn(1, self.grid_size + 1, config.embedding_dim),
                                                 requires_grad=True)

    def forward(self, x):
        cls_token = self.class_token.expand(self.config.batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.positional_embedding + x
        return x


class ImageLinearProjection(nn.Module):
    def __init__(self, config: ProjectConfig):
        super(ImageLinearProjection, self).__init__()
        self.config = config
        self.image_patch = ImagePatch(config)
        self.image_embedding = ImageEmbedding(config)

    def forward(self, x):
        x = self.image_embedding(self.image_patch(x))
        return x
