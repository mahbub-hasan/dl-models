import torch
import torch.nn as nn
from project_configuration import ProjectConfiguration


class ImagePatching(nn.Module):
    def __init__(self, config: ProjectConfiguration):
        super(ImagePatching, self).__init__()
        self.config = config
        self.img_patch = nn.Conv2d(in_channels=config.image_channel,
                                   out_channels=config.image_embedding,
                                   kernel_size=config.patch_size,
                                   stride=config.patch_size)

    def forward(self, x):
        x = self.img_patch(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
        return x


class ImageEmbedding(nn.Module):
    def __init__(self, config: ProjectConfiguration):
        super(ImageEmbedding, self).__init__()
        self.config = config
        self.number_of_patch = (config.image_size//config.patch_size)**2
        self.image_patch = ImagePatching(config=config)
        # create cls_token
        self.class_token = nn.Parameter(torch.randn(size=(1, 1, config.image_embedding)),
                                        requires_grad=True)
        # create positional embedding
        self.pos_embd = nn.Parameter(torch.randn(size=(1, self.number_of_patch+1, config.image_embedding)),
                                     requires_grad=True)

    def forward(self, x):
        x = self.image_patch(x)
        cls_token = self.class_token.expand(size=(self.config.batch_size, -1, -1))
        x = torch.cat([cls_token, x], dim=1)
        x = self.pos_embd + x
        return x


