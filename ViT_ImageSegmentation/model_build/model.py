import torch
import torch.nn as nn
from project_configuration import ProjectConfiguration
from encoder.image_encoder import ImageEncoder
from decoder.image_decoder import DoubleConv, ImageDecoder


class BuildModel(nn.Module):
    def __init__(self, config: ProjectConfiguration):
        super(BuildModel, self).__init__()
        self.config = config
        self.grid_size = config.image_size // config.patch_size

        self.encoder = ImageEncoder(config=config)

        self.bottle_neck = DoubleConv(in_channels=config.image_embedding, out_channels=512)

        self.decoder_1 = ImageDecoder(config=config, in_channels=512, out_channels=256)
        self.decoder_2 = ImageDecoder(config=config, in_channels=256, out_channels=128)
        self.decoder_3 = ImageDecoder(config=config, in_channels=128, out_channels=64)
        self.decoder_4 = ImageDecoder(config=config, in_channels=64, out_channels=32)

        self.output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)

    def forward(self, x):
        # batch_size, grid_size*grid_size, embed_dim
        x = self.encoder(x)

        batch_size, seq_len, embed_dim = x.shape

        # transpose size [batch_size, embed_dim, grid_size*grid_size
        x = x.transpose(1, 2)

        # now create a view to split grid_size
        # new dimension [batch_size, embed_dim, grid_size, grid_size]
        x = x.view(batch_size, embed_dim, self.grid_size, self.grid_size)

        # change embed_dim from 768 to 512 (called bottleneck)
        x = self.bottle_neck(x)

        # now appy decoder and reconstruction back
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)

        # get final output [batch_size, channel (previously it was embed_dim), image_size, image_size
        x = self.output(x)

        return x