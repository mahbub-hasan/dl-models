import torch
import torch.nn as nn
from project_configuration import ProjectConfiguration


class ImageDecoder(nn.Module):
    def __init__(self, config: ProjectConfiguration):
        super(ImageDecoder, self).__init__()
        self.config = config
        self.grid_size = config.image_size // config.patch_size
        self.conv_layer = nn.Conv2d(in_channels=config.image_embedding,
                                    out_channels=512, kernel_size=1)

        # up sampling process
        self.up_sampling_1 = nn.ConvTranspose2d(in_channels=512,
                                                out_channels=256,
                                                kernel_size=(4, 4),
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                output_padding=(0, 0),
                                                dilation=(1, 1))
        self.up_sampling_2 = nn.ConvTranspose2d(in_channels=256,
                                                out_channels=128,
                                                kernel_size=(4, 4),
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                output_padding=(0, 0),
                                                dilation=(1, 1))
        self.up_sampling_3 = nn.ConvTranspose2d(in_channels=128,
                                                out_channels=64,
                                                kernel_size=(4, 4),
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                output_padding=(0, 0),
                                                dilation=(1, 1))
        self.up_sampling_4 = nn.ConvTranspose2d(in_channels=64,
                                                out_channels=32,
                                                kernel_size=(4, 4),
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                output_padding=(0, 0),
                                                dilation=(1, 1))
        self.up_sampling_5 = nn.ConvTranspose2d(in_channels=32,
                                                out_channels=3,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                output_padding=(0, 0),
                                                dilation=(1, 1))

    def forward(self, image):
        batch_size, seq_len, embed_dim = image.shape
        # [barch_size, embed_dim, seq_len]
        x = image.transpose(1, 2)
        # [batch_size, embed_dim, grid_size, grid_size]
        x = x.view(batch_size, embed_dim, self.grid_size, self.grid_size)

        # apply first conv layer to reduce the channel
        x = self.conv_layer(x)
        x = self.up_sampling_5(self.up_sampling_4(self.up_sampling_3(self.up_sampling_2(self.up_sampling_1(x)))))

        return x
