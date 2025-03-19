import torch
import torch.nn as nn
from project_configuration import ProjectConfiguration
from encoder.seg_encoder import DoubleConv


class SegmentationDecoder(nn.Module):
    def __init__(self, config: ProjectConfiguration, in_channels: int, out_channels: int):
        super(SegmentationDecoder, self).__init__()
        self.config = config
        self.conv_transpose_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2,
                                                       kernel_size=2, stride=2)
        self.double_conv_layer = DoubleConv(config=config, in_channels=in_channels, out_channels=out_channels)

    def forward(self, image, skip_conn):
        x = self.conv_transpose_layer(image)
        x = torch.cat([skip_conn, x], dim=1)
        x = self.double_conv_layer(x)
        return x
