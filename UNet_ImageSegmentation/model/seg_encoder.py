import torch
import torch.nn as nn
from project_configuration import ProjectConfiguration


class DoubleConv(nn.Module):
    def __init__(self, config: ProjectConfiguration, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.config = config
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        print(x.shape)
        return x


class SegmentationEncoder(nn.Module):
    def __init__(self, config: ProjectConfiguration, in_channels: int, out_channels: int):
        super(SegmentationEncoder, self).__init__()
        self.config = config
        self.double_conv_layer = DoubleConv(config=config, in_channels=in_channels, out_channels=out_channels)
        self.max_pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_conn = self.double_conv_layer(x)
        x = self.max_pool_layer(skip_conn)
        return x, skip_conn
