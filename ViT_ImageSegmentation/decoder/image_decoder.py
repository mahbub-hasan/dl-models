import torch
import torch.nn as nn
from project_configuration import ProjectConfiguration


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class ImageDecoder(nn.Module):
    def __init__(self, config: ProjectConfiguration, in_channels: int, out_channels: int):
        super(ImageDecoder, self).__init__()
        self.config = config
        self.conv_up = nn.ConvTranspose2d(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=2,
                                          stride=2)
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, image):
        x = self.conv_up(image)
        x = self.double_conv(x)

        return x
