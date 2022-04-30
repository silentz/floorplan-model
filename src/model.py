from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models.feature_extraction import create_feature_extractor


class SpatialContextualModule(nn.Module):

    def __init__(self, channels: int):
        super().__init__()

        self.conv_b1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv_b2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv_b3 = nn.Conv2d(in_channels=channels, out_channels=1,        kernel_size=1, stride=1, padding=0)

        self.conv_r1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv_r2 = nn.Conv2d(in_channels=channels, out_channels=1,        kernel_size=1, stride=1, padding=0)

    def forward(self, xb: torch.Tensor, xr: torch.Tensor) -> torch.Tensor:
        xb = F.relu(self.conv_b1(xb))
        xb = F.relu(self.conv_b2(xb))
        xb = torch.sigmoid(self.conv_b3(xb))

        xr = F.relu(self.conv_r1(xr))
        xr = torch.sigmoid(self.conv_r2(xr))

        # TODO: attention
        # TODO: direction aware kernels
        # TODO: attention 2
        # TODO: expand
        # TODO: concat
        # TODO: reduce



class Model(nn.Module):

    def __init__(self, pretrained_vgg: bool = True,
                       n_boundary_classes: int = 3):
        super().__init__()

        # init vgg encoder
        self.encoder = create_feature_extractor(
                model=torchvision.models.vgg16(pretrained=pretrained_vgg).features,
                return_nodes={'4': '256', '9': '128', '16': '64', '23': '32', '30': '16'},
            )

        #  boundary decoder
        ## block 1 (16 -> 32)
        self.b_01_upsample = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.b_01_encoder  = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.b_01_filter   = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        ## block 2 (32 -> 64)
        self.b_02_upsample = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.b_02_encoder  = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.b_02_filter   = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        ## block 3 (64 -> 128)
        self.b_03_upsample = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.b_03_encoder  = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.b_03_filter   = nn.Conv2d(in_channels=64,  out_channels=64, kernel_size=3, stride=1, padding=1)

        ## block 4 (128 -> 256)
        self.b_04_upsample = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.b_04_encoder  = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.b_04_filter   = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        ## finalize (256 -> 256)
        self.b_finalize    = nn.Conv2d(in_channels=32, out_channels=n_boundary_classes,
                                       kernel_size=3, stride=1, padding=1)

        # TODO: add room features


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        encoder_out = self.encoder(input)

        b_01_upsample = self.b_01_upsample(encoder_out['16'])
        b_01_encoder  = self.b_01_encoder(encoder_out['32'])
        b_01_filter   = self.b_01_filter(b_01_upsample + b_01_encoder)
        b_01_out      = F.relu(b_01_filter)

        b_02_upsample = self.b_02_upsample(b_01_out)
        b_02_encoder  = self.b_02_encoder(encoder_out['64'])
        b_02_filter   = self.b_02_filter(b_02_upsample + b_02_encoder)
        b_02_out      = F.relu(b_02_filter)

        b_03_upsample = self.b_03_upsample(b_02_out)
        b_03_encoder  = self.b_03_encoder(encoder_out['128'])
        b_03_filter   = self.b_03_filter(b_03_upsample + b_03_encoder)
        b_03_out      = F.relu(b_03_filter)

        b_04_upsample = self.b_04_upsample(b_03_out)
        b_04_encoder  = self.b_04_encoder(encoder_out['256'])
        b_04_filter   = self.b_04_filter(b_04_upsample + b_04_encoder)
        b_04_out      = F.relu(b_04_filter)

        b_out = self.b_finalize(b_04_out)
        b_out = F.interpolate(b_out, size=512, mode='nearest')

        return b_out

