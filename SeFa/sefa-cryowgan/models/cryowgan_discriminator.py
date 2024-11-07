import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from collections import OrderedDict


class Cryo_EM_Discriminator(nn.Module):
    def __init__(
        self, in_ch=3, norm_layer=nn.BatchNorm2d, final_activation=None
    ):
        super().__init__()
        self.in_ch = in_ch
        self.final_activation = final_activation

        self.net = nn.Sequential(
            # * 128x128
            nn.Conv2d(self.in_ch, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # * 64x64
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            norm_layer(64, affine=True),
            nn.LeakyReLU(0.2),
            # * 32x32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            norm_layer(128, affine=True),
            nn.LeakyReLU(0.2),
            # * 16x16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            norm_layer(256, affine=True),
            nn.LeakyReLU(0.2),
            # * 8x8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            norm_layer(512, affine=True),
            nn.LeakyReLU(0.2),
            # * 4x4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

        self.features_to_prob = nn.Sequential(
            #nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.net(x)
        #print(x.shape)
        x = x.view(batch_size, -1)
        #print(x.shape)
        return self.features_to_prob(x)
    