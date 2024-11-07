from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Cryo_EM_Generator(nn.Module):
    """
    NetG DCGAN Generator. Outputs 64x64 images.
    """

    def __init__(
        self,
        z_dim=100,
        out_ch=3,
        norm_layer=nn.BatchNorm2d,
        final_activation=None,
        
    ):
        super().__init__()
        self.z_dim = z_dim
        self.out_ch = out_ch
        self.final_activation = final_activation

        self.net = nn.Sequential(
            # * Layer 1: 1x1
            nn.ConvTranspose2d(self.z_dim, 512, 4, 1, 0, bias=False),
            norm_layer(512),
            nn.ReLU(),
            # * Layer 2: 4x4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            norm_layer(256),
            nn.ReLU(),
            # * Layer 3: 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            norm_layer(128),
            nn.ReLU(),
            # * Layer 4: 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            norm_layer(64),
            nn.ReLU(),
            # * Layer 5: 32x32
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            norm_layer(32),
            nn.ReLU(),
            # * Layer 6: 64x64
            nn.ConvTranspose2d(32, self.out_ch, 4, 2, 1, bias=False),
            # * Output: 128x128
        )
        ####################################################### Dictionary For Layers
        self.layer_dict = {f'layer{i}': layer for i, layer in enumerate(self.net) if isinstance(layer, nn.ConvTranspose2d)}

    
    def forward(self, x, synthesize=False):
        if synthesize:
            # Skip the first layer if synthesize=True
            x = self.net[1:](x)  
        else:
            x = self.net(x)  
        return (
            x if self.final_activation is None else self.final_activation(x)
        )
    
    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.z_dim, 1, 1))
