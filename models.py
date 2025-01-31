import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#====================================================================old=================================================================
class Generator_(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(Generator, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feature_sizes = (int(self.img_size[0] / 16), int(self.img_size[1] / 16))

        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 8 * dim * self.feature_sizes[0] * self.feature_sizes[1]),
            nn.ReLU()
        )

        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, self.img_size[2], 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        # Reshape
        x = x.view(-1, 8 * self.dim, self.feature_sizes[0], self.feature_sizes[1])
        # Return generated image
        return self.features_to_image(x)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))
        

class Discriminator_(nn.Module):
    def __init__(self, img_size, dim):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.image_to_features = nn.Sequential(
            #128
            nn.Conv2d(self.img_size[2], dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            #64
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            #Norm Layer
            #nn.LayerNorm([2 * dim, self.img_size[2]/2**2, self.img_size[2]/2**2]),
            nn.LeakyReLU(0.2),
            #32
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            #Norm Layer
            #nn.LayerNorm([4 * dim, self.img_size[2]/2**3, self.img_size[2]/2**3]),
            nn.LeakyReLU(0.2),
            #16
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            #8
            #nn.Sigmoid()
        )

        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        output_size = int(8 * dim * (img_size[0] / 16) * (img_size[1] / 16))
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)

#====================================================================end_of_old=================================================================
#NEW WITH GAUSSIAN FILTERS AND num_octaves

from collections import OrderedDict

class Generator(nn.Module):
    """
    NetG DCGAN Generator. Outputs 64x64 images.
    """

    def __init__(
        self,
        z_dim=100,
        first_channel_size=256,
        out_ch=1,                       #CHANGED TO 1 from 3

        norm_layer=nn.BatchNorm2d,
        final_activation=None,
        
    ):
        super().__init__()
        self.z_dim = z_dim
        self.out_ch = out_ch
        self.final_activation = final_activation
        self.fcs = first_channel_size #first channel size

            )
        self.net = nn.Sequential(
            # * Layer 1: 1x1
            nn.ConvTranspose2d(self.z_dim, self.fcs, 4, 1, 0, bias=False),
            norm_layer(self.fcs),
            nn.ReLU(),
            # * Layer 2: 4x4
            nn.ConvTranspose2d(self.fcs, self.fcs//2, 4, 2, 1, bias=False),
            norm_layer(self.fcs//2),
            nn.ReLU(),
            # * Layer 3: 8x8
            nn.ConvTranspose2d(self.fcs//2, self.fcs//(2**2), 4, 2, 1, bias=False),
            norm_layer(self.fcs//(2**2)),
            nn.ReLU(),
            # * Layer 4: 16x16
            nn.ConvTranspose2d(self.fcs//(2**2), self.fcs//(2**3), 4, 2, 1, bias=False),
            norm_layer(self.fcs//(2**3)),
            nn.ReLU(),
            # * Layer 5: 32x32
            nn.ConvTranspose2d(self.fcs//(2**3), self.fcs//(2**4), 4, 2, 1, bias=False),
            norm_layer(self.fcs//(2**4)),
            nn.ReLU(),
            # * Layer 6: 64x64
            nn.ConvTranspose2d(self.fcs//(2**4), self.fcs//(2**5), 4, 2, 1, bias=False),
            norm_layer(self.fcs//(2**5)),
            nn.ReLU(),
            # * Layer 7: 128x128
            nn.ConvTranspose2d(self.fcs//(2**5), self.out_ch, 4, 2, 1, bias=False),
            # * Output Layer 8: 256x256
        )


    def forward(self, x):
        x = self.net(x)
  
        return (
            x if self.final_activation is None else self.final_activation(x)
        )
    
    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.z_dim, 1, 1))

# for cryospin
from encoders import CNNEncoderVGG16, GaussianPyramid

class Discriminator(nn.Module):
    def __init__(
        self, in_ch=1, norm_layer=nn.BatchNorm2d, final_activation=None, num_octaves=4   #CHANGED IN_CH to 1 from 3
    ):
        super().__init__()
        self.in_ch = in_ch
        self.final_activation = final_activation
        self.num_octaves = num_octaves
        self.gaussian_filters = GaussianPyramid(
                kernel_size=11,
                kernel_variance=0.01,
                num_octaves=num_octaves,
                octave_scaling=10
            )
      
        
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
        x = self.gaussian_filters(x)
        x = self.net(x)
        #print(x.shape)
        x = x.view(batch_size, -1)
        #print(x.shape)
        return self.features_to_prob(x)
    
''' PREVIOUS ONES that we used
from collections import OrderedDict

class Generator(nn.Module):
    """
    NetG DCGAN Generator. Outputs 64x64 images.
    """

    def __init__(
        self,
        z_dim=100,
        out_ch=3,
        norm_layer=nn.BatchNorm2d,
        final_activation=None,
        wscale = 1
    ):
        super().__init__()
        self.z_dim = z_dim
        self.out_ch = out_ch
        self.final_activation = final_activation
        self.wscale = wscale

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

        if self.wscale-1:
            self.wscale = self.wscale/np.sqrt(4 * 4 * self.z_dim)

    def forward(self, x):
        #Scale first layers
        self.net[0].weight = nn.Parameter(self.net[0].weight*self.wscale)

        x = self.net(x)
        return (
            x if self.final_activation is None else self.final_activation(x)
        )
    
    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.z_dim, 1, 1))

class Discriminator(nn.Module):
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
    
'''
