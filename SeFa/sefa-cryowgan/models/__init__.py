# python3.7
"""Collects all available models together."""
import torch
import torch.nn as nn


from .model_zoo import MODEL_ZOO
from .pggan_generator import PGGANGenerator
from .pggan_discriminator import PGGANDiscriminator
from .stylegan_generator import StyleGANGenerator
from .stylegan_discriminator import StyleGANDiscriminator
from .stylegan2_generator import StyleGAN2Generator
from .stylegan2_discriminator import StyleGAN2Discriminator
from .cryowgan_generator import Cryo_EM_Generator
from .cryowgan_discriminator import Cryo_EM_Discriminator

__all__ = [
    'MODEL_ZOO','Cryo_EM_Generator', 'Cryo_EM_Discriminator','PGGANGenerator', 'PGGANDiscriminator', 'StyleGANGenerator',
    'StyleGANDiscriminator', 'StyleGAN2Generator', 'StyleGAN2Discriminator',
    'build_generator', 'build_discriminator', 'build_model'
]

_GAN_TYPES_ALLOWED = ['pggan', 'stylegan', 'stylegan2','cryowgan']
_MODULES_ALLOWED = ['generator', 'discriminator']

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, affine: bool = True):
        super().__init__()
        self.norm = nn.GroupNorm(num_channels, num_channels, affine=affine)

    def forward(self, x):
        return self.norm(x)


def build_generator(gan_type, resolution, **kwargs):
    """Builds generator by GAN type.

    Args:
        gan_type: GAN type to which the generator belong.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the generator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if gan_type not in _GAN_TYPES_ALLOWED:
        raise ValueError(f'Invalid GAN type: `{gan_type}`!\n'
                         f'Types allowed: {_GAN_TYPES_ALLOWED}.')

    if gan_type == 'pggan':
        return PGGANGenerator(resolution, **kwargs)
    if gan_type == 'stylegan':
        return StyleGANGenerator(resolution, **kwargs)
    if gan_type == 'stylegan2':
        return StyleGAN2Generator(resolution, **kwargs)
    if gan_type == 'cryowgan':
        return Cryo_EM_Generator(z_dim=100,
            out_ch=1,#for grayscale
            norm_layer=LayerNorm2d,
            final_activation=torch.tanh)
    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def build_discriminator(gan_type, resolution, **kwargs):
    """Builds discriminator by GAN type.

    Args:
        gan_type: GAN type to which the discriminator belong.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the discriminator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if gan_type not in _GAN_TYPES_ALLOWED:
        raise ValueError(f'Invalid GAN type: `{gan_type}`!\n'
                         f'Types allowed: {_GAN_TYPES_ALLOWED}.')

    if gan_type == 'pggan':
        return PGGANDiscriminator(resolution, **kwargs)
    if gan_type == 'stylegan':
        return StyleGANDiscriminator(resolution, **kwargs)
    if gan_type == 'stylegan2':
        return StyleGAN2Discriminator(resolution, **kwargs)
    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def build_model(gan_type, module, resolution, **kwargs):
    """Builds a GAN module (generator/discriminator/etc).

    Args:
        gan_type: GAN type to which the model belong.
        module: GAN module to build, such as generator or discrimiantor.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the discriminator.

    Raises:
        ValueError: If the `module` is not supported.
        NotImplementedError: If the `module` is not implemented.
    """
    if module not in _MODULES_ALLOWED:
        raise ValueError(f'Invalid module: `{module}`!\n'
                         f'Modules allowed: {_MODULES_ALLOWED}.')

    if module == 'generator':
        return build_generator(gan_type, resolution, **kwargs)
    if module == 'discriminator':
        return build_discriminator(gan_type, resolution, **kwargs)
    raise NotImplementedError(f'Unsupported module `{module}`!')


def parse_gan_type(module):
    """Parses GAN type of a given module.

    Args:
        module: The module to parse GAN type from.

    Returns:
        A string, indicating the GAN type.

    Raises:
        ValueError: If the GAN type is unknown.
    """
    if isinstance(module, (PGGANGenerator, PGGANDiscriminator)):
        return 'pggan'
    if isinstance(module, (StyleGANGenerator, StyleGANDiscriminator)):
        return 'stylegan'
    if isinstance(module, (StyleGAN2Generator, StyleGAN2Discriminator)):
        return 'stylegan2'
    if isinstance(module, (Cryo_EM_Generator,Cryo_EM_Discriminator)):
        return 'cryowgan'
    raise ValueError(f'Unable to parse GAN type from type `{type(module)}`!')
