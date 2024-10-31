import numpy as np
import torch
from torch import Tensor
from typing import Tuple
import torch.nn as nn

### Ideal low-pass

def _get_center_distance(size: Tuple[int], device: str = 'cpu') -> Tensor:
    """Compute the distance of each matrix element to the center.

    Args:
        size (Tuple[int]): [m, n].
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [m, n].
    """
    m, n = size
    i_ind = torch.tile(
                torch.tensor([[[i]] for i in range(m)], device=device),
                dims=[1, n, 1]).float()  # [m, n, 1]
    j_ind = torch.tile(
                torch.tensor([[[i] for i in range(n)]], device=device),
                dims=[m, 1, 1]).float()  # [m, n, 1]
    ij_ind = torch.cat([i_ind, j_ind], dim=-1)  # [m, n, 2]
    ij_ind = ij_ind.reshape([m * n, 1, 2])  # [m * n, 1, 2]
    center_ij = torch.tensor(((m - 1) / 2, (n - 1) / 2), device=device).reshape(1, 2)
    center_ij = torch.tile(center_ij, dims=[m * n, 1, 1])
    dist = torch.cdist(ij_ind, center_ij, p=2).reshape([m, n])
    return dist


def _get_ideal_weights(size: Tuple[int], D0: int, lowpass: bool = True, device: str = 'cpu') -> Tensor:
    """Get H(u, v) of ideal bandpass filter.

    Args:
        size (Tuple[int]): [H, W].
        D0 (int): The cutoff frequency.
        lowpass (bool): True for low-pass filter, otherwise for high-pass filter. Defaults to True.
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [H, W].
    """
    center_distance = _get_center_distance(size, device)
    center_distance[center_distance > D0] = -1
    center_distance[center_distance != -1] = 1
    if lowpass is True:
        center_distance[center_distance == -1] = 0
    else:
        center_distance[center_distance == 1] = 0
        center_distance[center_distance == -1] = 1
    return center_distance


def _to_freq(image: Tensor) -> Tensor:
    """Convert from spatial domain to frequency domain.

    Args:
        image (Tensor): [B, C, H, W].

    Returns:
        Tensor: [B, C, H, W]
    """
    img_fft = torch.fft.fft2(image)
    img_fft_shift = torch.fft.fftshift(img_fft)
    return img_fft_shift


def _to_space(image_fft: Tensor) -> Tensor:
    """Convert from frequency domain to spatial domain.

    Args:
        image_fft (Tensor): [B, C, H, W].

    Returns:
        Tensor: [B, C, H, W].
    """
    img_ifft_shift = torch.fft.ifftshift(image_fft)
    img_ifft = torch.fft.ifft2(img_ifft_shift)
    img = img_ifft.real.clamp(0, 1)
    return img

def ideal_bandpass(image: Tensor, D0: int, lowpass: bool = True) -> Tensor:
    """Low-pass filter for images.

    Args:
        image (Tensor): [B, C, H, W].
        D0 (int): Cutoff frequency.
        lowpass (bool): True for low-pass filter, otherwise for high-pass filter. Defaults to True.

    Returns:
        Tensor: [B, C, H, W].
    """
    img_fft = _to_freq(image)
    weights = _get_ideal_weights(img_fft.shape[-2:], D0=D0, lowpass=lowpass, device=image.device)
    img_fft = img_fft * weights
    img = _to_space(img_fft)
    return img

#### Butterworth

def _get_butterworth_weights(size: Tuple[int], D0: int, n: int, device: str = 'cpu') -> Tensor:
    """Get H(u, v) of Butterworth filter.

    Args:
        size (Tuple[int]): [H, W].
        D0 (int): The cutoff frequency.
        n (int): Order of Butterworth filters.
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [H, W].
    """
    center_distance = _get_center_distance(size=size, device=device)
    weights = 1 / (1 + torch.pow(center_distance / D0, 2 * n))
    return weights


def butterworth(image: Tensor, D0: int, n: int) -> Tensor:
    """Butterworth low-pass filter for images.

    Args:
        image (Tensor): [B, C, H, W].
        D0 (int): Cutoff frequency.
        n (int): Order of the Butterworth low-pass filter.

    Returns:
        Tensor: [B, C, H, W].
    """
    img_fft = _to_freq(image)
    weights = _get_butterworth_weights(image.shape[-2:], D0, n, device=image.device)
    img_fft = weights * img_fft
    img = _to_space(img_fft)
    return img


#### Gaussian


def _get_gaussian_weights(size: Tuple[int], D0: float, device: str = 'cpu') -> Tensor:
    """Get H(u, v) of Gaussian filter.

    Args:
        size (Tuple[int]): [H, W].
        D0 (float): The cutoff frequency.
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [H, W].
    """
    center_distance = _get_center_distance(size=size, device=device)
    weights = torch.exp(- (torch.square(center_distance) / (2 * D0 ** 2)))
    return weights


def gaussian(image: Tensor, D0: float, weights=[]) -> Tensor:
    """Gaussian low-pass filter for images.

    Args:
        image (Tensor): [B, C, H, W].
        D0 (int): Cutoff frequency.

    Returns:
        Tensor: [B, C, H, W].
    """
    if not len(weights):
        weights = _get_gaussian_weights(image.shape[-2:], D0=D0, device=image.device)
    image_fft = _to_freq(image)
    image_fft = image_fft * weights
    image = _to_space(image_fft)
    return image

def normalize_array(arr, min_value=None, max_value=None):
    divider = (arr.max() - arr.min())
    if not divider:
        return np.zeros(arr.shape)
    normalized_array = (arr - arr.min()) / (arr.max() - arr.min())  # Normalize to 0-1
    if max_value or min_value:
        normalized_array = normalized_array * (max_value - min_value) + min_value  # Scale to min_value-max_value
    return normalized_array

def normalize_tensor(tensor):
    t_min, t_max = torch.amin(tensor, dim=(-1, -2), keepdim=True), torch.amax(tensor, dim=(-1, -2), keepdim=True)
    divider = t_max - t_min
    #Normalize tensor
    normalized_tensor = (tensor - t_min) / divider  # Normalize to 0-1
    return normalized_tensor

import torch.nn as nn

# FROM ZOO GAN gp

def init_weight(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        # nn.init.xavier_normal_(m.weight)
        # nn.init.kaiming_uniform_(m.weight)
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            if m.bias.data is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


class PixelNorm(nn.Module):
    """
    PixelNorm PixelNorm from PG GAN
    thanks https://github.com/facebookresearch/pytorch_GAN_zoo/blob/b75dee40918caabb4fe7ec561522717bf096a8cb/models/networks/custom_layers.py#L9

    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x ** 2).mean(dim=1, keepdim=True) + epsilon).rsqrt())


class IdentityLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, affine: bool = True):
        super().__init__()
        self.norm = nn.GroupNorm(num_channels, num_channels, affine=affine)

    def forward(self, x):
        return self.norm(x)