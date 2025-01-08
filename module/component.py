import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from .data_format import (DataFormat, DataDim, DATA_FORMAT_DIM_INDEX,)

def rotate(
    x: torch.Tensor, angle: int, data_format: str = DataFormat.BCHW
    ) -> torch.Tensor:
    """Rotate images by 90 degrees clockwise. Can handle any 2D data format.
    Args:
        x (Tensor): Image or batch of images.
        angle (int): Clockwise rotation angle in multiples of 90.
        data_format (str, optional): Format of input image data, e.g. BCHW,
            HWC. Defaults to BCHW.
    Returns:
        Tensor: Copy of tensor with rotation applied.
    """
    dims = DATA_FORMAT_DIM_INDEX[data_format]
    h_dim = dims[DataDim.HEIGHT]
    w_dim = dims[DataDim.WIDTH]

    if angle == 0:
        return x
    elif angle == 90:
        return x.flip(w_dim).transpose(h_dim, w_dim)
    elif angle == 180:
        return x.flip(w_dim).flip(h_dim)
    elif angle == 270:
        return x.flip(h_dim).transpose(h_dim, w_dim)
    else:
        raise NotImplementedError("Must be rotation divisible by 90 degrees")

class Crop2d(nn.Module):
    """Crop input using slicing. Assumes BCHW data.
    Args:
        crop (Tuple[int, int, int, int]): Amounts to crop from each side of the image.
            Tuple is treated as [left, right, top, bottom]/
    """

    def __init__(self, crop: Tuple[int, int, int, int]):
        super().__init__()
        self.crop = crop
        assert len(crop) == 4

    def forward(self, x: Tensor) -> Tensor:
        (left, right, top, bottom) = self.crop
        x0, x1 = left, x.shape[-1] - right
        y0, y1 = top, x.shape[-2] - bottom
        return x[:, :, y0:y1, x0:x1]

class Shift2d(nn.Module):
    """Shift an image in either or both of the vertical and horizontal axis by first
    zero padding on the opposite side that the image is shifting towards before
    cropping the side being shifted towards.
    Args:
        shift (Tuple[int, int]): Tuple of vertical and horizontal shift. Positive values
            shift towards right and bottom, negative values shift towards left and top.
    """

    def __init__(self, shift: Tuple[int, int]):
        super().__init__()
        self.shift = shift
        vert, horz = self.shift
        y_a, y_b = abs(vert), 0
        x_a, x_b = abs(horz), 0
        if vert < 0:
            y_a, y_b = y_b, y_a
        if horz < 0:
            x_a, x_b = x_b, x_a
        # Order : Left, Right, Top Bottom
        self.pad = nn.ZeroPad2d((x_a, x_b, y_a, y_b))
        self.crop = Crop2d((x_b, x_a, y_b, y_a))
        self.shift_block = nn.Sequential(self.pad, self.crop)

    def forward(self, x: Tensor) -> Tensor:
        return self.shift_block(x)

class ShiftConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """Custom convolution layer as defined by Laine et al. for restricting the
        receptive field of a convolution layer to only be upwards. For a h × w kernel,
        a downwards offset of k = [h/2] pixels is used. This is applied as a k sized pad
        to the top of the input before applying the convolution. The bottom k rows are
        cropped out for output.
        """
        super(ShiftConv2d, self).__init__(*args, **kwargs)
        self.shift_size = (self.kernel_size[0] // 2, 0)
        # Use individual layers of shift for wrapping conv with shift
        shift = Shift2d(self.shift_size)
        self.pad = shift.pad
        self.crop = shift.crop

    def forward(self, x):
        x = self.pad(x)
        x = super(ShiftConv2d, self).forward(x)
        x = self.crop(x)
        return x
