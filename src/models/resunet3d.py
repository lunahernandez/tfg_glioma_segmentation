from monai.networks.nets import UNet
from typing import Tuple


def build_resunet3d(
    in_channels: int = 4, 
    out_channels: int = 5, 
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256), 
    strides: Tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2
) -> UNet:
    """Builds a 3D Residual U-Net (ResUNet) architecture using MONAI.

    Args:
        in_channels: Number of input channels (e.g., MRI modalities).
        out_channels: Number of output channels (segmentation classes).
        channels: Sequence of the number of filters for each level.
        strides: Sequence of strides for the downsampling convolutions.
        num_res_units: Number of residual subunits per block. A value > 0 makes it a ResUNet.

    Returns:
        An instance of the MONAI UNet model configured for 3D with residual connections.
    """
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
    )
