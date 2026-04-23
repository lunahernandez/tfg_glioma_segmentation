from typing import Tuple

from monai.networks.nets import UNet


def build_unet3d(
    in_channels: int = 4,
    out_channels: int = 5,
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: Tuple[int, ...] = (2, 2, 2, 2),
) -> UNet:
    """Construye un modelo U-Net 3D para segmentación volumétrica.

    Args:
        in_channels: Número de canales de entrada.
        out_channels: Número de canales de salida.
        channels: Secuencia con el número de canales de características en cada nivel.
        strides: Secuencia con los factores de reducción espacial entre niveles.

    Returns:
        Una instancia del modelo U-Net de MONAI para segmentación 3D.
    """
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
    )
