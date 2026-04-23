from monai.networks.nets import UNet
from typing import Tuple

def build_resunet3d(
    in_channels: int = 4, 
    out_channels: int = 5, 
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256), 
    strides: Tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2
) -> UNet:
    """Construye una arquitectura 3D Residual U-Net (ResUNet) usando MONAI.

    Args:
        in_channels: Número de canales de entrada (ej. modalidades MRI).
        out_channels: Número de canales de salida (clases de segmentación).
        channels: Secuencia de números de filtros para cada nivel.
        strides: Secuencia de strides para las convoluciones de bajada.
        num_res_units: Número de subunidades residuales por bloque. Un valor > 0 la convierte en ResUNet.

    Returns:
        Una instancia del modelo UNet de MONAI configurada para 3D con conexiones residuales.
    """
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
    )