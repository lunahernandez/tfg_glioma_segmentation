from typing import Tuple

from monai.networks.nets import SwinUNETR


def build_swin_unetr(
    in_channels: int = 4,
    out_channels: int = 5,
    feature_size: int = 48,
    use_checkpoint: bool = True,
) -> SwinUNETR:
    """Construye un modelo SwinUNETR para segmentación volumétrica.

    Args:
        in_channels: Número de canales de entrada.
        out_channels: Número de canales de salida.
        feature_size: Dimensión de las características del embedding.
        use_checkpoint: Indica si se usa gradient checkpointing para ahorrar memoria.

    Returns:
        Una instancia del modelo SwinUNETR de MONAI para segmentación 3D.
    """
    return SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        spatial_dims=3,
    )
