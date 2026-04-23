from torch.nn import Module

from src.models.unet3d import build_unet3d
from src.models.resunet3d import build_resunet3d
from src.models.swin_unetr import build_swin_unetr
from src.models.dense_unet_plus import DenseUNetPlus3D
from src.models.segmamba import SegMamba


def get_model(
    model_name: str,
    in_channels: int = 4,
    out_channels: int = 5,
    use_checkpoint: bool = True,
) -> Module:
    """
    Devuelve una instancia del modelo solicitado.

    Args:
        model_name: Nombre del modelo a construir.
        in_channels: Número de canales de entrada.
        out_channels: Número de canales de salida.
        use_checkpoint: Indica si se usa gradient checkpointing en los modelos que lo soportan.

    Returns:
        Instancia del modelo correspondiente al nombre indicado.

    Raises:
        ValueError: Si el nombre del modelo no está soportado.
    """
    model_name = model_name.lower()

    if model_name == "unet3d":
        return build_unet3d(in_channels=in_channels, out_channels=out_channels)

    if model_name == "resunet3d":
        return build_resunet3d(in_channels=in_channels, out_channels=out_channels)

    if model_name == "swin_unetr":
        return build_swin_unetr(
            in_channels=in_channels,
            out_channels=out_channels,
            use_checkpoint=use_checkpoint,
        )

    if model_name == "dense_unet_plus":
        return DenseUNetPlus3D(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=32,
            growth_rate=16,
            block_layers=3,
        )

    if model_name == "segmamba":
        return SegMamba(
            in_chans=in_channels,
            out_chans=out_channels,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
        )

    raise ValueError(f"Modelo no soportado: {model_name}")
