from torch.nn import Module

from src.models.unet3d import build_unet3d
from src.models.resunet3d import build_resunet3d
from src.models.swin_unetr import build_swin_unetr
from src.models.segmamba import SegMamba


def get_model(
    model_name: str,
    in_channels: int = 4,
    out_channels: int = 5,
    use_checkpoint: bool = True,
) -> Module:
    """
    Returns an instance of the requested model.

    Args:
        model_name: Name of the model to build.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        use_checkpoint: Indicates whether to use gradient checkpointing in models that support it.

    Returns:
        Instance of the model corresponding to the indicated name.

    Raises:
        ValueError: If the model name is not supported.
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

    if model_name == "segmamba":
        return SegMamba(
            in_chans=in_channels,
            out_chans=out_channels,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
        )

    raise ValueError(f"Unsupported model: {model_name}")