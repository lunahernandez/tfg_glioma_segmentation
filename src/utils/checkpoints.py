from pathlib import Path

import torch
from torch.nn import Module
from torch.optim import Optimizer


def save_checkpoint(
    model: Module,
    optimizer: Optimizer | None,
    epoch: int,
    best_metric: float,
    save_path: str | Path,
) -> None:
    """Guarda un checkpoint del modelo y, opcionalmente, del optimizador.

    Nota:
        Implementación adaptada de un tutorial oficial de MONAI:
        https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb

    Args:
        model: Modelo a guardar.
        optimizer: Optimizador del entrenamiento.
        epoch: Número de época correspondiente al checkpoint.
        best_metric: Mejor valor de la métrica registrado.
        save_path: Ruta donde se guardará el checkpoint.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint guardado en: {save_path}")


def load_checkpoint(
    model: Module,
    optimizer: Optimizer | None,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[Module, Optimizer | None, int, float]:
    """Carga un checkpoint en el modelo y, opcionalmente, en el optimizador.

    Args:
        model: Modelo en el que se cargará el estado guardado.
        optimizer: Optimizador en el que se cargará el estado guardado.
        checkpoint_path: Ruta del archivo de checkpoint.
        device: Dispositivo en el que se cargará el checkpoint.

    Returns:
        Una tupla con el modelo, el optimizador, la época del checkpoint y
        el mejor valor de la métrica registrado.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", 0.0)
    return model, optimizer, epoch, best_metric
