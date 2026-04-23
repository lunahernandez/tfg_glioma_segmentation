import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Configura las semillas para garantizar la reproducibilidad en
    los módulos random, numpy y torch. Además, ajusta las configuraciones
    de cuDNN para asegurar resultados deterministas.

    Args:
        seed: Valor de la semilla a usar.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
