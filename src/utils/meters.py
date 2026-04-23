class AverageMeter:
    """Calcula el promedio de una métrica.

    Nota:
        Implementación adaptada de un tutorial oficial de MONAI:
        https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb

    Attributes:
        val: Último valor registrado.
        avg: Promedio acumulado.
        sum: Suma acumulada de los valores.
        count: Número total de elementos acumulados.
    """
    def __init__(self) -> None:
        """Inicializa la clase y reinicia sus atributos."""
        self.reset()

    def reset(self) -> None:
        """Reinicia los valores de los atributos."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Actualiza los valores a partir de un nuevo registro.

        Args:
            val: Nuevo valor de la métrica.
            n: Número de elementos asociados a ese valor.
        """
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
