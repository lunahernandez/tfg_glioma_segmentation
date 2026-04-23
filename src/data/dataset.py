import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def get_case_data(case_dir: Path, include_label: bool = True) -> Optional[Dict]:
    """Obtiene información de un caso a partir de su directorio.
    
    Args:
        case_dir: Ruta a la carpeta del caso.
        include_label: Indica si se debe incluir la máscara de segmentación.

    Returns:
        Un diccionario con el identificador del caso, las rutas de las imágenes
        y, opcionalmente, la ruta de la etiqueta. Devuelve ``None`` si falta
        algún archivo.
    """
    case_id = case_dir.name

    image_paths = []
    for seq in ["t1c", "t1n", "t2f", "t2w"]:
        image_path = case_dir / f"{case_id}-{seq}.nii.gz"
        if not image_path.exists():
            return None
        image_paths.append(str(image_path))

    case_data = {
        "case_id": case_id,
        "image": image_paths,
    }

    if include_label:
        label_path = case_dir / f"{case_id}-seg.nii.gz"
        if not label_path.exists():
            return None
        case_data["label"] = str(label_path)
        case_data["label_path"] = str(label_path)

    return case_data


def get_cases_from_dirs(directories: List[Path], include_label: bool = True) -> List[Dict]:
    """Obtiene la información de los casos a partir de una lista de directorios.
    
    Args:
        directories: Lista de rutas a directorios que contienen carpetas de casos.
        include_label: Indica si se debe incluir la máscara de segmentación.

    Returns:
        Una lista de diccionarios con la información de cada caso.
    """
    cases = []
    for directory in directories:
        if not directory.exists():
            continue

        for case_dir in sorted(directory.iterdir()):
            if not case_dir.is_dir():
                continue

            case_data = get_case_data(case_dir, include_label=include_label)
            if case_data is not None:
                cases.append(case_data)

    return cases


def split_train_val_test(
    cases: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Divide los casos en entrenamiento, validación y prueba.

    Args:
        cases: Lista de casos a dividir.
        train_ratio: Proporción de casos de entrenamiento.
        val_ratio: Proporción de casos de validación.
        test_ratio: Proporción de casos de prueba.
        seed: Semilla usada para barajar los casos.

    Returns:
        Una tupla con las listas de casos de entrenamiento, validación y prueba.

    Raises:
        ValueError: Si la suma de las proporciones no es 1.0.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio debe sumar 1.0")

    random_generator = random.Random(seed)
    shuffled_cases = cases.copy()
    random_generator.shuffle(shuffled_cases)

    n = len(shuffled_cases)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_cases = shuffled_cases[:n_train]
    val_cases = shuffled_cases[n_train:n_train + n_val]
    test_cases = shuffled_cases[n_train + n_val:]

    return train_cases, val_cases, test_cases
