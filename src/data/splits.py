import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def save_split(
    train_cases: List[Dict],
    val_cases: List[Dict],
    test_cases: List[Dict],
    save_path: str | Path,
    fold: Optional[int] = None,
) -> None:
    """Guarda en un archivo JSON la partición de un fold.

    Args:
        train_cases: Lista de casos de entrenamiento.
        val_cases: Lista de casos de validación.
        test_cases: Lista de casos de prueba.
        save_path: Ruta donde se guardará el archivo JSON.
        fold: Identificador del fold.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    split_dict = {
        "fold": fold,
        "train": [case["case_id"] for case in train_cases],
        "val": [case["case_id"] for case in val_cases],
        "test": [case["case_id"] for case in test_cases],
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2)


def filter_cases_by_ids(cases: List[Dict], case_ids: List[str]) -> List[Dict]:
    """Filtra los casos cuyo identificador esté en una lista dada.

    Args:
        cases: Lista de casos.
        case_ids: Lista de identificadores de caso que se quieren conservar.

    Returns:
        Lista de casos filtrados.
    """
    case_ids = set(case_ids)
    return [case for case in cases if case["case_id"] in case_ids]


def split_train_val(
    cases: List[Dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Divide los casos en entrenamiento y validación.

    Args:
        cases: Lista de casos a dividir.
        val_ratio: Proporción de casos de validación.
        seed: Semilla usada para barajar los casos.

    Returns:
        Una tupla con las listas de entrenamiento y validación.

    Raises:
        ValueError: Si ``val_ratio`` no está entre 0 y 1.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio debe estar entre 0 y 1.")

    random_generator = random.Random(seed)
    shuffled_cases = cases.copy()
    random_generator.shuffle(shuffled_cases)

    n_val = max(1, int(len(shuffled_cases) * val_ratio))
    n_val = min(n_val, len(shuffled_cases) - 1) if len(shuffled_cases) > 1 else 0

    val_cases = shuffled_cases[:n_val]
    train_cases = shuffled_cases[n_val:]

    return train_cases, val_cases


def make_kfold_splits(
    cases: List[Dict],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[List[Dict], List[Dict]]]:
    """Genera particiones para validación cruzada en k folds.

    Args:
        cases: Lista de casos a dividir.
        n_splits: Número de folds.
        seed: Semilla usada para barajar los casos.

    Returns:
        Una lista de tuplas ``(train_val_cases, test_cases)`` para cada fold,
        donde train_val_cases es la lista de casos usados para entrenamiento 
        y validación, y test_cases es la lista de casos usados para prueba.

    Raises:
        ValueError: Si ``n_splits`` es menor que 2.
        ValueError: Si no hay suficientes casos para generar los folds.
    """
    if n_splits < 2:
        raise ValueError("n_splits debe ser >= 2.")
    if len(cases) < n_splits:
        raise ValueError("No hay suficientes casos para tantos folds.")

    random_generator = random.Random(seed)
    indices = list(range(len(cases)))
    random_generator.shuffle(indices)

    fold_sizes = [len(cases) // n_splits] * n_splits
    for i in range(len(cases) % n_splits):
        fold_sizes[i] += 1

    folds = []
    current = 0

    for fold_size in fold_sizes:
        test_idx = set(indices[current:current + fold_size])
        current += fold_size

        train_val_cases = [cases[i] for i in range(len(cases)) if i not in test_idx]
        test_cases = [cases[i] for i in range(len(cases)) if i in test_idx]

        folds.append((train_val_cases, test_cases))

    return folds
