from pathlib import Path
import argparse
import gc
import json
import math
import statistics
from typing import Any, Iterable

import torch
import torch.multiprocessing as mp
from monai.data import DataLoader, PersistentDataset

from src.config import *
from src.data.splits import make_kfold_splits, save_split, split_train_val
from src.data.transforms import (
    get_test_transforms,
    get_train_transforms,
    get_val_transforms,
)
from src.evaluate import evaluate_test
from src.models.get_model import get_model
from src.train import train_model
from src.utils.checkpoints import load_checkpoint
from src.utils.seed import set_seed
from src.data.dataset import get_cases_from_dirs


def safe_mean(values: Iterable[int | float | None]) -> float | None:
    """Calculates the mean while ignoring invalid values.

    Args:
        values: Sequence of numeric values that may contain None or NaN values.

    Returns:
        Mean of the valid values as a float. Returns None if there are no valid
        values.
    """
    valid_values = [v for v in values if v is not None and not math.isnan(v)]
    if len(valid_values) == 0:
        return None
    return float(sum(valid_values) / len(valid_values))


def safe_std(values: Iterable[int | float | None]) -> float | None:
    """Calculates the standard deviation while ignoring invalid values.

    Args:
        values: Sequence of numeric values that may contain None or NaN values.

    Returns:
        Standard deviation of the valid values. Returns 0.0 if there is only
        one valid value and None if there are no valid values.
    """
    valid_values = [v for v in values if v is not None and not math.isnan(v)]
    if len(valid_values) < 2:
        return 0.0 if len(valid_values) == 1 else None
    return float(statistics.stdev(valid_values))


def build_cache_dirs() -> dict[str, Path]:
    """Builds the persistent cache directories.

    Returns:
        A dictionary with the cache directories used for training and evaluation.
    """
    base = PERSISTENT_CACHE_DIR
    return {
        "train": base / "train",
        "eval": base / "eval",
    }


def build_train_val_loaders(
    train_cases: list[dict[str, Any]],
    val_cases: list[dict[str, Any]],
    cache_dirs: dict[str, Path],
) -> tuple[DataLoader, DataLoader]:
    """Builds the training and validation data loaders.

    Creates persistent datasets by applying the corresponding training and
    validation transforms. Then, it builds the DataLoader objects used during
    model training.

    Args:
        train_cases: List of cases used for training.
        val_cases: List of cases used for internal validation.
        cache_dirs: Dictionary with the persistent cache directories.

    Returns:
        A tuple (train_loader, val_loader), where train_loader is the DataLoader
        used for training and val_loader is the DataLoader used for internal
        validation.
    """
    train_ds = PersistentDataset(
        data=train_cases,
        transform=get_train_transforms(roi_size=ROI_SIZE, spacing=SPACING),
        cache_dir=cache_dirs["train"],
    )

    val_ds = PersistentDataset(
        data=val_cases,
        transform=get_val_transforms(roi_size=ROI_SIZE, spacing=SPACING),
        cache_dir=cache_dirs["eval"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        multiprocessing_context="spawn" if NUM_WORKERS > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        multiprocessing_context="spawn" if NUM_WORKERS > 0 else None,
    )

    return train_loader, val_loader


def build_test_loader(
    test_cases: list[dict[str, Any]],
    cache_dirs: dict[str, Path],
) -> DataLoader:
    """Builds the test data loader.

    Creates a persistent dataset by applying the test transforms and builds the
    corresponding DataLoader.

    Args:
        test_cases: List of cases used for testing.
        cache_dirs: Dictionary with the persistent cache directories.

    Returns:
        A DataLoader for the test set.
    """
    test_ds = PersistentDataset(
        data=test_cases,
        transform=get_test_transforms(roi_size=ROI_SIZE, spacing=SPACING),
        cache_dir=cache_dirs["eval"],
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        multiprocessing_context="spawn" if NUM_WORKERS > 0 else None,
    )

    return test_loader


def cleanup_memory() -> None:
    """Releases unused memory after a training or testing phase.

    Runs Python garbage collection and, if a GPU is available, synchronizes the
    device and clears the CUDA cache.
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Saves a dictionary to a JSON file.

    Args:
        path: Path where the JSON file will be saved.
        data: Dictionary containing the information to store.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict[str, Any] | None:
    """Loads a JSON file if it exists.

    Args:
        path: Path of the JSON file to load.

    Returns:
        A dictionary with the file contents. Returns None if the file does not
        exist.
    """
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_base_results(
    fold_idx: int,
    train_cases: list[dict[str, Any]],
    val_cases: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
    best_val: float | None = None,
    best_epoch: int | None = None,
    train_time_sec: float | None = None,
    train_memory_mb: float | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Builds the base result structure for a fold.

    This structure contains information about the fold, the model, the
    configuration used, the subset sizes, validation metrics, fields reserved
    for test metrics, and the training history.

    Args:
        fold_idx: Fold index.
        train_cases: Cases used for training.
        val_cases: Cases used for internal validation.
        test_cases: Cases used for testing.
        best_val: Best Dice value obtained during validation.
        best_epoch: Epoch where the best validation result was obtained.
        train_time_sec: Total training time in seconds.
        train_memory_mb: Maximum memory used during training, in megabytes.
        history: Model training history.

    Returns:
        A dictionary with the base result structure for the fold.
    """
    return {
        "fold": fold_idx,
        "model": MODEL_NAME,
        "best_val_dice": best_val,
        "best_epoch": best_epoch,
        "test_lesionwise_mean_dice": None,
        "test_lesionwise_mean_hd95": None,
        "test_lesionwise_by_label": None,
        "train_time_sec": train_time_sec,
        "inference_time_per_case_sec": None,
        "train_memory_mb": train_memory_mb,
        "test_memory_mb": None,
        "test_error": None,
        "config": {
            "roi_size": ROI_SIZE,
            "spacing": SPACING,
            "batch_size": BATCH_SIZE,
            "val_batch_size": VAL_BATCH_SIZE,
            "sw_batch_size": SW_BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "clip_grad": CLIP_GRAD,
            "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
            "use_checkpoint": USE_CHECKPOINT,
            "num_workers": NUM_WORKERS,
            "persistent_workers": False,
            "seed": SEED + fold_idx,
            "n_folds": N_FOLDS,
            "inner_val_ratio": INNER_VAL_RATIO,
        },
        "history": history,
        "sizes": {
            "train": len(train_cases),
            "val": len(val_cases),
            "test": len(test_cases),
        },
    }


def train_fold(
    fold_idx: int,
    train_cases: list[dict[str, Any]],
    val_cases: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Trains a model on a specific fold.

    Sets the fold seed, saves the data split, builds the training and validation
    loaders, creates the model and optimizer, runs the training process, and
    saves the results associated with the best checkpoint. It also records the
    maximum memory used during training.

    Args:
        fold_idx: Index of the fold to train.
        train_cases: Cases used for training.
        val_cases: Cases used for internal validation.
        test_cases: Cases reserved for testing.

    Returns:
        A dictionary with the training results for the fold.
    """
    set_seed(SEED + fold_idx)

    fold_dir = EXPERIMENTS_DIR / EXPERIMENT_NAME / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    save_split(
        train_cases=train_cases,
        val_cases=val_cases,
        test_cases=test_cases,
        save_path=fold_dir / "split.json",
        fold=fold_idx,
    )

    cache_dirs = build_cache_dirs()
    cache_dirs["train"].mkdir(parents=True, exist_ok=True)
    cache_dirs["eval"].mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_train_val_loaders(
        train_cases=train_cases,
        val_cases=val_cases,
        cache_dirs=cache_dirs,
    )

    model = get_model(
        model_name=MODEL_NAME,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        use_checkpoint=USE_CHECKPOINT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_info = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        optimizer=optimizer,
        max_epochs=MAX_EPOCHS,
        val_every=VAL_EVERY,
        experiment_dir=fold_dir,
        roi_size=ROI_SIZE,
        sw_batch_size=SW_BATCH_SIZE,
        clip_grad=CLIP_GRAD,
        grad_clip_max_norm=GRAD_CLIP_MAX_NORM or 1.0,
    )

    train_mem = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available()
        else None
    )

    model, _, best_epoch, best_val = load_checkpoint(
        model=model,
        optimizer=None,
        checkpoint_path=fold_dir / "best_model.pt",
        device=DEVICE,
    )

    fold_results = build_base_results(
        fold_idx=fold_idx,
        train_cases=train_cases,
        val_cases=val_cases,
        test_cases=test_cases,
        best_val=best_val,
        best_epoch=best_epoch,
        train_time_sec=train_info["total_train_time_sec"],
        train_memory_mb=train_mem,
        history=train_info["history"],
    )

    save_json(fold_dir / "results.json", fold_results)

    del model
    del optimizer
    del train_loader
    del val_loader
    cleanup_memory()

    return fold_results


def test_fold(
    fold_idx: int,
    train_cases: list[dict[str, Any]],
    val_cases: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluates the best model of a fold on the test set.

    Loads the checkpoint with the best validation performance, builds the test
    loader, and computes lesion-wise metrics on the test set. It also records
    the average inference time and the maximum memory used during evaluation.

    Args:
        fold_idx: Index of the fold to evaluate.
        train_cases: Cases used for training.
        val_cases: Cases used for internal validation.
        test_cases: Cases used for testing.

    Returns:
        A dictionary with the updated results for the fold.

    Raises:
        Exception: Re-raises any error produced during test evaluation after
        storing it in the results file.
    """
    set_seed(SEED + fold_idx)

    fold_dir = EXPERIMENTS_DIR / EXPERIMENT_NAME / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    cache_dirs = build_cache_dirs()
    cache_dirs["train"].mkdir(parents=True, exist_ok=True)
    cache_dirs["eval"].mkdir(parents=True, exist_ok=True)

    test_loader = build_test_loader(
        test_cases=test_cases,
        cache_dirs=cache_dirs,
    )

    model = get_model(
        model_name=MODEL_NAME,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        use_checkpoint=USE_CHECKPOINT,
    ).to(DEVICE)

    model, _, best_epoch, best_val = load_checkpoint(
        model=model,
        optimizer=None,
        checkpoint_path=fold_dir / "best_model.pt",
        device=DEVICE,
    )

    existing_results = load_json(fold_dir / "results.json")

    if existing_results is None:
        existing_results = build_base_results(
            fold_idx=fold_idx,
            train_cases=train_cases,
            val_cases=val_cases,
            best_val=best_val,
            best_epoch=best_epoch,
        )
        save_json(fold_dir / "results.json", existing_results)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        test_metrics = evaluate_test(
            model=model,
            test_loader=test_loader,
            device=DEVICE,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            output_dir=fold_dir / "test_eval",
        )

        test_mem = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if torch.cuda.is_available()
            else None
        )

        existing_results["best_val_dice"] = best_val
        existing_results["best_epoch"] = best_epoch
        existing_results["test_lesionwise_mean_dice"] = test_metrics["mean_lesionwise_dice"]
        existing_results["test_lesionwise_mean_hd95"] = test_metrics["mean_lesionwise_hd95"]
        existing_results["test_lesionwise_by_label"] = test_metrics["by_label"]
        existing_results["inference_time_per_case_sec"] = test_metrics["avg_inference_time_sec"]
        existing_results["test_memory_mb"] = test_mem
        existing_results["test_error"] = None

    except Exception as e:
        existing_results["test_error"] = repr(e)
        save_json(fold_dir / "results.json", existing_results)
        raise

    save_json(fold_dir / "results.json", existing_results)

    del model
    del test_loader
    cleanup_memory()

    return existing_results


def run_fold(
    fold_idx: int,
    train_cases: list[dict[str, Any]],
    val_cases: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    """Runs a fold according to the selected mode.

    Allows running only training, only testing, or both phases sequentially.

    Args:
        fold_idx: Fold index.
        train_cases: Cases used for training.
        val_cases: Cases used for internal validation.
        test_cases: Cases used for testing.
        mode: Execution mode.

    Returns:
        A dictionary with the results generated by the executed phase.

    Raises:
        ValueError: If the selected mode is not supported.
    """
    if mode == "train":
        return train_fold(fold_idx, train_cases, val_cases, test_cases)

    if mode == "test":
        return test_fold(fold_idx, train_cases, val_cases, test_cases)

    if mode == "all":
        train_fold(fold_idx, train_cases, val_cases, test_cases)
        cleanup_memory()
        return test_fold(fold_idx, train_cases, val_cases, test_cases)

    raise ValueError(f"Unsupported mode: {mode}")


def aggregate_cv_results(
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregates the results obtained from cross-validation folds.

    Computes means and standard deviations of the main validation and test
    metrics. It also aggregates the lesion-wise results for each BraTS region.

    Args:
        fold_results: List of dictionaries with the results of each fold.

    Returns:
        A dictionary with the global cross-validation summary.
    """
    summary = {
        "model": MODEL_NAME,
        "n_folds": N_FOLDS,
        "folds": fold_results,
        "cv_best_val_dice_mean": safe_mean(
            [f["best_val_dice"] for f in fold_results]
        ),
        "cv_best_val_dice_std": safe_std(
            [f["best_val_dice"] for f in fold_results]
        ),
        "cv_test_lesionwise_mean_dice_mean": safe_mean(
            [f["test_lesionwise_mean_dice"] for f in fold_results]
        ),
        "cv_test_lesionwise_mean_dice_std": safe_std(
            [f["test_lesionwise_mean_dice"] for f in fold_results]
        ),
        "cv_test_lesionwise_mean_hd95_mean": safe_mean(
            [f["test_lesionwise_mean_hd95"] for f in fold_results]
        ),
        "cv_test_lesionwise_mean_hd95_std": safe_std(
            [f["test_lesionwise_mean_hd95"] for f in fold_results]
        ),
        "regions": {},
    }

    region_names = ["ET", "NETC", "SNFH", "RC", "TC", "WT"]

    for region in region_names:
        dice_vals = [
            f["test_lesionwise_by_label"][region]["lesionwise_dice"]
            for f in fold_results
            if f["test_lesionwise_by_label"] is not None
        ]
        hd95_vals = [
            f["test_lesionwise_by_label"][region]["lesionwise_hd95"]
            for f in fold_results
            if f["test_lesionwise_by_label"] is not None
        ]

        summary["regions"][region] = {
            "lesionwise_dice_mean": safe_mean(dice_vals),
            "lesionwise_dice_std": safe_std(dice_vals),
            "lesionwise_hd95_mean": safe_mean(hd95_vals),
            "lesionwise_hd95_std": safe_std(hd95_vals),
        }

    return summary


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Allows selecting the execution mode and, optionally, a specific fold.

    Returns:
        An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "test", "all"],
        default="all",
        help="train: only trains, test: only evaluates, all: trains and evaluates.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="If specified, executes only that fold.",
    )

    return parser.parse_args()


def main() -> None:
    """Runs the main experimental workflow.

    Loads the available cases, creates the cross-validation folds, splits each
    fold into training, validation, and test subsets, runs the selected mode,
    and saves the results. If all folds are executed, it aggregates the final
    results into a cross-validation summary file.
    """
    args = parse_args()
    set_seed(SEED)

    experiment_root = EXPERIMENTS_DIR / EXPERIMENT_NAME
    experiment_root.mkdir(parents=True, exist_ok=True)

    all_cases = get_cases_from_dirs(TRAIN_DIRS, include_label=True)
    folds = make_kfold_splits(all_cases, n_splits=N_FOLDS, seed=SEED)

    fold_results = []

    for fold_idx, (train_val_cases, test_cases) in enumerate(folds, start=1):
        if args.fold is not None and fold_idx != args.fold:
            continue

        cleanup_memory()

        train_cases, val_cases = split_train_val(
            train_val_cases,
            val_ratio=INNER_VAL_RATIO,
            seed=SEED + fold_idx,
        )

        print("=" * 70)
        print(f"Fold {fold_idx}/{N_FOLDS}")
        print(f"Mode: {args.mode}")
        print(f"Train: {len(train_cases)} | Val: {len(val_cases)} | Test: {len(test_cases)}")

        fold_result = run_fold(
            fold_idx=fold_idx,
            train_cases=train_cases,
            val_cases=val_cases,
            test_cases=test_cases,
            mode=args.mode,
        )
        fold_results.append(fold_result)

        cleanup_memory()


    if args.fold is None:
        cv_results = aggregate_cv_results(fold_results)

        with open(experiment_root / "crossval_results.json", "w", encoding="utf-8") as f:
            json.dump(cv_results, f, indent=2)

        print(json.dumps(cv_results, indent=2))
    else:
        print(f"\nFold {args.fold} completed. Individual results are saved in its folder.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
