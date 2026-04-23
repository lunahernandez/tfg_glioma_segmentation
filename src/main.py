from pathlib import Path
import argparse
import json
import math
import statistics
import gc

import torch
import torch.multiprocessing as mp
from monai.data import PersistentDataset, DataLoader

from src.config import *
from src.utils.seed import set_seed
from src.utils.checkpoints import load_checkpoint
from src.data.dataset import collect_cases_from_dirs
from src.data.splits import save_split, split_train_val, make_kfold_splits
from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
)
from src.models.get_model import get_model
from src.train import train_model
from src.evaluate import evaluate_test


def safe_mean(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if len(values) == 0:
        return None
    return float(sum(values) / len(values))


def safe_std(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    return float(statistics.stdev(values))


def build_cache_dirs():
    base = PERSISTENT_CACHE_DIR
    return {
        "train": base / "train",
        "eval": base / "eval",
    }


def build_train_val_loaders(train_cases, val_cases, cache_dirs):
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


def build_test_loader(test_cases, cache_dirs):
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


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_base_results(
    fold_idx,
    train_cases,
    val_cases,
    test_cases,
    best_val=None,
    best_epoch=None,
    train_time_sec=None,
    train_memory_mb=None,
    history=None,
):
    return {
        "fold": fold_idx,
        "model": MODEL_NAME,
        "best_val_dice": best_val,
        "best_epoch": best_epoch,
        "test_lesionwise_mean_dice": None,
        "test_lesionwise_mean_hd95": None,
        "test_lesionwise_by_label": None,
        "test_cases": None,
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


def train_fold(fold_idx, train_cases, val_cases, test_cases):
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
        num_classes=OUT_CHANNELS,
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


def test_fold(fold_idx, train_cases, val_cases, test_cases):
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
            test_cases=test_cases,
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
            num_classes=OUT_CHANNELS,
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
        existing_results["test_cases"] = test_metrics["per_case"]
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


def run_fold(fold_idx, train_cases, val_cases, test_cases, mode):
    if mode == "train":
        return train_fold(fold_idx, train_cases, val_cases, test_cases)

    if mode == "test":
        return test_fold(fold_idx, train_cases, val_cases, test_cases)

    if mode == "all":
        train_fold(fold_idx, train_cases, val_cases, test_cases)
        cleanup_memory()
        return test_fold(fold_idx, train_cases, val_cases, test_cases)

    raise ValueError(f"Modo no soportado: {mode}")


def aggregate_cv_results(fold_results):
    summary = {
        "model": MODEL_NAME,
        "n_folds": N_FOLDS,
        "folds": fold_results,
        "cv_best_val_dice_mean": safe_mean([f["best_val_dice"] for f in fold_results]),
        "cv_best_val_dice_std": safe_std([f["best_val_dice"] for f in fold_results]),
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "test", "all"],
        default="all",
        help="train: solo entrena, test: solo evalúa, all: entrena y evalúa.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Si se indica, ejecuta solo ese fold.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    experiment_root = EXPERIMENTS_DIR / EXPERIMENT_NAME
    experiment_root.mkdir(parents=True, exist_ok=True)

    all_cases = collect_cases_from_dirs(TRAIN_DIRS, include_label=True)
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
        print(f"Modo: {args.mode}")
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

    cv_results = aggregate_cv_results(fold_results)

    with open(experiment_root / "crossval_results.json", "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2)

    print(json.dumps(cv_results, indent=2))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
