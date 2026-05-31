from pathlib import Path
import argparse
import json
from typing import Any

from src.config import EXPERIMENTS_DIR, EXPERIMENT_NAME, MODEL_NAME
from scripts.utils_results import load_json, safe_mean, safe_std, REGION_NAMES


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Save a dictionary as a JSON file.

    Args:
        path: Path where the JSON file will be saved.
        data: Dictionary containing the data to store.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def get_region_metric(
    fold: dict[str, Any],
    region: str,
    metric: str,
) -> float | None:
    """Get a lesion-wise metric for a specific region from a fold result.

    Args:
        fold: Dictionary containing the results of one fold.
        region: Name of the BraTS region to retrieve.
        metric: Name of the metric to retrieve.

    Returns:
        The requested metric value if it exists, otherwise ``None``.
    """
    test_lesionwise_by_label = fold.get("test_lesionwise_by_label")

    if not isinstance(test_lesionwise_by_label, dict):
        return None

    region_data = test_lesionwise_by_label.get(region)

    if not isinstance(region_data, dict):
        return None

    return region_data.get(metric)


def collect_fold_results(experiment_root: Path) -> list[dict[str, Any]]:
    """Collect the individual results of all folds in an experiment.

    Args:
        experiment_root: Path to the experiment directory containing the fold
            subdirectories.

    Returns:
        A list of dictionaries with the results of each fold, sorted by fold
        index.
    """
    fold_results = []

    for fold_dir in sorted(experiment_root.glob("fold_*")):
        results_path = fold_dir / "results.json"

        if not results_path.exists():
            print(f"Skipping {fold_dir.name}: results.json not found")
            continue

        fold_results.append(load_json(results_path))
        print(f"Loaded {results_path}")

    return sorted(fold_results, key=lambda fold: fold.get("fold", 0))


def aggregate_cv_results(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates the results obtained from cross-validation folds.

    Computes means and standard deviations of the main validation and test
    metrics. It also aggregates the lesion-wise results for each BraTS region.
    This function can be used after the folds have been executed independently,
    without requiring a sequential execution of the complete cross-validation
    workflow.

    Args:
        fold_results: List of dictionaries with the results of each fold.

    Returns:
        A dictionary with the global cross-validation summary.
    """

    summary = {
        "model": fold_results[0].get("model", MODEL_NAME),
        "n_folds": len(fold_results),
        "folds": fold_results,
        "cv_best_val_dice_mean": safe_mean(
            [fold.get("best_val_dice") for fold in fold_results]
        ),
        "cv_best_val_dice_std": safe_std(
            [fold.get("best_val_dice") for fold in fold_results]
        ),
        "cv_test_lesionwise_mean_dice_mean": safe_mean(
            [fold.get("test_lesionwise_mean_dice") for fold in fold_results]
        ),
        "cv_test_lesionwise_mean_dice_std": safe_std(
            [fold.get("test_lesionwise_mean_dice") for fold in fold_results]
        ),
        "cv_test_lesionwise_mean_hd95_mean": safe_mean(
            [fold.get("test_lesionwise_mean_hd95") for fold in fold_results]
        ),
        "cv_test_lesionwise_mean_hd95_std": safe_std(
            [fold.get("test_lesionwise_mean_hd95") for fold in fold_results]
        ),
        "regions": {},
    }

    for region in REGION_NAMES:
        dice_values = [
            get_region_metric(fold, region, "lesionwise_dice")
            for fold in fold_results
        ]
        hd95_values = [
            get_region_metric(fold, region, "lesionwise_hd95")
            for fold in fold_results
        ]

        summary["regions"][region] = {
            "lesionwise_dice_mean": safe_mean(dice_values),
            "lesionwise_dice_std": safe_std(dice_values),
            "lesionwise_hd95_mean": safe_mean(hd95_values),
            "lesionwise_hd95_std": safe_std(hd95_values),
        }

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=EXPERIMENTS_DIR / EXPERIMENT_NAME,
        help="Experiment folder containing fold_1, fold_2, etc.",
    )

    return parser.parse_args()


def main() -> None:
    """Generate the aggregated cross-validation results file.

    The function loads the individual fold results, aggregates the metrics and
    saves the final summary as ``crossval_results.json`` inside the experiment
    directory.
    """
    args = parse_args()

    fold_results = collect_fold_results(args.experiment_root)

    if not fold_results:
        raise FileNotFoundError(
            f"No fold results found in {args.experiment_root}"
        )

    cv_results = aggregate_cv_results(fold_results)
    output_path = args.experiment_root / "crossval_results.json"

    save_json(output_path, cv_results)

    print(f"\nSaved: {output_path}")
    print(json.dumps(cv_results, indent=2))


if __name__ == "__main__":
    main()


