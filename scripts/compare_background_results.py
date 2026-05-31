from pathlib import Path
import argparse
from typing import Any

from scripts.utils_results import (
    load_json,
    safe_mean,
    safe_std,
    write_csv,
)


def get_folds_by_id(data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Return the fold results indexed by fold identifier.

    Args:
        data: Cross-validation results dictionary.

    Returns:
        A dictionary where each key is a fold identifier and each value is the
        corresponding fold result.
    """
    folds = {}

    for fold in data.get("folds", []):
        fold_id = fold.get("fold")

        if isinstance(fold_id, int):
            folds[fold_id] = fold

    return folds


def build_fold_comparison(
    without_bg: dict[str, Any],
    with_bg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare fold-level metrics between experiments without and with background.

    Args:
        without_bg: Cross-validation results from the experiment where the
            background class was excluded from the loss.
        with_bg: Cross-validation results from the experiment where the
            background class was included in the loss.

    Returns:
        A list of rows containing the fold-level metrics and the differences
        between both configurations.
    """
    without_folds = get_folds_by_id(without_bg)
    with_folds = get_folds_by_id(with_bg)

    common_folds = sorted(set(without_folds) & set(with_folds))
    rows = []

    for fold_id in common_folds:
        fold_without = without_folds[fold_id]
        fold_with = with_folds[fold_id]

        dice_without = fold_without.get("test_lesionwise_mean_dice")
        dice_with = fold_with.get("test_lesionwise_mean_dice")

        hd95_without = fold_without.get("test_lesionwise_mean_hd95")
        hd95_with = fold_with.get("test_lesionwise_mean_hd95")

        delta_dice = (
            dice_with - dice_without
            if dice_with is not None and dice_without is not None
            else None
        )

        delta_hd95 = (
            hd95_with - hd95_without
            if hd95_with is not None and hd95_without is not None
            else None
        )

        rows.append(
            {
                "fold": fold_id,
                "dice_without_bg": dice_without,
                "dice_with_bg": dice_with,
                "delta_dice_with_minus_without": delta_dice,
                "hd95_without_bg": hd95_without,
                "hd95_with_bg": hd95_with,
                "delta_hd95_with_minus_without": delta_hd95,
                "dice_better_with_bg": delta_dice is not None and delta_dice > 0,
                "hd95_better_with_bg": delta_hd95 is not None and delta_hd95 < 0,
            }
        )

    return rows


def build_summary_comparison(
    fold_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize the fold-level differences between both configurations.

    Args:
        fold_rows: Fold-level comparison rows generated from the experiments
            without and with background.

    Returns:
        A list with the mean and standard deviation of the Dice and HD95
        differences across folds.
    """
    dice_deltas = [
        row["delta_dice_with_minus_without"]
        for row in fold_rows
    ]

    hd95_deltas = [
        row["delta_hd95_with_minus_without"]
        for row in fold_rows
    ]

    return [
        {
            "metric": "test_lesionwise_mean_dice",
            "delta_mean_with_minus_without": safe_mean(dice_deltas),
            "delta_std_with_minus_without": safe_std(dice_deltas),
            "interpretation": "positive means with background is better",
        },
        {
            "metric": "test_lesionwise_mean_hd95",
            "delta_mean_with_minus_without": safe_mean(hd95_deltas),
            "delta_std_with_minus_without": safe_std(hd95_deltas),
            "interpretation": "negative means with background is better",
        },
    ]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--without-bg",
        type=Path,
        required=True,
        help="Path to crossval_results.json from the experiment without background.",
    )

    parser.add_argument(
        "--with-bg",
        type=Path,
        required=True,
        help="Path to crossval_results.json from the experiment with background.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/background_comparison"),
        help="Directory where comparison CSV files will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    """Compare experiments with and without background in the loss function.

    The script compares the same model trained under two configurations:
    excluding and including the background class in the loss. It saves a
    fold-level comparison and a global summary of the metric differences.
    """
    args = parse_args()

    without_bg = load_json(args.without_bg)
    with_bg = load_json(args.with_bg)

    fold_rows = build_fold_comparison(
        without_bg=without_bg,
        with_bg=with_bg,
    )

    summary_rows = build_summary_comparison(fold_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(args.output_dir / "fold_comparison.csv", fold_rows)
    write_csv(args.output_dir / "summary_comparison.csv", summary_rows)

    print("Summary comparison:")
    for row in summary_rows:
        print(row)

    print(f"\nSaved comparison files in: {args.output_dir}")


if __name__ == "__main__":
    main()


