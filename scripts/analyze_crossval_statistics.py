from pathlib import Path
import argparse
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

from utils_results import (
    REGION_NAMES,
    REGION_TEST_NAMES,
    collect_results,
    collect_fold_level_points,
    calculate_pearson_correlation,
    get_model_color,
    get_significance_stars,
    get_valid_pairs,
    is_valid_number,
    safe_mean,
    write_csv,
)

from plot_crossval_results import (
    set_plot_style,
    format_axes,
    save_current_figure,
)


def save_correlation_table(
    output_path: Path,
    fold_rows: list[dict[str, Any]],
) -> None:
    """Save a CSV table with performance-cost correlations.

    Args:
        output_path: The path where the CSV file is saved.
        fold_rows: The fold-level rows containing performance and cost metrics.
    """
    performance_metrics = [
        ("test_dice", "Lesion-wise Dice"),
        ("test_hd95", "Lesion-wise HD95"),
    ]

    cost_metrics = [
        ("train_time_h", "Training time (h)"),
        ("inference_time_s", "Inference time per case (s)"),
        ("train_memory_gb", "Training memory (GB)"),
        ("test_memory_gb", "Test memory (GB)"),
    ]

    rows: list[dict[str, Any]] = []

    for y_key, y_name in performance_metrics:
        for x_key, x_name in cost_metrics:
            x_values, y_values = get_valid_pairs(fold_rows, x_key, y_key)
            r_value, p_value = calculate_pearson_correlation(x_values, y_values)

            rows.append(
                {
                    "performance_metric": y_name,
                    "cost_metric": x_name,
                    "n_points": len(x_values),
                    "pearson_r": round(r_value, 4) if r_value is not None else None,
                    "p_value": f"{p_value:.2e}" if p_value is not None else None,
                    "significance": get_significance_stars(p_value),
                }
            )

    write_csv(output_path, rows)


def apply_holm_correction(p_values: list[float]) -> list[float]:
    """Apply the Holm correction to a list of p-values.

    Args:
        p_values: The raw p-values to correct.

    Returns:
        A list with the corrected p-values in the original order.
    """
    n_values = len(p_values)
    indexed_p_values = sorted(enumerate(p_values), key=lambda item: item[1])

    corrected_p_values = [1.0] * n_values
    previous_corrected = 0.0

    for rank, (original_index, p_value) in enumerate(indexed_p_values):
        corrected_p = min((n_values - rank) * p_value, 1.0)
        corrected_p = max(corrected_p, previous_corrected)
        corrected_p_values[original_index] = corrected_p
        previous_corrected = corrected_p

    return corrected_p_values


def collect_region_metric_matrix(
    results: list[dict[str, Any]],
    metric_key: str,
) -> dict[str, list[float]]:
    """Collect regional metric values aligned by model and fold.

    Args:
        results: Cross-validation results loaded from JSON files.
        metric_key: Metric to collect, such as ``lesionwise_dice`` or
            ``lesionwise_hd95``.

    Returns:
        A dictionary with one list of values per original BraTS region. Values
        at the same position correspond to the same model and fold.
    """
    region_values: dict[str, list[float]] = {
        region: []
        for region in REGION_TEST_NAMES
    }

    for result in results:
        for fold in result.get("folds", []):
            labels_data = fold.get("test_lesionwise_by_label", {})
            row_values: dict[str, float] = {}

            for region in REGION_TEST_NAMES:
                value = labels_data.get(region, {}).get(metric_key)

                if not is_valid_number(value):
                    row_values = {}
                    break

                row_values[region] = float(value)

            if not row_values:
                continue

            for region in REGION_TEST_NAMES:
                region_values[region].append(row_values[region])

    return region_values


def save_region_significance_tests(
    output_path: Path,
    results: list[dict[str, Any]],
) -> None:
    """Run Friedman and pairwise Wilcoxon tests between BraTS regions.

    Args:
        output_path: The path where the CSV file is saved.
        results: The cross-validation results loaded from JSON files.
    """
    metric_configs = [
        ("lesionwise_dice", "Dice lesion-wise"),
        ("lesionwise_hd95", "HD95 lesion-wise"),
    ]

    rows: list[dict[str, Any]] = []

    for metric_key, metric_name in metric_configs:
        region_values = collect_region_metric_matrix(
            results=results,
            metric_key=metric_key,
        )

        valid_regions = [
            region
            for region in REGION_TEST_NAMES
            if len(region_values[region]) > 0
        ]

        if len(valid_regions) < 2:
            continue

        n_points = min(len(region_values[region]) for region in valid_regions)

        if n_points < 3:
            continue

        aligned_values = [
            region_values[region][:n_points]
            for region in valid_regions
        ]

        statistic, p_value = friedmanchisquare(*aligned_values)

        rows.append(
            {
                "metric": metric_name,
                "test": "Global Friedman",
                "comparison": "All original regions",
                "n_points": n_points,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "p_value_corrected": None,
                "significance": get_significance_stars(float(p_value)),
            }
        )

        pairwise_rows: list[dict[str, Any]] = []
        raw_p_values: list[float] = []

        for region_a, region_b in combinations(valid_regions, 2):
            values_a = region_values[region_a][:n_points]
            values_b = region_values[region_b][:n_points]

            try:
                pair_statistic, pair_p_value = wilcoxon(values_a, values_b)
                pair_statistic = float(pair_statistic)
                pair_p_value = float(pair_p_value)
            except ValueError:
                pair_statistic = None
                pair_p_value = None

            pairwise_rows.append(
                {
                    "metric": metric_name,
                    "test": "Paired Wilcoxon",
                    "comparison": f"{region_a} vs {region_b}",
                    "n_points": n_points,
                    "statistic": pair_statistic,
                    "p_value": pair_p_value,
                    "p_value_corrected": None,
                    "significance": None,
                }
            )

            raw_p_values.append(pair_p_value if pair_p_value is not None else 1.0)

        corrected_p_values = apply_holm_correction(raw_p_values)

        for row, corrected_p in zip(pairwise_rows, corrected_p_values):
            row["p_value_corrected"] = corrected_p
            row["significance"] = get_significance_stars(corrected_p)
            rows.append(row)

    write_csv(output_path, rows)


def save_fold_scatter_plot(
    output_path: Path,
    fold_rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """Save a fold-level scatter plot.

    Args:
        output_path: The path where the PDF figure is saved.
        fold_rows: The fold-level rows containing performance and cost metrics.
        x_key: The key used for the x-axis variable.
        y_key: The key used for the y-axis variable.
        xlabel: The x-axis label.
        ylabel: The y-axis label.
        title: The plot title.
    """
    plt.figure(figsize=(8, 5))

    models = sorted(set(row["model"] for row in fold_rows))

    for model in models:
        model_rows = [row for row in fold_rows if row["model"] == model]
        x_values, y_values = get_valid_pairs(model_rows, x_key, y_key)

        if not x_values:
            continue

        plt.scatter(
            x_values,
            y_values,
            label=model,
            color=get_model_color(model),
            edgecolor="black",
            linewidth=0.8,
            s=70,
            alpha=0.9,
        )

    all_x, all_y = get_valid_pairs(fold_rows, x_key, y_key)
    r_value, p_value = calculate_pearson_correlation(all_x, all_y)

    if r_value is not None:
        stars = get_significance_stars(p_value)
        r_text = f"{r_value:.3f}".replace(".", ",")
        title = f"{title} (r = {r_text}{stars})"

    format_axes(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        grid_axis="both",
    )

    plt.legend(frameon=True)
    plt.tight_layout()
    save_current_figure(output_path)
    plt.close()


def collect_et_netc_points(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect ET and NETC Dice values by model and fold.

    Args:
        results: The cross-validation results loaded from JSON files.

    Returns:
        A list of rows with the model, fold, ET Dice and NETC Dice values.
    """
    rows: list[dict[str, Any]] = []

    for result in results:
        model_name = result["model"]

        for fold in result.get("folds", []):
            labels_data = fold.get("test_lesionwise_by_label", {})
            et_dice = labels_data.get("ET", {}).get("lesionwise_dice")
            netc_dice = labels_data.get("NETC", {}).get("lesionwise_dice")

            if is_valid_number(et_dice) and is_valid_number(netc_dice):
                rows.append(
                    {
                        "model": model_name,
                        "fold": fold.get("fold"),
                        "et_dice": float(et_dice),
                        "netc_dice": float(netc_dice),
                    }
                )

    return rows


def compute_relation_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute slope, Pearson correlation and p-value for ET-NETC points.

    Args:
        rows: The rows containing ET and NETC Dice values.

    Returns:
        A dictionary with the number of points, slope, intercept, Pearson
        correlation, p-value and significance notation.
    """
    x_values = [row["et_dice"] for row in rows]
    y_values = [row["netc_dice"] for row in rows]

    stats: dict[str, Any] = {
        "n_points": len(rows),
        "slope": None,
        "intercept": None,
        "pearson_r": None,
        "p_value": None,
        "significance": "",
    }

    if len(x_values) >= 2 and np.std(x_values) > 0:
        slope, intercept = np.polyfit(x_values, y_values, 1)
        stats["slope"] = float(slope)
        stats["intercept"] = float(intercept)

    r_value, p_value = calculate_pearson_correlation(x_values, y_values)
    stats["pearson_r"] = r_value
    stats["p_value"] = p_value
    stats["significance"] = get_significance_stars(p_value)

    return stats


def check_et_netc_simpson_paradox(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Check whether the ET-NETC relation may show Simpson's paradox.

    Args:
        output_dir: The directory where the CSV report is saved.
        rows: The rows containing ET and NETC Dice values.
    """
    if len(rows) < 3:
        print("There are not enough points to prove Simpson's paradox.")
        return

    global_stats = compute_relation_stats(rows)

    report_rows: list[dict[str, Any]] = [
        {
            "group": "GLOBAL",
            "n_points": global_stats["n_points"],
            "slope": global_stats["slope"],
            "pearson_r": global_stats["pearson_r"],
            "p_value": global_stats["p_value"],
            "significance": global_stats["significance"],
        }
    ]

    model_stats: list[tuple[str, dict[str, Any]]] = []

    for model_name in sorted(set(row["model"] for row in rows)):
        subset = [row for row in rows if row["model"] == model_name]
        stats = compute_relation_stats(subset)
        model_stats.append((model_name, stats))

        report_rows.append(
            {
                "group": model_name,
                "n_points": stats["n_points"],
                "slope": stats["slope"],
                "pearson_r": stats["pearson_r"],
                "p_value": stats["p_value"],
                "significance": stats["significance"],
            }
        )

    csv_path = output_dir / "simpson_check_et_vs_netc.csv"
    write_csv(csv_path, report_rows)

    global_slope = global_stats["slope"]
    valid_model_slopes = [
        stats["slope"]
        for _, stats in model_stats
        if is_valid_number(stats["slope"]) and stats["slope"] != 0
    ]

    simpson_detected = False

    if is_valid_number(global_slope) and global_slope != 0 and valid_model_slopes:
        global_sign = np.sign(global_slope)
        model_signs = [np.sign(slope) for slope in valid_model_slopes]
        simpson_detected = all(sign != global_sign for sign in model_signs)

    print("\n" + "=" * 75)
    print("CHECKING FOR SIMPSON'S PARADOX: ET vs NETC")
    print("=" * 75)
    print(f"Global points: {len(rows)}")
    print(f"Global slope: {global_stats['slope']}")
    print(f"Global Pearson: {global_stats['pearson_r']}")
    print(f"Global p-value: {global_stats['p_value']}")
    print("-" * 75)

    for model_name, stats in model_stats:
        print(
            f"{model_name}: "
            f"n={stats['n_points']}, "
            f"slope={stats['slope']}, "
            f"r={stats['pearson_r']}, "
            f"p={stats['p_value']}"
        )

    if simpson_detected:
        print("Result: possible Simpson's paradox detected.")
    else:
        print("Result: no strict Simpson's paradox observed.")

    print(f"Report saved to: {csv_path}")
    print("=" * 75 + "\n")


def save_et_vs_netc_plot(
    output_path: Path,
    rows: list[dict[str, Any]],
    title: str,
    with_local_lines: bool = False,
) -> None:
    """Save an ET vs NETC scatter plot.

    Args:
        output_path: The path where the PDF figure is saved.
        rows: The rows containing ET and NETC Dice values.
        title: The plot title.
        with_local_lines: Whether to include one trend line per model.
    """
    if len(rows) < 3:
        return

    et_scores = [row["et_dice"] for row in rows]
    netc_scores = [row["netc_dice"] for row in rows]

    r_value, p_value = calculate_pearson_correlation(et_scores, netc_scores)

    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    unique_models = sorted(set(row["model"] for row in rows))

    for model in unique_models:
        subset = [row for row in rows if row["model"] == model]
        x_values = [row["et_dice"] for row in subset]
        y_values = [row["netc_dice"] for row in subset]

        color = get_model_color(model)

        plt.scatter(
            x_values,
            y_values,
            label=model,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            s=70,
            alpha=0.9,
            zorder=4,
        )

        if with_local_lines and len(x_values) >= 2 and np.std(x_values) > 0:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            x_line = np.linspace(min(x_values), max(x_values), 100)
            y_line = slope * x_line + intercept

            plt.plot(
                x_line,
                y_line,
                color=color,
                linestyle="-",
                linewidth=1.5,
                alpha=0.85,
                zorder=3,
            )

    if len(et_scores) >= 2 and np.std(et_scores) > 0:
        slope, intercept = np.polyfit(et_scores, netc_scores, 1)
        x_line = np.linspace(min(et_scores), max(et_scores), 100)
        y_line = slope * x_line + intercept

        plt.plot(
            x_line,
            y_line,
            color="black",
            linestyle="--",
            linewidth=2.0,
            alpha=0.8,
            label="Tendencia global",
            zorder=2,
        )

    if r_value is not None and p_value is not None:
        stars = get_significance_stars(p_value)
        r_text = f"{r_value:.3f}".replace(".", ",")
        title = f"{title} (r = {r_text}{stars})"

    format_axes(
        title=title,
        xlabel="Coeficiente Dice lesion-wise (ET)",
        ylabel="Coeficiente Dice lesion-wise (NETC)",
        grid_axis="both",
    )

    x_range = max(et_scores) - min(et_scores)
    y_range = max(netc_scores) - min(netc_scores)

    if x_range > 0:
        ax.set_xlim(
            left=min(et_scores) - x_range * 0.05,
            right=max(et_scores) + x_range * 0.05,
        )

    if y_range > 0:
        ax.set_ylim(
            bottom=min(netc_scores) - y_range * 0.05,
            top=max(netc_scores) + y_range * 0.05,
        )

    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    save_current_figure(output_path)
    plt.close()


def save_model_significance_tests(
    output_path: Path,
    results: list[dict[str, Any]],
) -> None:
    """Run Friedman and pairwise Wilcoxon tests between models.

    Args:
        output_path: The path where the CSV file is saved.
        results: The cross-validation results loaded from JSON files.
    """
    metric_keys = [
        ("test_lesionwise_mean_dice", "Dice lesion-wise"),
        ("test_lesionwise_mean_hd95", "HD95 lesion-wise"),
    ]

    rows: list[dict[str, Any]] = []

    for metric_key, metric_name in metric_keys:
        model_values: dict[str, list[float]] = {}

        for result in results:
            model = result["model"]
            values: list[float] = []

            for fold in result.get("folds", []):
                value = fold.get(metric_key)

                if is_valid_number(value):
                    values.append(float(value))

            model_values[model] = values

        valid_models = [
            model
            for model, values in model_values.items()
            if len(values) > 0
        ]

        if len(valid_models) < 2:
            continue

        min_len = min(len(model_values[model]) for model in valid_models)

        if min_len < 2:
            continue

        aligned_values = [
            model_values[model][:min_len]
            for model in valid_models
        ]

        if len(valid_models) > 2:
            statistic, p_value = friedmanchisquare(*aligned_values)

            rows.append(
                {
                    "metric": metric_name,
                    "comparison": "Friedman global",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "p_value_corrected": None,
                }
            )

        pairwise_rows: list[dict[str, Any]] = []
        raw_p_values: list[float] = []

        for model_a, model_b in combinations(valid_models, 2):
            values_a = model_values[model_a][:min_len]
            values_b = model_values[model_b][:min_len]

            try:
                statistic, p_value = wilcoxon(values_a, values_b)
                statistic = float(statistic)
                p_value = float(p_value)
            except ValueError:
                statistic = None
                p_value = None

            pairwise_rows.append(
                {
                    "metric": metric_name,
                    "comparison": f"{model_a} vs {model_b}",
                    "statistic": statistic,
                    "p_value": p_value,
                    "p_value_corrected": None,
                }
            )

            raw_p_values.append(p_value if p_value is not None else 1.0)

        corrected_p_values = apply_holm_correction(raw_p_values)

        for row, corrected_p in zip(pairwise_rows, corrected_p_values):
            row["p_value_corrected"] = corrected_p
            rows.append(row)

    write_csv(output_path, rows)


def get_region_fold_metric(
    fold: dict[str, Any],
    region_name: str,
    metric_key: str,
) -> float | None:
    """Get a regional metric from a fold.


    Args:
        fold: The fold-level result dictionary.
        region_name: The BraTS region name.
        metric_key: The regional metric to extract.


    Returns:
        The metric value as a float, or ``None`` if the metric is not available.
    """
    region_metrics = fold.get("test_lesionwise_by_label", {}).get(region_name, {})
    value = region_metrics.get(metric_key)


    return float(value) if is_valid_number(value) else None




def collect_region_metric_by_model(
    results: list[dict[str, Any]],
    region_name: str,
    metric_key: str,
) -> dict[str, dict[int, float]]:
    """Collect a regional metric by model and fold.


    Args:
        results: The cross-validation results loaded from JSON files.
        region_name: The BraTS region name.
        metric_key: The regional metric to collect.


    Returns:
        A dictionary mapping each model to its fold-level metric values.
    """
    values_by_model: dict[str, dict[int, float]] = {}


    for result in results:
        model_name = result["model"]
        values_by_fold: dict[int, float] = {}


        for fold in result.get("folds", []):
            fold_id = fold.get("fold")
            value = get_region_fold_metric(
                fold=fold,
                region_name=region_name,
                metric_key=metric_key,
            )


            if fold_id is not None and is_valid_number(value):
                values_by_fold[int(fold_id)] = float(value)


        values_by_model[model_name] = values_by_fold


    return values_by_model




def get_friedman_critical_values(
    n_blocks: int,
    n_models: int,
) -> tuple[float | None, float | None]:
    """Get Friedman critical values for the current experimental setting.


    Args:
        n_blocks: The number of paired blocks.
        n_models: The number of compared models.


    Returns:
        The critical values for alpha 0.05 and alpha 0.01. Returns ``None``
        values if the current setting is not included.
    """
    if n_blocks == 5 and n_models == 4:
        return 7.800, 9.960


    return None, None


def collect_model_values_by_region(
    results: list[dict[str, Any]],
    region_name: str,
    metric_key: str,
) -> dict[str, dict[int, float]]:
    """Collect regional metric values by model and fold.

    Args:
        results: The cross-validation results loaded from JSON files.
        region_name: The BraTS region to analyze.
        metric_key: The metric to collect.

    Returns:
        A dictionary mapping each model to its fold-level values.
    """
    values_by_model: dict[str, dict[int, float]] = {}

    for result in results:
        model_name = result["model"]
        values_by_fold: dict[int, float] = {}

        for fold in result.get("folds", []):
            fold_id = fold.get("fold")
            region_metrics = fold.get("test_lesionwise_by_label", {}).get(
                region_name,
                {},
            )
            value = region_metrics.get(metric_key)

            if fold_id is not None and is_valid_number(value):
                values_by_fold[int(fold_id)] = float(value)

        values_by_model[model_name] = values_by_fold

    return values_by_model


def save_model_significance_tests_by_region(
    output_path: Path,
    results: list[dict[str, Any]],
) -> None:
    """Run Friedman and pairwise Wilcoxon tests between models by region.

    Args:
        output_path: The path where the CSV file is saved.
        results: The cross-validation results loaded from JSON files.
    """
    metric_configs = [
        ("lesionwise_dice", "Dice lesion-wise"),
        ("lesionwise_hd95", "HD95 lesion-wise"),
    ]

    rows: list[dict[str, Any]] = []

    for metric_key, metric_name in metric_configs:
        for region_name in REGION_TEST_NAMES:
            values_by_model = collect_model_values_by_region(
                results=results,
                region_name=region_name,
                metric_key=metric_key,
            )

            valid_models = [
                model
                for model, fold_values in values_by_model.items()
                if len(fold_values) > 0
            ]

            if len(valid_models) < 2:
                continue

            common_folds = sorted(
                set.intersection(
                    *[
                        set(values_by_model[model].keys())
                        for model in valid_models
                    ]
                )
            )

            if len(common_folds) < 3:
                continue

            aligned_values = [
                [
                    values_by_model[model][fold]
                    for fold in common_folds
                ]
                for model in valid_models
            ]

            if len(valid_models) > 2:
                statistic, p_value = friedmanchisquare(*aligned_values)

                rows.append(
                    {
                        "metric": metric_name,
                        "region": region_name,
                        "test": "Friedman global",
                        "comparison": "All models",
                        "n_blocks": len(common_folds),
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                        "p_value_corrected": None,
                    }
                )

            pairwise_rows: list[dict[str, Any]] = []
            raw_p_values: list[float] = []

            for model_a, model_b in combinations(valid_models, 2):
                pair_folds = sorted(
                    set(values_by_model[model_a].keys())
                    & set(values_by_model[model_b].keys())
                )

                values_a = [
                    values_by_model[model_a][fold]
                    for fold in pair_folds
                ]
                values_b = [
                    values_by_model[model_b][fold]
                    for fold in pair_folds
                ]

                if len(values_a) < 2:
                    continue

                try:
                    statistic, p_value = wilcoxon(values_a, values_b)
                    statistic = float(statistic)
                    p_value = float(p_value)
                except ValueError:
                    statistic = None
                    p_value = None

                pairwise_rows.append(
                    {
                        "metric": metric_name,
                        "region": region_name,
                        "test": "Paired Wilcoxon",
                        "comparison": f"{model_a} vs {model_b}",
                        "n_blocks": len(pair_folds),
                        "statistic": statistic,
                        "p_value": p_value,
                        "p_value_corrected": None,
                    }
                )

                raw_p_values.append(p_value if p_value is not None else 1.0)

            corrected_p_values = apply_holm_correction(raw_p_values)

            for row, corrected_p in zip(pairwise_rows, corrected_p_values):
                row["p_value_corrected"] = corrected_p
                rows.append(row)

    write_csv(output_path, rows)


def get_fold_metric(fold: dict[str, Any], metric_key: str) -> float | None:
    """Get a fold metric, converting units when needed.

    Args:
        fold: The fold-level result dictionary.
        metric_key: The metric to extract.

    Returns:
        The metric value as a float, or ``None`` if the metric is not available.
    """
    if metric_key == "test_dice":
        value = fold.get("test_lesionwise_mean_dice")
        return float(value) if is_valid_number(value) else None

    if metric_key == "test_hd95":
        value = fold.get("test_lesionwise_mean_hd95")
        return float(value) if is_valid_number(value) else None

    if metric_key == "train_time_h":
        value = fold.get("train_time_sec")
        return value / 3600 if is_valid_number(value) else None

    if metric_key == "inference_time_s":
        value = fold.get("inference_time_per_case_sec")
        return float(value) if is_valid_number(value) else None

    if metric_key == "train_memory_gb":
        value = fold.get("train_memory_mb")
        return value / 1024 if is_valid_number(value) else None

    if metric_key == "test_memory_gb":
        value = fold.get("test_memory_mb")
        return value / 1024 if is_valid_number(value) else None

    return None


def collect_model_fold_metric(
    results: list[dict[str, Any]],
    model_name: str,
    metric_key: str,
) -> dict[int, float]:
    """Collect one metric by fold for a given model.

    Args:
        results: The cross-validation results loaded from JSON files.
        model_name: The model to filter.
        metric_key: The metric to collect.

    Returns:
        A dictionary mapping fold identifiers to metric values.
    """
    values_by_fold: dict[int, float] = {}

    for result in results:
        if result["model"] != model_name:
            continue

        for fold in result.get("folds", []):
            fold_id = fold.get("fold")
            value = get_fold_metric(fold, metric_key)

            if fold_id is not None and is_valid_number(value):
                values_by_fold[int(fold_id)] = float(value)

    return values_by_fold


def save_swin_segmamba_comparison(
    output_path: Path,
    results: list[dict[str, Any]],
) -> None:
    """Compare Swin UNETR and SegMamba using paired Wilcoxon tests.

    The comparison includes segmentation performance and computational cost.

    Args:
        output_path: The path where the CSV file is saved.
        results: The cross-validation results loaded from JSON files.
    """
    model_a = "Swin UNETR"
    model_b = "SegMamba"

    metrics = [
        ("test_dice", "Lesion-wise Dice", "max"),
        ("test_hd95", "Lesion-wise HD95", "min"),
        ("train_time_h", "Training time (h)", "min"),
        ("inference_time_s", "Inference time per case (s)", "min"),
        ("train_memory_gb", "Training memory (GB)", "min"),
        ("test_memory_gb", "Test memory (GB)", "min"),
    ]

    rows: list[dict[str, Any]] = []

    for metric_key, metric_name, objective in metrics:
        values_a_by_fold = collect_model_fold_metric(
            results=results,
            model_name=model_a,
            metric_key=metric_key,
        )

        values_b_by_fold = collect_model_fold_metric(
            results=results,
            model_name=model_b,
            metric_key=metric_key,
        )

        common_folds = sorted(
            set(values_a_by_fold.keys()) & set(values_b_by_fold.keys())
        )

        values_a = [values_a_by_fold[fold] for fold in common_folds]
        values_b = [values_b_by_fold[fold] for fold in common_folds]

        mean_a = safe_mean(values_a)
        mean_b = safe_mean(values_b)

        difference = None
        relative_difference = None
        statistic = None
        p_value = None
        better_model = None
        if mean_a is not None and mean_b is not None:
            difference = mean_b - mean_a

            if mean_a != 0:
                relative_difference = (difference / mean_a) * 100

            if objective == "max":
                better_model = model_b if mean_b > mean_a else model_a

            elif objective == "min":
                better_model = model_b if mean_b < mean_a else model_a

        if len(values_a) >= 2 and len(values_b) >= 2:
            try:
                statistic, p_value = wilcoxon(values_a, values_b)
                statistic = float(statistic)
                p_value = float(p_value)
            except ValueError:
                statistic = None
                p_value = None

        rows.append(
            {
                "metric": metric_name,
                "objective": objective,
                "model_a": model_a,
                "model_b": model_b,
                "n_pairs": len(common_folds),
                "mean_swin_unetr": mean_a,
                "mean_segmamba": mean_b,
                "difference_segmamba_minus_swin": difference,
                "relative_difference_percent": relative_difference,
                "better_mean_model": better_model,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "significance": get_significance_stars(p_value),
            }
        )

    write_csv(output_path, rows)

    print("\n" + "=" * 75)
    print("COMPARISON: Swin UNETR vs SegMamba")
    print("=" * 75)

    for row in rows:
        print(
            f"{row['metric']}: "
            f"Swin={row['mean_swin_unetr']:.4f}, "
            f"SegMamba={row['mean_segmamba']:.4f}, "
            f"diff={row['difference_segmamba_minus_swin']:.4f}, "
            f"relative_diff={row['relative_difference_percent']:.2f}%, "
            f"best_mean_model={row['better_mean_model']}, "
            f"p={row['p_value']}, "
            f"{row['significance']}"
        )

    print(f"Table saved at: {output_path}")
    print("-" * 75 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--crossval_json",
        nargs="+",
        required=True,
        help="Paths to crossval_results.json files.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/plots",
        help="Directory where the plots and tables will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    """Run statistical tests and correlation analyses from cross-validation results."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_plot_style()

    paths = [Path(path) for path in args.crossval_json]
    results = collect_results(paths)
    fold_rows = collect_fold_level_points(results)

    save_correlation_table(
        output_path=output_dir / "performance_cost_correlations.csv",
        fold_rows=fold_rows,
    )

    save_region_significance_tests(
        output_path=output_dir / "region_significance_tests.csv",
        results=results,
    )

    save_fold_scatter_plot(
        output_path=output_dir / "fold_dice_vs_train_time.pdf",
        fold_rows=fold_rows,
        x_key="train_time_h",
        y_key="test_dice",
        xlabel="Tiempo de entrenamiento (h)",
        ylabel="Dice lesion-wise",
        title="Dice frente a tiempo de entrenamiento por partición",
    )

    save_fold_scatter_plot(
        output_path=output_dir / "fold_dice_vs_inference_time.pdf",
        fold_rows=fold_rows,
        x_key="inference_time_s",
        y_key="test_dice",
        xlabel="Tiempo de inferencia por caso (s)",
        ylabel="Dice lesion-wise",
        title="Dice frente a tiempo de inferencia por partición",
    )

    save_fold_scatter_plot(
        output_path=output_dir / "fold_dice_vs_train_memory.pdf",
        fold_rows=fold_rows,
        x_key="train_memory_gb",
        y_key="test_dice",
        xlabel="Memoria de entrenamiento (GB)",
        ylabel="Dice lesion-wise",
        title="Dice frente a memoria de entrenamiento por partición",
    )

    save_fold_scatter_plot(
        output_path=output_dir / "fold_dice_vs_test_memory.pdf",
        fold_rows=fold_rows,
        x_key="test_memory_gb",
        y_key="test_dice",
        xlabel="Memoria en prueba (GB)",
        ylabel="Dice lesion-wise",
        title="Dice frente a memoria en prueba por partición",
    )

    save_fold_scatter_plot(
        output_path=output_dir / "fold_hd95_vs_train_time.pdf",
        fold_rows=fold_rows,
        x_key="train_time_h",
        y_key="test_hd95",
        xlabel="Tiempo de entrenamiento (h)",
        ylabel="HD95 lesion-wise",
        title="HD95 frente a tiempo de entrenamiento por partición",
    )

    save_fold_scatter_plot(
        output_path=output_dir / "fold_hd95_vs_inference_time.pdf",
        fold_rows=fold_rows,
        x_key="inference_time_s",
        y_key="test_hd95",
        xlabel="Tiempo de inferencia por caso (s)",
        ylabel="HD95 lesion-wise",
        title="HD95 frente a tiempo de inferencia por partición",
    )

    save_fold_scatter_plot(
        output_path=output_dir / "fold_hd95_vs_train_memory.pdf",
        fold_rows=fold_rows,
        x_key="train_memory_gb",
        y_key="test_hd95",
        xlabel="Memoria de entrenamiento (GB)",
        ylabel="HD95 lesion-wise",
        title="HD95 frente a memoria de entrenamiento por partición",
    )

    save_fold_scatter_plot(
        output_path=output_dir / "fold_hd95_vs_test_memory.pdf",
        fold_rows=fold_rows,
        x_key="test_memory_gb",
        y_key="test_hd95",
        xlabel="Memoria en prueba (GB)",
        ylabel="HD95 lesion-wise",
        title="HD95 frente a memoria en prueba por partición",
    )

    et_netc_rows = collect_et_netc_points(results)

    check_et_netc_simpson_paradox(
        output_dir=output_dir,
        rows=et_netc_rows,
    )

    save_et_vs_netc_plot(
        output_path=output_dir / "scatter_et_vs_netc_context.pdf",
        rows=et_netc_rows,
        title="Coeficiente Dice NETC frente a Coeficiente Dice ET",
        with_local_lines=False,
    )

    save_et_vs_netc_plot(
        output_path=output_dir / "simpson_check_et_vs_netc.pdf",
        rows=et_netc_rows,
        title="Comprobación de Simpson: Dice ET frente a Dice NETC",
        with_local_lines=True,
    )

    save_model_significance_tests(
        output_path=output_dir / "model_significance_tests.csv",
        results=results,
    )

    save_model_significance_tests_by_region(
        output_path=output_dir / "model_significance_tests_by_region.csv",
        results=results,
    )


    save_swin_segmamba_comparison(
        output_path=output_dir / "swin_unetr_vs_segmamba_comparison.csv",
        results=results,
    )

    print(f"Analyses saved in: {output_dir}")
    print(f"Correlation table: {output_dir / 'performance_cost_correlations.csv'}")
    print(f"Simpson's paradox check: {output_dir / 'simpson_check_et_vs_netc.csv'}")
    print(f"Tests by region: {output_dir / 'region_significance_tests.csv'}")
    print(f"Tests by model and region: {output_dir / 'model_significance_tests_by_region.csv'}")


if __name__ == "__main__":
    main()


# python scripts/analyze_crossval_statistics.py 
# --crossval_json 
# ruta1/crossval_results.json 
# ruta2/crossval_results.json 
# ruta3/crossval_results.json 
# ruta4/crossval_results.json 
# --output_dir results/plots
