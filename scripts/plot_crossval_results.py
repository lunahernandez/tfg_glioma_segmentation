from pathlib import Path
import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from utils_results import (
    REGION_NAMES,
    REGION_COLORS,
    collect_results,
    collect_epoch_stats,
    get_model_color,
    is_valid_number,
    none_to_nan,
)


def set_plot_style() -> None:
    """Set the common visual style used by all generated plots."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save_current_figure(output_path: Path) -> None:
    """Save the current figure in PDF format.

    Args:
        output_path: Path where the PDF figure is saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_suffix(".pdf")

    plt.savefig(output_path)


def format_axes(
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid_axis: str = "y",
) -> None:
    """Apply common axis formatting to the current plot.

    Args:
        title: Plot title.
        xlabel: Optional x-axis label.
        ylabel: Optional y-axis label.
        grid_axis: Axis where the grid is drawn.
    """
    ax = plt.gca()
    ax.set_title(title, fontweight="bold")

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.grid(axis=grid_axis, linestyle="--", alpha=0.35)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("black")


def set_ylim_above_error_bars(
    means: list[float | None],
    stds: list[float | None],
    padding_factor: float = 1.12,
) -> None:
    """Set the y-axis upper limit above the highest mean plus standard deviation.

    Args:
        means: Mean values shown in the plot.
        stds: Standard deviation values used as error bars.
        padding_factor: Multiplicative padding added above the highest value.
    """
    ax = plt.gca()
    max_peak = 0.0

    for mean_val, std_val in zip(means, stds):
        if is_valid_number(mean_val):
            peak = float(mean_val) + (
                float(std_val) if is_valid_number(std_val) else 0.0
            )
            max_peak = max(max_peak, peak)

    if max_peak > 0:
        ax.set_ylim(top=max_peak * padding_factor)


def add_bar_labels(
    means: list[float | None],
    stds: list[float | None] | None = None,
    decimals: int = 4,
) -> None:
    """Add numeric labels above bars and their error bars.

    Args:
        means: Mean values represented by the bars.
        stds: Optional standard deviation values used as error bars.
        decimals: Number of decimal places shown in the labels.
    """
    ax = plt.gca()
    y_min, y_max = ax.get_ylim()
    vertical_padding = (y_max - y_min) * 0.02

    if stds is None:
        stds = [0.0] * len(means)

    for patch, mean_val, std_val in zip(ax.patches, means, stds):
        if not is_valid_number(mean_val):
            continue

        std_offset = float(std_val) if is_valid_number(std_val) else 0.0
        label_y = float(mean_val) + std_offset + vertical_padding

        ax.text(
            patch.get_x() + patch.get_width() / 2,
            label_y,
            f"{float(mean_val):.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def save_bar_plot(
    output_path: Path,
    labels: list[str],
    means: list[float | None],
    stds: list[float | None],
    ylabel: str,
    title: str,
    decimals: int = 4,
) -> None:
    """Save a vertical bar plot with error bars.

    Args:
        output_path: Path where the PDF figure is saved.
        labels: Labels shown on the x-axis.
        means: Mean values represented by the bars.
        stds: Standard deviation values used as error bars.
        ylabel: Y-axis label.
        title: Plot title.
        decimals: Number of decimal places shown in the bar labels.
    """
    x = np.arange(len(labels))
    y = none_to_nan(means)
    yerr = none_to_nan(stds)

    colors = [
        get_model_color(label)
        for index, label in enumerate(labels)
    ]

    plt.figure(figsize=(9, 5))

    plt.bar(
        x,
        y,
        yerr=yerr,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=1.1,
        alpha=0.9,
    )

    plt.xticks(x, labels, rotation=0, ha="center")
    format_axes(title=title, ylabel=ylabel)

    set_ylim_above_error_bars(
        means=means,
        stds=stds,
        padding_factor=1.12,
    )

    add_bar_labels(means=means, stds=stds, decimals=decimals)

    plt.tight_layout()
    save_current_figure(output_path)
    plt.close()


def save_lollipop_plot(
    output_path: Path,
    labels: list[str],
    means: list[float | None],
    stds: list[float | None],
    xlabel: str,
    title: str,
    decimals: int = 2,
) -> None:
    """Save a horizontal lollipop plot with error bars.

    Args:
        output_path: Path where the PDF figure is saved.
        labels: Labels shown on the y-axis.
        means: Mean values represented by the points.
        stds: Standard deviation values used as horizontal error bars.
        xlabel: X-axis label.
        title: Plot title.
        decimals: Number of decimal places shown in the value labels.
    """
    y_pos = np.arange(len(labels))
    values = np.array(none_to_nan(means), dtype=float)
    errors = np.array(none_to_nan(stds), dtype=float)

    colors = [
        get_model_color(label)
        for index, label in enumerate(labels)
    ]

    plt.figure(figsize=(8.5, 5))
    ax = plt.gca()

    for i, (value, error, color) in enumerate(zip(values, errors, colors)):
        if np.isnan(value):
            continue

        error_value = 0.0 if np.isnan(error) else float(error)

        ax.hlines(
            y=i,
            xmin=0,
            xmax=value,
            color=color,
            linewidth=3,
            alpha=0.85,
        )

        ax.errorbar(
            x=value,
            y=i,
            xerr=error_value,
            fmt="o",
            color=color,
            markeredgecolor="black",
            markersize=8,
            capsize=5,
            linewidth=1.2,
        )

        label_x = value + (np.nanmax(values) * 0.02)

        ax.text(
            label_x,
            i,
            f"{value:.{decimals}f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    format_axes(
        title=title,
        xlabel=xlabel,
        grid_axis="x",
    )

    valid_value_mask = ~np.isnan(values)

    valid_values = values[valid_value_mask]
    valid_errors = errors[valid_value_mask]
    valid_errors = np.nan_to_num(valid_errors, nan=0.0)

    if len(valid_values) > 0:
        max_peak = float(np.max(valid_values + valid_errors))
        ax.set_xlim(right=max_peak * 1.22)

    plt.tight_layout()
    save_current_figure(output_path)
    plt.close()


def save_grouped_region_plot(
    output_path: Path,
    results: list[dict[str, Any]],
    metric_key: str,
    std_key: str,
    ylabel: str,
    title: str,
) -> None:
    """Save a grouped bar plot comparing models across BraTS regions.

    Args:
        output_path: Path where the PDF figure is saved.
        results: Cross-validation results loaded from the input JSON files.
        metric_key: Key used to get the regional mean metric.
        std_key: Key used to get the regional standard deviation.
        ylabel: Y-axis label.
        title: Plot title.
    """
    x = np.arange(len(REGION_NAMES))
    width = 0.8 / max(1, len(results))

    plt.figure(figsize=(11, 5))

    plt.axvline(
        x=3.5,
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )

    all_means: list[float | None] = []
    all_stds: list[float | None] = []

    for i, result in enumerate(results):
        means = []
        stds = []

        for region in REGION_NAMES:
            region_data = result["regions"].get(region, {})
            means.append(region_data.get(metric_key))
            stds.append(region_data.get(std_key))

        offset = (i - (len(results) - 1) / 2) * width

        plt.bar(
            x + offset,
            none_to_nan(means),
            width,
            yerr=none_to_nan(stds),
            capsize=3,
            label=result["model"],
            color=get_model_color(result["model"]),
            edgecolor="black",
            linewidth=0.9,
            alpha=0.9,
        )

        all_means.extend(means)
        all_stds.extend(stds)

    plt.xticks(x, REGION_NAMES)
    format_axes(title=title, ylabel=ylabel)

    set_ylim_above_error_bars(
        means=all_means,
        stds=all_stds,
        padding_factor=1.12,
    )

    plt.legend(frameon=True)
    plt.tight_layout()
    save_current_figure(output_path)
    plt.close()


def save_memory_plot(
    output_path: Path,
    results: list[dict[str, Any]],
) -> None:
    """Save a grouped bar plot with training and testing memory.

    Args:
        output_path: Path where the PDF figure is saved.
        results: Cross-validation results loaded from the input JSON files.
    """
    models = [result["model"] for result in results]
    x = np.arange(len(models))
    width = 0.35

    train_means = none_to_nan([r["train_memory_gb_mean"] for r in results])
    test_means = none_to_nan([r["test_memory_gb_mean"] for r in results])
    train_stds = none_to_nan([r["train_memory_gb_std"] for r in results])
    test_stds = none_to_nan([r["test_memory_gb_std"] for r in results])

    train_color = "#0069A3"
    test_color = "#FFA100"

    plt.figure(figsize=(9, 5))
    ax = plt.gca()

    bars_train = plt.bar(
        x - width / 2,
        train_means,
        width,
        yerr=train_stds,
        capsize=5,
        label="Entrenamiento",
        color=train_color,
        edgecolor="black",
        linewidth=1.1,
        alpha=0.9,
    )

    bars_test = plt.bar(
        x + width / 2,
        test_means,
        width,
        yerr=test_stds,
        capsize=5,
        label="Prueba",
        color=test_color,
        edgecolor="black",
        linewidth=1.1,
        alpha=0.9,
    )

    set_ylim_above_error_bars(
        means=train_means + test_means,
        stds=train_stds + test_stds,
        padding_factor=1.15,
    )

    y_min, y_max = ax.get_ylim()
    vertical_padding = (y_max - y_min) * 0.02

    for bars, means, stds in [
        (bars_train, train_means, train_stds),
        (bars_test, test_means, test_stds),
    ]:
        for bar, mean_val, std_val in zip(bars, means, stds):
            if not is_valid_number(mean_val):
                continue

            std_offset = float(std_val) if is_valid_number(std_val) else 0.0
            label_y = float(mean_val) + std_offset + vertical_padding

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{float(mean_val):.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.xticks(x, models, rotation=0, ha="center")

    format_axes(
        title="Memoria máxima utilizada por modelo",
        ylabel="Memoria (GB)",
    )

    plt.legend(frameon=True)
    plt.tight_layout()
    save_current_figure(output_path)
    plt.close()


def save_dice_vs_cost_plot(
    output_path: Path,
    results: list[dict[str, Any]],
    cost_key: str,
    xlabel: str,
    title: str,
) -> None:
    """Save a model-level scatter plot comparing Dice and computational cost.

    Args:
        output_path: Path where the PDF figure is saved.
        results: Cross-validation results loaded from the input JSON files.
        cost_key: Result key used as the x-axis variable.
        xlabel: X-axis label.
        title: Plot title.
    """
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    x_values: list[float] = []
    y_values: list[float] = []

    for i, result in enumerate(results):
        x = result.get(cost_key)
        y = result.get("dice_mean")

        if x is None or y is None:
            continue

        x_values.append(float(x))
        y_values.append(float(y))

        plt.scatter(
            x,
            y,
            color=get_model_color(result["model"]),
            edgecolor="black",
            linewidth=0.8,
            s=80,
            zorder=3,
        )

    if not x_values or not y_values:
        plt.close()
        return

    x_threshold = sum(x_values) / len(x_values)
    y_max = max(y_values)
    y_min = min(y_values)
    y_range = y_max - y_min if y_max > y_min else 0.1

    for result in results:
        x = result.get(cost_key)
        y = result.get("dice_mean")

        if x is None or y is None:
            continue

        pos_x = float(x)
        pos_y = float(y)

        if pos_x > x_threshold:
            ha = "right"
            offset_x = -8
        else:
            ha = "left"
            offset_x = 8

        if pos_y > (y_max - 0.15 * y_range):
            va = "center"
            offset_y = 0
        else:
            va = "bottom"
            offset_y = 6

        plt.annotate(
            result["model"],
            (x, y),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            ha=ha,
            va=va,
            fontsize=10,
            zorder=4,
        )

    format_axes(
        title=title,
        xlabel=xlabel,
        ylabel="Dice lesion-wise medio",
        grid_axis="both",
    )

    x_min = min(x_values)
    x_max = max(x_values)
    x_range = x_max - x_min if x_max > x_min else 1.0

    ax.set_xlim(
        left=x_min - x_range * 0.05,
        right=x_max + x_range * 0.05,
    )
    ax.set_ylim(
        bottom=y_min - y_range * 0.05,
        top=y_max + y_range * 0.08,
    )

    plt.subplots_adjust(left=0.12, right=0.88, bottom=0.15, top=0.90)
    save_current_figure(output_path)
    plt.close()


def save_bubble_tradeoff_plot(
    output_path: Path,
    results: list[dict[str, Any]],
) -> None:
    """Save a bubble plot combining Dice, inference time and test memory.

    Args:
        output_path: Path where the PDF figure is saved.
        results: Cross-validation results loaded from the input JSON files.
    """
    fig, ax = plt.subplots(figsize=(8, 5.2))

    x_values: list[float] = []
    y_values: list[float] = []
    memories: list[float] = []
    models: list[str] = []

    for result in results:
        x = result.get("inference_time_s_mean")
        y = result.get("dice_mean")
        mem = result.get("test_memory_gb_mean")

        if x is None or y is None or mem is None:
            continue

        x_values.append(float(x))
        y_values.append(float(y))
        memories.append(float(mem))
        models.append(result["model"])

    if not x_values:
        plt.close()
        return

    max_memory = max(memories)
    scaled_sizes = []

    for mem in memories:
        scaled_size = 60 + (mem / max_memory) ** 1.4 * 650
        scaled_sizes.append(scaled_size)

    for i, (x, y, size, model) in enumerate(
        zip(x_values, y_values, scaled_sizes, models)
    ):
        ax.scatter(
            x,
            y,
            s=size,
            color=get_model_color(model),
            edgecolor="black",
            linewidth=1.2,
            alpha=0.85,
            zorder=3,
        )

    x_threshold = sum(x_values) / len(x_values)

    for x, y, model, mem in zip(x_values, y_values, models, memories):
        ha = "right" if x > x_threshold else "left"
        offset_x = -18 if x > x_threshold else 18

        ax.annotate(
            f"{model}\n({mem:.1f} GB)",
            (x, y),
            textcoords="offset points",
            xytext=(offset_x, 0),
            ha=ha,
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=4,
        )

    format_axes(
        title="Relación entre precisión y coste en inferencia",
        xlabel="Tiempo de inferencia por caso (s)",
        ylabel="Dice lesion-wise medio",
        grid_axis="both",
    )

    x_range = max(x_values) - min(x_values) if max(x_values) > min(x_values) else 1.0
    y_range = max(y_values) - min(y_values) if max(y_values) > min(y_values) else 0.1

    ax.set_xlim(
        left=min(x_values) - x_range * 0.18,
        right=max(x_values) + x_range * 0.22,
    )
    ax.set_ylim(
        bottom=min(y_values) - y_range * 0.12,
        top=max(y_values) + y_range * 0.12,
    )

    ax.text(
        0.5,
        -0.18,
        "Nota: el área de la burbuja representa la memoria máxima usada durante la prueba.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )

    fig.tight_layout()
    save_current_figure(output_path)
    plt.close()


def save_training_curve_plot(
    output_path: Path,
    results: list[dict[str, Any]],
    metric_key: str,
    ylabel: str,
    title: str,
    show_markers: bool = False,
) -> None:
    """Save a training or validation curve aggregated across CV folds.

    Args:
        output_path: Path where the PDF figure is saved.
        results: Cross-validation results loaded from the input JSON files.
        metric_key: History key to plot across epochs.
        ylabel: Y-axis label.
        title: Plot title.
        show_markers: Whether to show markers on the curve.
    """
    plt.figure(figsize=(9, 5))

    for i, result in enumerate(results):
        epochs, means, stds = collect_epoch_stats(
            folds=result["folds"],
            metric_key=metric_key,
        )

        if len(epochs) == 0:
            continue

        epochs_array = np.array(epochs)
        means_array = np.array(means, dtype=float)
        stds_array = np.array(stds, dtype=float)

        color = get_model_color(result["model"])
        marker = "o" if show_markers else None

        plt.plot(
            epochs_array,
            means_array,
            marker=marker,
            linewidth=2.0,
            markersize=5,
            color=color,
            label=result["model"],
        )

        lower = means_array - stds_array
        upper = means_array + stds_array

        plt.fill_between(
            epochs_array,
            lower,
            upper,
            alpha=0.18,
            color=color,
        )

    format_axes(
        title=title,
        xlabel="Época",
        ylabel=ylabel,
        grid_axis="both",
    )

    plt.legend(frameon=True)
    plt.tight_layout()
    save_current_figure(output_path)
    plt.close()


def save_aggregated_region_boxplot(
    output_path: Path,
    results: list[dict[str, Any]],
    title: str,
    metric_type: str = "dice",
) -> None:
    """Save a boxplot of regional scores aggregated across folds and models.

    Args:
        output_path: Path where the PDF figure is saved.
        results: Cross-validation results loaded from the input JSON files.
        title: Plot title.
        metric_type: Metric to plot. Accepted values are "dice" and "hd95".
    """
    region_data_collected: list[list[float]] = []
    plot_regions: list[str] = []

    metric_key = "lesionwise_hd95" if metric_type == "hd95" else "lesionwise_dice"
    ylabel = "HD95 lesion-wise" if metric_type == "hd95" else "Coeficiente Dice lesion-wise"

    for region in REGION_NAMES:
        combined_values: list[float] = []

        for result in results:
            for fold in result.get("folds", []):
                label_metrics = fold.get("test_lesionwise_by_label", {})
                region_data = label_metrics.get(region, {})
                value = region_data.get(metric_key)

                if is_valid_number(value):
                    combined_values.append(float(value))

        if combined_values:
            region_data_collected.append(combined_values)
            plot_regions.append(region)

    if not region_data_collected:
        print(f"No values of {metric_type.upper()} were found by region.")
        return

    plt.figure(figsize=(10, 5))

    if "RC" in plot_regions and "TC" in plot_regions:
        rc_index = plot_regions.index("RC") + 1
        tc_index = plot_regions.index("TC") + 1
        separator_x = (rc_index + tc_index) / 2.0

        plt.axvline(
            x=separator_x,
            color="#666666",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )

    box_data = plt.boxplot(
        region_data_collected,
        tick_labels=plot_regions,
        patch_artist=True,
        showmeans=True,
        medianprops={"color": "black", "linewidth": 1.2},
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 5,
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markersize": 4,
            "alpha": 0.6,
        },
    )

    for patch, region_name in zip(box_data["boxes"], plot_regions):
        color = REGION_COLORS.get(region_name, "#cccccc")
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.1)
        patch.set_alpha(0.85)

    format_axes(
        title=title,
        ylabel=ylabel,
        grid_axis="y",
    )

    plt.tight_layout()
    save_current_figure(output_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
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
        help="Directory where the plots will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    """Create plots from one or more cross-validation result files."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_plot_style()

    paths = [Path(path) for path in args.crossval_json]
    results = collect_results(paths)
    models = [result["model"] for result in results]

    save_bar_plot(
        output_path=output_dir / "global_dice_by_model.pdf",
        labels=models,
        means=[r["dice_mean"] for r in results],
        stds=[r["dice_std"] for r in results],
        ylabel="Coeficiente Dice lesion-wise medio",
        title="Coeficiente Dice global por modelo",
    )

    save_bar_plot(
        output_path=output_dir / "global_hd95_by_model.pdf",
        labels=models,
        means=[r["hd95_mean"] for r in results],
        stds=[r["hd95_std"] for r in results],
        ylabel="HD95 lesion-wise medio",
        title="HD95 global por modelo",
        decimals=2,
    )

    save_grouped_region_plot(
        output_path=output_dir / "dice_by_region.pdf",
        results=results,
        metric_key="lesionwise_dice_mean",
        std_key="lesionwise_dice_std",
        ylabel="Coeficiente Dice lesion-wise medio",
        title="Coeficiente Dice por región BraTS",
    )

    save_grouped_region_plot(
        output_path=output_dir / "hd95_by_region.pdf",
        results=results,
        metric_key="lesionwise_hd95_mean",
        std_key="lesionwise_hd95_std",
        ylabel="HD95 lesion-wise medio",
        title="HD95 por región BraTS",
    )

    save_lollipop_plot(
        output_path=output_dir / "training_time_by_model.pdf",
        labels=models,
        means=[r["train_time_h_mean"] for r in results],
        stds=[r["train_time_h_std"] for r in results],
        xlabel="Tiempo de entrenamiento (h)",
        title="Tiempo de entrenamiento por modelo",
        decimals=2,
    )

    save_lollipop_plot(
        output_path=output_dir / "inference_time_by_model.pdf",
        labels=models,
        means=[r["inference_time_s_mean"] for r in results],
        stds=[r["inference_time_s_std"] for r in results],
        xlabel="Tiempo de inferencia por caso (s)",
        title="Tiempo de inferencia por modelo",
        decimals=2,
    )

    save_memory_plot(
        output_path=output_dir / "memory_by_model.pdf",
        results=results,
    )

    save_dice_vs_cost_plot(
        output_path=output_dir / "dice_vs_inference_time.pdf",
        results=results,
        cost_key="inference_time_s_mean",
        xlabel="Tiempo de inferencia por caso (s)",
        title="Dice frente a tiempo de inferencia",
    )

    save_dice_vs_cost_plot(
        output_path=output_dir / "dice_vs_training_memory.pdf",
        results=results,
        cost_key="train_memory_gb_mean",
        xlabel="Memoria de entrenamiento (GB)",
        title="Dice frente a memoria de entrenamiento",
    )

    save_bubble_tradeoff_plot(
        output_path=output_dir / "bubble_global_tradeoff.pdf",
        results=results,
    )

    save_training_curve_plot(
        output_path=output_dir / "train_loss_by_epoch.pdf",
        results=results,
        metric_key="train_loss",
        ylabel="Pérdida media de entrenamiento",
        title="Evolución de la pérdida de entrenamiento",
        show_markers=False,
    )

    save_training_curve_plot(
        output_path=output_dir / "val_dice_by_epoch.pdf",
        results=results,
        metric_key="val_mean_dice",
        ylabel="Coeficiente Dice medio en validación",
        title="Evolución del coeficiente Dice en validación",
        show_markers=True,
    )

    save_training_curve_plot(
        output_path=output_dir / "val_hd95_by_epoch.pdf",
        results=results,
        metric_key="val_mean_hd95",
        ylabel="HD95 medio en validación",
        title="Evolución de HD95 en validación",
        show_markers=True,
    )

    save_aggregated_region_boxplot(
        output_path=output_dir / "boxplot_dice_by_region_aggregated.pdf",
        results=results,
        title="Distribución del coeficiente Dice por región BraTS",
        metric_type="dice",
    )

    save_aggregated_region_boxplot(
        output_path=output_dir / "boxplot_hd95_by_region_aggregated.pdf",
        results=results,
        title="Distribución de HD95 por región BraTS",
        metric_type="hd95",
    )

    print(f"Graphics saved in PDF format: {output_dir}")


if __name__ == "__main__":
    main()


# python scripts/plot_crossval_results.py 
# --crossval_json 
# experiments/brats_unet3d_roi128_bs1_nworkers4_cv5_labelmapfix/crossval_results.json 
# experiments/brats_resunet3d_roi128_bs1_nworkers4_cv5_labelmapfix/crossval_results.json 
# experiments/brats_swin_unetr_roi128_bs1_nworkers4_cv5_labelmapfix/crossval_results.json 
# experiments/brats_segmamba_roi128_bs1_nworkers4_cv5_labelmapfix/crossval_results.json 
# --output_dir results/plots
