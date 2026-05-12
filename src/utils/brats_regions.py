import math
from collections import OrderedDict
from typing import Any

import torch
from monai.metrics import HausdorffDistanceMetric
from monai.utils.enums import MetricReduction


BRATS_REGION_LABELS = OrderedDict({
    "NETC": [1],
    "SNFH": [2],
    "ET": [3],
    "RC": [4],
    "TC": [1, 3],        # NETC + ET
    "WT": [1, 2, 3],     # NETC + SNFH + ET
})


def initialize_region_metrics() -> dict[str, dict[str, list[float]]]:
    """Initializes a structure to store metrics per BraTS region.

    Creates a dictionary with an entry for each region defined in
    ``BRATS_REGION_LABELS``. For each region, two empty lists are created:
    one for Dice coefficient values and another for HD95 values.

    Returns:
        A dictionary with initialized metric lists for each region.
    """
    return {
        region_name: {
            "dice_values": [],
            "hd95_values": [],
        }
        for region_name in BRATS_REGION_LABELS
    }


def convert_to_label_map(x: torch.Tensor) -> torch.Tensor:
    """Converts network outputs or labels to a 3D/4D label map.

    Removes the channel dimension if it is equal to 1 (e.g., from [B, 1, H, W, D] 
    to [B, H, W, D]) and converts the tensor to a long integer type.

    Args:
        x: Input tensor (a prediction or ground truth).

    Returns:
        A label map tensor of type torch.long.

    Raises:
        ValueError: If the input tensor shape is not supported (not 4D, or not 5D with 1 channel).
    """
    if x.ndim == 5 and x.shape[1] == 1:
        return x[:, 0].long()
    if x.ndim == 4:
        return x.long()
    raise ValueError(f"Unsupported shape for labels: {tuple(x.shape)}")


def create_region_mask(label_map: torch.Tensor, region_values: list[int]) -> torch.Tensor:
    """Creates a boolean mask indicating the presence of any target label.

    Args:
        label_map: Tensor containing categorical integer labels.
        region_values: List of integer values that make up the target region.

    Returns:
        A boolean tensor with the same shape as `label_map`, where True 
        indicates the presence of the region.
    """
    mask = torch.zeros_like(label_map, dtype=torch.bool)
    for value in region_values:
        mask = mask | (label_map == value)
    return mask


def calculate_binary_dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Calculates the binary Dice coefficient between two masks.

    Args:
        pred_mask: Boolean tensor representing the predicted region.
        gt_mask: Boolean tensor representing the ground truth region.

    Returns:
        The calculated Dice score. Returns 1.0 if both masks are empty,
        and 0.0 if only one of them is empty.
    """
    pred_sum = int(pred_mask.sum().item())
    gt_sum = int(gt_mask.sum().item())

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    intersection = int((pred_mask & gt_mask).sum().item())
    return (2.0 * intersection) / (pred_sum + gt_sum)


def calculate_binary_hd95(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float | None:
    """Calculates the 95th percentile Hausdorff Distance (HD95) between two masks.

    Args:
        pred_mask: Boolean tensor representing the predicted region.
        gt_mask: Boolean tensor representing the ground truth region.

    Returns:
        The calculated HD95 value. Returns 0.0 if both masks are empty,
        and None if only one is empty.
    """
    pred_sum = int(pred_mask.sum().item())
    gt_sum = int(gt_mask.sum().item())

    if pred_sum == 0 and gt_sum == 0:
        return 0.0
    if pred_sum == 0 or gt_sum == 0:
        return None

    metric = HausdorffDistanceMetric(
        include_background=True,
        percentile=95,
        reduction=MetricReduction.MEAN,
    )

    y_pred = pred_mask.unsqueeze(0).unsqueeze(0).float()
    y = gt_mask.unsqueeze(0).unsqueeze(0).float()

    metric(y_pred=y_pred, y=y)
    value = metric.aggregate().item()
    metric.reset()

    if math.isnan(value):
        return None

    return float(value)


def update_region_metrics(
    stats: dict[str, dict[str, list[float]]], 
    pred_labels: torch.Tensor, 
    gt_labels: torch.Tensor
) -> None:
    """Updates the metric statistics from a new batch.

    Iterates over the batch, calculates the Dice and HD95 values for each 
    defined BraTS region, and appends the results to the statistics dictionary.

    Args:
        stats: Dictionary containing the lists of metrics to be updated.
        pred_labels: Predicted label map tensor.
        gt_labels: Ground truth label map tensor.
    """
    pred_labels = convert_to_label_map(pred_labels)
    gt_labels = convert_to_label_map(gt_labels)

    batch_size = pred_labels.shape[0]

    for batch in range(batch_size):
        pred_case = pred_labels[batch]
        gt_case = gt_labels[batch]

        for region_name, region_values in BRATS_REGION_LABELS.items():
            pred_mask = create_region_mask(pred_case, region_values)
            gt_mask = create_region_mask(gt_case, region_values)

            dice = calculate_binary_dice(pred_mask, gt_mask)
            hd95 = calculate_binary_hd95(pred_mask, gt_mask)

            stats[region_name]["dice_values"].append(float(dice))
            if hd95 is not None:
                stats[region_name]["hd95_values"].append(float(hd95))


def summarize_region_metrics(stats: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    """Calculates the mean metrics for all regions and aggregates the final results.

    Args:
        stats: Dictionary containing lists of the computed Dice and HD95 values per region.

    Returns:
        A dictionary containing the mean Dice and HD95 for each specific region,
        along with the global mean values across all regions.
    """
    region_results = {}

    dice_means = []
    hd95_means = []

    for region_name, values in stats.items():
        dice_vals = values["dice_values"]
        hd95_vals = values["hd95_values"]

        dice_mean = sum(dice_vals) / len(dice_vals) if len(dice_vals) > 0 else None
        hd95_mean = sum(hd95_vals) / len(hd95_vals) if len(hd95_vals) > 0 else None

        if dice_mean is not None:
            dice_mean = float(dice_mean)
            dice_means.append(dice_mean)

        if hd95_mean is not None:
            hd95_mean = float(hd95_mean)
            hd95_means.append(hd95_mean)

        region_results[region_name] = {
            "dice": dice_mean,
            "hd95": hd95_mean,
            "num_cases": len(dice_vals),
            "num_valid_hd95_cases": len(hd95_vals),
        }

    mean_dice = float(sum(dice_means) / len(dice_means)) if len(dice_means) > 0 else None
    mean_hd95 = float(sum(hd95_means) / len(hd95_means)) if len(hd95_means) > 0 else None

    return {
        "regions": region_results,
        "mean_dice": mean_dice,
        "mean_hd95": mean_hd95,
        "metric_note": (
            "Metrics per BraTS subregion (ET, NETC, SNFH, RC, TC, WT). "
            "This does not replicate the official lesion-wise challenge evaluator."
        ),
    }
