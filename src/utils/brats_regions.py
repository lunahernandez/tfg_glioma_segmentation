import math
from collections import OrderedDict
import torch
from monai.metrics import HausdorffDistanceMetric
from monai.utils.enums import MetricReduction

BRATS_REGION_LABELS = OrderedDict({
    "ET": [1],
    "NETC": [2],
    "SNFH": [3],
    "RC": [4],
    "TC": [1, 2],        # ET + NETC
    "WT": [1, 2, 3],     # ET + NETC + SNFH
})


def init_region_stats():
    return {
        region_name: {
            "dice_values": [],
            "hd95_values": [],
        }
        for region_name in BRATS_REGION_LABELS
    }


def _to_label_map(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5 and x.shape[1] == 1:
        return x[:, 0].long()
    if x.ndim == 4:
        return x.long()
    raise ValueError(f"Forma no soportada para etiquetas: {tuple(x.shape)}")


def _region_mask(label_map: torch.Tensor, region_values) -> torch.Tensor:
    mask = torch.zeros_like(label_map, dtype=torch.bool)
    for value in region_values:
        mask |= (label_map == value)
    return mask


def _binary_dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    pred_sum = int(pred_mask.sum().item())
    gt_sum = int(gt_mask.sum().item())

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    intersection = int((pred_mask & gt_mask).sum().item())
    return (2.0 * intersection) / (pred_sum + gt_sum)


def _binary_hd95(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float | None:
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


def update_region_stats(stats, pred_labels: torch.Tensor, gt_labels: torch.Tensor):
    pred_labels = _to_label_map(pred_labels)
    gt_labels = _to_label_map(gt_labels)

    batch_size = pred_labels.shape[0]

    for b in range(batch_size):
        pred_case = pred_labels[b]
        gt_case = gt_labels[b]

        for region_name, region_values in BRATS_REGION_LABELS.items():
            pred_mask = _region_mask(pred_case, region_values)
            gt_mask = _region_mask(gt_case, region_values)

            dice = _binary_dice(pred_mask, gt_mask)
            hd95 = _binary_hd95(pred_mask, gt_mask)

            stats[region_name]["dice_values"].append(float(dice))
            if hd95 is not None:
                stats[region_name]["hd95_values"].append(float(hd95))


def finalize_region_stats(stats):
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
            "Métricas por subregión BraTS (ET, NETC, SNFH, RC, TC, WT). "
            "Esto no replica el evaluador lesion-wise oficial del challenge."
        ),
    }
