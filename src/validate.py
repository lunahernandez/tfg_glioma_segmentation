import torch
from tqdm import tqdm
from monai.inferers import sliding_window_inference

from src.utils.brats_regions import (
    init_region_stats,
    update_region_stats,
    finalize_region_stats,
)


def validate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 1,
) -> dict:
    """Evaluates the model on the validation set.

    The model generates predictions using ``sliding_window_inference``
    and calculates metrics for each BraTS subregion based on the
    predicted and ground truth labels.

    Args:
        model: Segmentation model to evaluate.
        data_loader: Data loader for the validation set.
        device: Device on which the evaluation is performed.
        roi_size: Size of the region of interest.
        sw_batch_size: Number of windows processed simultaneously in
            ``sliding_window_inference``.

    Returns:
        A dictionary with the validation metrics.
    """
    model.eval()

    use_amp = device.type == "cuda"
    region_stats = init_region_stats()

    progress_bar = tqdm(
        data_loader,
        desc="Validation",
        leave=False,
    )

    with torch.no_grad():
        for batch_data in progress_bar:
            images = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                )

            pred_labels = torch.argmax(outputs, dim=1)
            update_region_stats(region_stats, pred_labels, labels)

    return finalize_region_stats(region_stats)