import torch
from tqdm import tqdm
from monai.inferers import sliding_window_inference

from src.utils.brats_regions import (
    init_region_stats,
    update_region_stats,
    finalize_region_stats,
)


def validate(model, data_loader, device, roi_size=(96, 96, 96), sw_batch_size=1, num_classes=5):
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
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                )

            pred_labels = torch.argmax(outputs, dim=1)
            update_region_stats(region_stats, pred_labels, labels)

    return finalize_region_stats(region_stats)
