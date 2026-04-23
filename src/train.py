import time
import torch
from tqdm import tqdm
from monai.losses import DiceCELoss
from src.utils.meters import AverageMeter
from src.validate import validate
from src.utils.checkpoints import save_checkpoint


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    max_epochs,
    val_every,
    experiment_dir,
    roi_size=(96, 96, 96),
    sw_batch_size=1,
    num_classes=5,
):
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    best_metric = -1.0
    best_metric_epoch = -1
    history = []

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_start = time.perf_counter()

    for epoch in range(max_epochs):
        print("-" * 50)
        print(f"epoch {epoch + 1}/{max_epochs}")

        epoch_start = time.perf_counter()

        model.train()
        epoch_loss_meter = AverageMeter()

        progress_bar = tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{max_epochs}",
            leave=True,
        )

        for batch_data in progress_bar:
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss_meter.update(loss.item(), n=inputs.shape[0])

            progress_bar.set_postfix({
                "loss": f"{epoch_loss_meter.avg:.4f}"
            })

        epoch_time = time.perf_counter() - epoch_start

        print(f"epoch {epoch + 1} loss: {epoch_loss_meter.avg:.4f}")
        print(f"epoch {epoch + 1} time: {epoch_time:.2f} s")

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": float(epoch_loss_meter.avg),
            "epoch_time_sec": float(epoch_time),
        }

        if (epoch + 1) % val_every == 0:
            metrics = validate(
                model=model,
                data_loader=val_loader,
                device=device,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                num_classes=num_classes,
            )

            val_mean_dice = metrics["mean_dice"]
            val_mean_hd95 = metrics["mean_hd95"]

            print(f"val mean Dice: {val_mean_dice:.4f}" if val_mean_dice is not None else "val mean Dice: None")
            print(f"val mean HD95: {val_mean_hd95:.4f}" if val_mean_hd95 is not None else "val mean HD95: None")

            epoch_info["val_mean_dice"] = val_mean_dice
            epoch_info["val_mean_hd95"] = val_mean_hd95
            epoch_info["val_regions"] = metrics["regions"]

            current_metric = val_mean_dice if val_mean_dice is not None else -1.0

            if current_metric > best_metric:
                best_metric = current_metric
                best_metric_epoch = epoch + 1

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    best_metric=best_metric,
                    save_path=experiment_dir / "best_model.pt",
                )

                print("nuevo mejor modelo guardado")

        history.append(epoch_info)

    total_time = time.perf_counter() - total_start

    print(f"Best validation mean Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
    print(f"Total training time: {total_time:.2f} s")

    return {
        "best_val_dice": float(best_metric),
        "best_epoch": int(best_metric_epoch),
        "total_train_time_sec": float(total_time),
        "history": history,
    }
