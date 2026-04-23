import os
import time
import tempfile
from pathlib import Path

import nibabel as nib
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from monai.inferers import sliding_window_inference

from src.utils.brats_lesionwise import get_LesionWiseResults


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return float(sum(values) / len(values)) if len(values) > 0 else None


def _df_to_records(df):
    records = df.to_dict(orient='records')
    cleaned_records = []
    
    for row in records:
        clean_row = {}
        for k, v in row.items():
            # 1. Si es una lista o un array de numpy, lo guardamos directamente
            if isinstance(v, (list, np.ndarray)):
                # Convertimos a lista normal para evitar errores de serialización JSON
                clean_row[k] = v.tolist() if isinstance(v, np.ndarray) else v
            # 2. Si es un valor nulo (NaN), lo convertimos a None (null en JSON)
            elif pd.isna(v):
                clean_row[k] = None
            # 3. Cualquier otro valor escalar (números, strings, booleanos)
            else:
                clean_row[k] = v
                
        cleaned_records.append(clean_row)
        
    return cleaned_records


def evaluate_test(
    model,
    test_loader,
    device,
    roi_size=(96, 96, 96),
    sw_batch_size=1,
    num_classes=5,
    output_dir=None,
):
    model.eval()

    if output_dir is None:
        raise ValueError("output_dir es obligatorio para guardar predicciones y métricas lesion-wise.")

    output_dir = Path(output_dir)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    total_time = 0.0
    n_cases = 0
    use_amp = device.type == "cuda"

    per_case_results = []
    by_label = {
        "WT": {"dice": [], "hd95": []},
        "TC": {"dice": [], "hd95": []},
        "NETC": {"dice": [], "hd95": []},
        "SNFH": {"dice": [], "hd95": []},
        "ET": {"dice": [], "hd95": []},
        "RC": {"dice": [], "hd95": []},
    }

    progress_bar = tqdm(
        test_loader,
        desc="Test",
        leave=True,
    )

    with torch.no_grad():
        for batch_data in progress_bar:
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"]
            case_ids = batch_data["case_id"]

            start = time.perf_counter()

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            total_time += elapsed
            n_cases += inputs.shape[0]

            pred_labels = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            gt_labels = labels[:, 0].detach().cpu().numpy()

            for b in range(pred_labels.shape[0]):
                case_id = case_ids[b]

                pred_arr = pred_labels[b].astype("int16")
                gt_arr = gt_labels[b].astype("int16")

                affine = np.diag([1.0, 1.0, 1.0, 1.0])

                pred_path = pred_dir / f"{case_id}.nii.gz"
                gt_path = output_dir / "ground_truth" / f"{case_id}.nii.gz"
                gt_path.parent.mkdir(parents=True, exist_ok=True)

                nib.save(nib.Nifti1Image(pred_arr, affine=affine), str(pred_path))
                nib.save(nib.Nifti1Image(gt_arr, affine=affine), str(gt_path))


                with tempfile.TemporaryDirectory(dir=output_dir) as tmp_dir:
                    old_cwd = os.getcwd()
                    try:
                        os.chdir(tmp_dir)
                        results_df, lesion_df = get_LesionWiseResults(
                            pred_file=str(pred_path.resolve()),
                            gt_file=str(gt_path.resolve()),
                            challenge_name="BraTS-GLI",
                        )

                    finally:
                        os.chdir(old_cwd)

                case_summary = {
                    "case_id": case_id,
                    "summary": _df_to_records(results_df),
                    "lesions": _df_to_records(lesion_df),
                }
                per_case_results.append(case_summary)

                for _, row in results_df.iterrows():
                    label = row["Labels"]
                    if label not in by_label:
                        continue

                    dice_val = row["LesionWise_Score_Dice"]
                    hd95_val = row["LesionWise_Score_HD95"]

                    if not pd.isna(dice_val):
                        by_label[label]["dice"].append(float(dice_val))
                    if not pd.isna(hd95_val):
                        by_label[label]["hd95"].append(float(hd95_val))

            avg_time = total_time / n_cases if n_cases > 0 else 0.0
            progress_bar.set_postfix({
                "avg_inf_s": f"{avg_time:.2f}"
            })

    by_label_mean = {}
    global_dice = []
    global_hd95 = []

    for label, values in by_label.items():
        mean_dice = _mean(values["dice"])
        mean_hd95 = _mean(values["hd95"])

        by_label_mean[label] = {
            "lesionwise_dice": mean_dice,
            "lesionwise_hd95": mean_hd95,
            "num_cases_with_dice": len(values["dice"]),
            "num_cases_with_hd95": len(values["hd95"]),
        }

        if mean_dice is not None:
            global_dice.append(mean_dice)
        if mean_hd95 is not None:
            global_hd95.append(mean_hd95)

    avg_time = total_time / n_cases if n_cases > 0 else 0.0

    return {
        "mean_lesionwise_dice": _mean(global_dice),
        "mean_lesionwise_hd95": _mean(global_hd95),
        "by_label": by_label_mean,
        "per_case": per_case_results,
        "avg_inference_time_sec": float(avg_time),
        "total_inference_time_sec": float(total_time),
    }
