import os
import argparse
from utils import (
    save_mri_sequences_plot,
    save_mri_segmentation_plot,
    check_image_dimensions,
    verify_dataset_dimensions,
    extract_voxel_statistics,
    show_class_balance,
    save_eda_figures
)


def print_section_header(title: str) -> None:
    """Prints a clean console header to separate sections."""
    print(f"\n{'=' * 75}")
    print(title.upper())
    print(f"{'=' * 75}")

def main():
    parser = argparse.ArgumentParser(description="EDA pipeline for BraTS 2024")
    parser.add_argument(
        "--patient", 
        type=str, 
        default="BraTS-GLI-03063-100", 
        help="Patient ID for individual analysis"
    )
    args = parser.parse_args()

    PATIENT_ID = args.patient
    DATA_BASE_DIR = "data"
    RESULTS_DIR = "results"
    TRAIN = os.path.join(DATA_BASE_DIR, "training_data1_v2")
    TRAIN_ADD = os.path.join(DATA_BASE_DIR, "training_data_additional")
    VAL_DATA = os.path.join(DATA_BASE_DIR, "validation_data")
    if os.path.exists(os.path.join(TRAIN, PATIENT_ID)):
        PATIENT_PATH = os.path.join(TRAIN, PATIENT_ID)
    elif os.path.exists(os.path.join(TRAIN_ADD, PATIENT_ID)):
        PATIENT_PATH = os.path.join(TRAIN_ADD, PATIENT_ID)
    else:
        print(f"\n[ERROR] Patient {PATIENT_ID} not found in training directories.")
        return
    PATIENT_FIGURES_DIR = os.path.join("figures", "patients", PATIENT_ID)
    GLOBAL_FIGURES_DIR = os.path.join("figures", "global_eda")
    GLOBAL_CSV_PATH = os.path.join(RESULTS_DIR, "dataset_global_stats.csv")

    os.makedirs(PATIENT_FIGURES_DIR, exist_ok=True)
    os.makedirs(GLOBAL_FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print_section_header("Dataset dimension verification")
    
    print(f"Main dataset: {TRAIN}")
    verify_dataset_dimensions(TRAIN)
    
    print(f"\nAdditional dataset: {TRAIN_ADD}")
    verify_dataset_dimensions(TRAIN_ADD)
    
    print(f"\nValidation dataset: {VAL_DATA}")
    verify_dataset_dimensions(VAL_DATA, sequences=['t1n', 't1c', 't2w', 't2f'])

    print_section_header(f"Individual EDA: Patient {PATIENT_ID}")
    check_image_dimensions(PATIENT_ID, PATIENT_PATH)
    
    print("\nGenerating MRI sequence comparison...")
    save_mri_sequences_plot(PATIENT_ID, PATIENT_PATH, PATIENT_FIGURES_DIR)
    
    print("\nGenerating segmented visualization...")
    save_mri_segmentation_plot(PATIENT_ID, PATIENT_PATH, PATIENT_FIGURES_DIR)

    print(f"\nIndividual results saved in: {PATIENT_FIGURES_DIR}")

    print_section_header("Global Exploratory Analysis")
    
    train_dataset_paths = [TRAIN, TRAIN_ADD]
    
    df_stats = extract_voxel_statistics(
        dataset_directories=train_dataset_paths, 
        csv_path=GLOBAL_CSV_PATH
    )
    
    show_class_balance(df=df_stats)
    
    print(f"Generating PDF figures in directory: {GLOBAL_FIGURES_DIR}...")
    save_eda_figures(
        df=df_stats, 
        output_dir=GLOBAL_FIGURES_DIR
    )

    print_section_header("Pipeline completed successfully")

if __name__ == "__main__":
    main()
