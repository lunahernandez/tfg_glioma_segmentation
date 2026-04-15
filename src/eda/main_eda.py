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
    """Imprime un encabezado limpio en la consola para separar secciones."""
    print(f"\n{'=' * 75}")
    print(title.upper())
    print(f"{'=' * 75}")

def main():
    parser = argparse.ArgumentParser(description="Pipeline de EDA para BraTS 2024")
    parser.add_argument(
        "--patient", 
        type=str, 
        default="BraTS-GLI-03063-100", 
        help="ID del paciente para el análisis individual"
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
        print(f"\n[ERROR] No se encontró al paciente {PATIENT_ID} en los directorios de entrenamiento.")
        return
    PATIENT_FIGURES_DIR = os.path.join("figures", "patients", PATIENT_ID)
    GLOBAL_FIGURES_DIR = os.path.join("figures", "global_eda")
    GLOBAL_CSV_PATH = os.path.join(RESULTS_DIR, "dataset_global_stats.csv")

    os.makedirs(PATIENT_FIGURES_DIR, exist_ok=True)
    os.makedirs(GLOBAL_FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print_section_header("Verificación de dimensiones de los datasets")
    
    print(f"Dataset principal: {TRAIN}")
    verify_dataset_dimensions(TRAIN)
    
    print(f"\nDataset adicional: {TRAIN_ADD}")
    verify_dataset_dimensions(TRAIN_ADD)
    
    print(f"\nDataset de validación: {VAL_DATA}")
    verify_dataset_dimensions(VAL_DATA, sequences=['t1n', 't1c', 't2w', 't2f'])

    print_section_header(f"EDA Individual: Paciente {PATIENT_ID}")
    check_image_dimensions(PATIENT_ID, PATIENT_PATH)
    
    print("\nGenerando comparación de secuencias de RM...")
    save_mri_sequences_plot(PATIENT_ID, PATIENT_PATH, PATIENT_FIGURES_DIR)
    
    print("\nGenerando visualización segmentada...")
    save_mri_segmentation_plot(PATIENT_ID, PATIENT_PATH, PATIENT_FIGURES_DIR)

    print(f"\nResultados individuales guardados en: {PATIENT_FIGURES_DIR}")

    print_section_header("Análisis Exploratorio Global")
    
    train_dataset_paths = [TRAIN, TRAIN_ADD]
    
    df_stats = extract_voxel_statistics(
        dataset_directories=train_dataset_paths, 
        csv_path=GLOBAL_CSV_PATH
    )
    
    show_class_balance(df=df_stats)
    
    print(f"Generando gráficas en PDF en el directorio: {GLOBAL_FIGURES_DIR}...")
    save_eda_figures(
        df=df_stats, 
        output_dir=GLOBAL_FIGURES_DIR
    )

    print_section_header("Pipeline finalizado con éxito")

if __name__ == "__main__":
    main()