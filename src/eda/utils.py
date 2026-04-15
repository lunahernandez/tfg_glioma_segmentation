import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import pandas as pd
from tqdm import tqdm
from typing import Optional, Tuple


LABEL_NAMES_ES = {
    0: "Sano / Fondo",
    1: "NETC (Núcleo no realzado)",
    2: "SNFH (Edema/Hiperintensidad)",
    3: "ET (Tejido realzado)",
    4: "RC (Cavidad de resección)"
}


def load_nifti(file_path: str) -> Tuple[np.ndarray, nib.nifti1.Nifti1Header]:
    """
    Carga un archivo NIfTI y extrae sus datos volumétricos y metadatos.

    Args:
        file_path (str): Ruta al archivo .nii o .nii.gz.

    Returns:
        Una tupla (data, header) donde data es una matriz 3D con los datos 
        de intensidad de la imagen y header son los metadatos.

    Raises:
        FileNotFoundError: La ruta especificada no existe en el sistema.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    img = nib.load(file_path)
    return img.get_fdata(), img.header


def save_mri_sequences_plot(
        patient_id: str, 
        patient_dir: str, 
        output_dir: str, 
        slice_idx: Optional[int] = None
) -> None:
    """
    Genera y guarda una figura con 4 subfiguras (2x2) comparando las secuencias 
    principales (T1, T1c, T2 y FLAIR) de una resonancia magnética en formato PDF
    en la ruta especificada.

    Args:
        patient_id (str): Identificador único del paciente.
        patient_dir (str): Ruta al directorio que contiene los archivos NIfTI del paciente.
        output_dir (str): Ruta al directorio donde se exportará el archivo PDF.
        slice_idx (Optional[int], opcional): Índice del corte axial a visualizar. Si es None, 
            se calcula automáticamente el corte con mayor número de vóxeles tumorales. 
            Por defecto es None.
    """
    sequences = ['t1n', 't1c', 't2w', 't2f']
    titles = ['T1 Nativa', 'T1 Contraste', 'T2 Ponderada', 'T2 FLAIR']
    
    if slice_idx is None:
        seg_path = os.path.join(patient_dir, f"{patient_id}-seg.nii.gz")
        seg_data, _ = load_nifti(seg_path)
        tumor_pixels_per_slice = np.sum(seg_data > 0, axis=(0, 1))
        slice_idx = int(np.argmax(tumor_pixels_per_slice))
        print(f"Slice automático para visualización: {slice_idx}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    fig.suptitle(
        f"Paciente: {patient_id} (Corte Axial: {slice_idx})", 
        fontsize=24, 
        fontweight='bold', 
        fontfamily='serif',
        y=0.95
    )
    
    axes_flat = axes.flatten()
    
    for i, seq in enumerate(sequences):
        path = os.path.join(patient_dir, f"{patient_id}-{seq}.nii.gz")
        data, _ = load_nifti(path)
        
        axes_flat[i].imshow(data[:, :, slice_idx].T, cmap='gray', origin='lower')

        axes_flat[i].set_title(
            f"{titles[i]}", 
            fontsize=22,            
            fontweight='bold',      
            fontstyle='normal',     
            color='black',          
            fontfamily='serif'      
        )
        
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.90) 
    
    file_name = f"{patient_id}_mri_sequences.pdf"
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()


def save_mri_segmentation_plot(
        patient_id: str, 
        patient_dir: str, 
        output_dir: str, 
        slice_idx: Optional[int] = None
) -> None:
    """
    Genera y guarda una figura de un corte axial de la secuencia T2 FLAIR y 
    la máscara de segmentación superpuesta en formato PDF en la ruta especificada.

    Args:
        patient_id (str): Identificador único del paciente.
        patient_dir (str): Ruta al directorio que contiene los archivos NIfTI del paciente.
        output_dir (str): Ruta al directorio donde se exportará el archivo PDF.
        slice_idx (Optional[int], opcional): Índice del corte axial a visualizar. Si es None, 
            se calcula automáticamente el corte con mayor número de vóxeles tumorales. 
            Por defecto es None.
    """
    color_map = {
        1: '#FF5C5C', # NETC
        2: '#66F466', # SNFH
        3: '#7070DC', # ET
        4: '#FFFF5C'  # RC
    }
    
    custom_cmap = ListedColormap(list(color_map.values()))

    seg_path = os.path.join(patient_dir, f"{patient_id}-seg.nii.gz")
    seg_data, _ = load_nifti(seg_path)
    
    if slice_idx is None:
        tumor_pixels_per_slice = np.sum(seg_data > 0, axis=(0, 1))
        slice_idx = int(np.argmax(tumor_pixels_per_slice))

    t2f_path = os.path.join(patient_dir, f"{patient_id}-t2f.nii.gz")
    t2f_data, _ = load_nifti(t2f_path)
    
    flair_slice = t2f_data[:, :, slice_idx].T
    mask_slice = seg_data[:, :, slice_idx].T
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    fig.suptitle(
        f"Paciente: {patient_id} (Corte Axial: {slice_idx})", 
        fontsize=22, fontweight='bold', fontfamily='serif', y=0.98
    )
    
    ax.imshow(flair_slice, cmap='gray', origin='lower')
    
    masked_seg = np.ma.masked_where(mask_slice == 0, mask_slice)
    ax.imshow(
        masked_seg, cmap=custom_cmap, alpha=0.5, 
        origin='lower', interpolation='none', vmin=1, vmax=4
    )
    
    ax.set_title("(FLAIR + Máscara)", fontsize=18, fontweight='bold', fontfamily='serif')
    ax.axis('off')
    
    legend_patches = []
    for i, color in color_map.items():
        name = LABEL_NAMES_ES.get(i, f"Etiqueta {i}")
        patch = mpatches.Patch(color=color, label=name, alpha=0.7)
        legend_patches.append(patch)
        
    leg = ax.legend(
        handles=legend_patches, title="Leyenda de Segmentación", 
        loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2,
        prop={'family': 'serif', 'size': 15}
    )
    
    plt.setp(leg.get_title(), family='serif', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.2)
    
    file_name = f"{patient_id}_mri_segmentation.pdf"
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    plt.close()


def check_image_dimensions(patient_id: str, patient_dir: str) -> None:
    """
    Verifica que las dimensiones de las secuencias NIfTI de un paciente
    coincidan. Muestra por consola el resultado.

    Args:
        patient_id (str): Identificador único del paciente.
        patient_dir (str): Ruta al directorio que contiene los archivos NIfTI del paciente.
    """
    mri_sequences_files = {
        'T1 Nativa': f"{patient_id}-t1n.nii.gz",
        'T1 Contraste': f"{patient_id}-t1c.nii.gz",
        'T2 Ponderada': f"{patient_id}-t2w.nii.gz",
        'T2 FLAIR': f"{patient_id}-t2f.nii.gz",
        'Segmentación': f"{patient_id}-seg.nii.gz"
    }
    
    print(f"\nVerificando dimensiones para el paciente: {patient_id}")
    print("-" * 75)
    print(f"{'Secuencia':<15} | {'Dimensiones (Vóxeles)':<25} | {'Resolución Física (mm)':<25}")
    print("-" * 75)
    
    reference_shape = None
    all_match = True
    
    for sequence_name, file_name in mri_sequences_files.items():
        file_path = os.path.join(patient_dir, file_name)
        try:
            img = nib.load(file_path)
            shape = img.shape
            physical_resolution = img.header.get_zooms() 
            formatted_resolution = tuple(round(z, 4) for z in physical_resolution)
            print(f"{sequence_name:<15} | {str(shape):<25} | {str(formatted_resolution):<25}")
            
            if reference_shape is None:
                reference_shape = shape
            elif shape != reference_shape:
                all_match = False
                print(f"[ADVERTENCIA] Las dimensiones de {sequence_name} no coincide con la referencia {reference_shape}")
                
        except FileNotFoundError:
            print(f"{sequence_name:<15} | {'[Archivo no encontrado]':<25} | {'-':<25}")
            all_match = False
            
    print("-" * 75)
    if all_match:
        print("ESTADO: OK. Todas las secuencias tienen las mismas dimensiones.")
    else:
        print("ESTADO: ADVERTENCIA. Hay diferencias en las dimensiones.")
    print("-" * 75)


def verify_dataset_dimensions(
        dataset_dir: str, 
        sequences: list[str] = ['t1n', 't1c', 't2w', 't2f', 'seg']
) -> None:
    """
    Recorre todas las carpetas de pacientes en el dataset y verifica que 
    todos los archivos NIfTI compartan las mismas dimensiones y resolución.

    Args:
        dataset_dir (str): Ruta al directorio raíz que contiene las carpetas de pacientes.
        sequences (list[str], optional): Secuencias a verificar.
    """
    patient_directories = [f for f in os.listdir(dataset_dir) 
                           if os.path.isdir(os.path.join(dataset_dir, f))]
    
    total_patients = len(patient_directories)
    print(f"\nIniciando verificación de dimensiones de {total_patients} pacientes...")
    print("-" * 75)
    
    expected_shape = None
    expected_physical_resolution = None
    
    dimension_anomalies = []
    missing_files = []

    for i, patient_id in enumerate(patient_directories, 1):
        patient_path = os.path.join(dataset_dir, patient_id)
        
        if i % 50 == 0 or i == total_patients:
            print(f"Procesando: {i}/{total_patients} pacientes...")
            
        for seq in sequences:
            sequence_filename = f"{patient_id}-{seq}.nii.gz"
            file_path = os.path.join(patient_path, sequence_filename)
            
            if not os.path.exists(file_path):
                missing_files.append((patient_id, sequence_filename))
                continue
                
            try:
                nifti_image = nib.load(file_path)
                shape = nifti_image.shape
                physical_resolution = tuple(round(z, 4) for z in nifti_image.header.get_zooms())
                
                if expected_shape is None:
                    expected_shape = shape
                    expected_physical_resolution = physical_resolution
                elif shape != expected_shape or physical_resolution != expected_physical_resolution:
                    dimension_anomalies.append({
                        'patient': patient_id, 
                        'file': sequence_filename,
                        'shape': shape, 
                        'physical_resolution': physical_resolution
                    })
            except Exception as e:
                dimension_anomalies.append({'patient': patient_id, 'file': sequence_filename, 'error': str(e)})

    print("\n" + "=" * 75)
    print("VERIFICACIÓN DEL DATASET")
    print("=" * 75)
    
    if not dimension_anomalies and not missing_files:
        print("RESULTADO: OK.")
        print(f"Todos los archivos comparten las dimensiones {expected_shape} y resolución {expected_physical_resolution} mm.")
    else:
        if missing_files:
            print(f"[ADVERTENCIA] Faltan {len(missing_files)} archivos.")
        if dimension_anomalies:
            print(f"[ADVERTENCIA] Se encontraron {len(dimension_anomalies)} archivos con dimensiones distintas o corruptos.")
            for anomaly in dimension_anomalies[:10]:
                print(f"  - {anomaly['file']} -> Shape: {anomaly.get('shape', 'N/A')}")


def extract_voxel_statistics(
    dataset_directories: list[str], 
    csv_path: str
) -> pd.DataFrame:
    """
    Extrae el conteo de vóxeles de las máscaras de segmentación y lo guarda en un CSV.
    Si el CSV ya existe, carga los datos directamente desde el archivo.

    Args:
        dataset_directories (list[str]): Lista de rutas a los directorios de datasets.
        csv_path (str): Ruta completa donde se guardará o leerá el archivo CSV.

    Returns:
        Un dataframe con las estadísticas de vóxeles obtenidas.
    """
    if os.path.exists(csv_path):
        print(f"\nCargando estadísticas desde {csv_path}...")
        return pd.read_csv(csv_path)

    print("\nExtrayendo datos de segmentación...")
    if isinstance(dataset_directories, str):
        dataset_directories = [dataset_directories]

    results = []

    for dataset_dir in dataset_directories:
        print(f"\nAnalizando directorio: {dataset_dir}")
        patient_directories = [f for f in os.listdir(dataset_dir) 
                               if os.path.isdir(os.path.join(dataset_dir, f))]
        
        for patient_id in tqdm(patient_directories, desc=f"Procesando {os.path.basename(dataset_dir)}"):
            seg_file = os.path.join(dataset_dir, patient_id, f"{patient_id}-seg.nii.gz")
            
            if not os.path.exists(seg_file):
                continue
                
            img = nib.load(seg_file)
            data = img.get_fdata()
            unique, counts = np.unique(data, return_counts=True)
            counts_dict = dict(zip(unique, counts))
            
            results.append({
                'Patient_ID': patient_id,
                'Source': os.path.basename(dataset_dir),
                'Background': counts_dict.get(0, 0),
                'NETC': counts_dict.get(1, 0),
                'SNFH': counts_dict.get(2, 0),
                'ET': counts_dict.get(3, 0),
                'RC': counts_dict.get(4, 0),
                'Total_Voxels': data.size
            })
            
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    return df


def show_class_balance(df: pd.DataFrame) -> None:
    """
    Calcula e imprime las estadísticas de desbalance de clases por consola.

    Args:
        df (pd.DataFrame): DataFrame con los conteos de vóxeles por paciente.
    """
    total_voxels = df['Total_Voxels'].sum()
    
    if total_voxels == 0:
        print("\n[ADVERTENCIA]: Dataset sin vóxeles.")
        return

    bg_pct = (df['Background'].sum() / total_voxels) * 100
    
    tumor_classes = ['NETC', 'SNFH', 'ET', 'RC']
    tumor_counts = {cls: df[cls].sum() for cls in tumor_classes}
    total_tumor = sum(tumor_counts.values())
    
    print("\n" + "=" * 75)
    print("Desbalance del dataset")
    print("=" * 75)
    print(f"Pacientes analizados : {len(df)}")
    print(f"Vóxeles totales      : {total_voxels:,}")
    print(f"Fondo (Background)   : {bg_pct:.4f}%\n")
    
    if total_tumor > 0:
        print("Clases del tumor (Excluyendo fondo):")
        for cls_name, count in tumor_counts.items():
            pct = (count / total_tumor) * 100
            print(f"    - {cls_name:<4} : {pct:>5.2f}%  ({count:>12,} vóxeles)")
    else:
        print("[ADVERTENCIA] No se detectaron vóxeles de tumor.")
        
    print("=" * 75 + "\n")


def save_eda_figures(df: pd.DataFrame, output_dir: str) -> None:
    """
    Genera y guarda en formato PDF las gráficas globales de frecuencia y volumen
    obtenidas a partir de un dataframe en el directorio especificado.

    Args:
        df (pd.DataFrame): DataFrame con los conteos de vóxeles por paciente.
        output_dir (str): Directorio donde se guardarán los archivos PDF.
    """
    os.makedirs(output_dir, exist_ok=True)

    colors = ['#FF5C5C', '#66F466', '#7070DC', '#FFFF5C']
    labels = ['NETC', 'SNFH', 'ET', 'RC']

    presence = {l: (df[l] > 0).mean() * 100 for l in labels}
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    bars1 = ax1.bar(labels, list(presence.values()), color=colors, edgecolor='black')
    ax1.set_title("Frecuencia de subregiones", fontsize=22, fontweight='bold', fontfamily='serif')
    ax1.set_ylabel("Porcentaje de pacientes (%)", fontsize=18, fontfamily='serif')
    ax1.set_ylim(0, 115)
    ax1.tick_params(axis='both', labelsize=16)

    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontfamily('serif')

    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval:.1f}%', 
                 ha='center', va='bottom', fontsize=18, fontfamily='serif')
                 
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "frecuencia_global.pdf"))
    plt.close(fig1)

    totals = [df[l].sum() for l in labels]
    total_tumor_voxels = sum(totals)

    if total_tumor_voxels > 0:
        vol_pct = [(t / total_tumor_voxels) * 100 for t in totals]
    else:
        vol_pct = [0 for _ in labels]
    
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    bars2 = ax2.bar(labels, vol_pct, color=colors, edgecolor='black')
    
    ax2.set_title("Distribución de las subregiones", fontsize=22, fontweight='bold', fontfamily='serif')
    ax2.set_ylabel("Porcentaje del volumen segmentado (%)", fontsize=18, fontfamily='serif')
    ax2.set_ylim(0, max(vol_pct) + 12) 
    ax2.tick_params(axis='both', labelsize=16)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontfamily('serif')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.8, f'{yval:.1f}%', 
                 ha='center', va='bottom', fontsize=18, fontfamily='serif')
                 
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "volumen_global.pdf"))
    plt.close(fig2)
