# Análisis comparativo de arquitecturas de aprendizaje profundo para la segmentación de gliomas en imágenes de resonancia magnética

Repositorio de código desarrollado para el Trabajo de Fin de Grado **“Análisis comparativo de arquitecturas de aprendizaje profundo para la segmentación de gliomas en imágenes de resonancia magnética”**.

El objetivo del proyecto es comparar diferentes modelos de aprendizaje profundo para la segmentación automática de gliomas en imágenes de resonancia magnética postratamiento pertenecientes al conjunto de datos BraTS 2024. La comparación se realiza teniendo en cuenta métricas de rendimiento, como el coeficiente Dice y la distancia de Hausdorff al percentil 95, así como métricas computacionales, como el tiempo de entrenamiento, el tiempo de inferencia y la memoria máxima utilizada.

Este repositorio incluye el código necesario para la carga de datos, preprocesamiento, entrenamiento, validación, evaluación, cálculo de métricas, análisis de validación cruzada y visualización de resultados.

## Descripción del proyecto

El flujo de trabajo incluye:

* análisis exploratorio de volúmenes de resonancia magnética y máscaras de segmentación;
* preparación y preprocesamiento de los datos;
* entrenamiento, validación y evaluación de modelos 3D de segmentación;
* cálculo de métricas de rendimiento y coste computacional;
* agregación de resultados de validación cruzada;
* análisis y visualización de los resultados.

## Estructura del repositorio

```text
tfg_glioma_segmentation/
├── src/                         # Código fuente principal del proyecto
│   ├── config.py                # Configuración global de rutas, modelos y experimentos
│   ├── main.py                  # Ejecución principal del pipeline experimental
│   ├── train.py                 # Entrenamiento de modelos
│   ├── validate.py              # Validación durante el entrenamiento
│   ├── evaluate.py              # Evaluación de modelos entrenados
│   ├── data/                    # Carga, particionado y transformación de datos
│   │   ├── dataset.py
│   │   ├── splits.py
│   │   └── transforms.py
│   ├── eda/                     # Análisis exploratorio del dataset
│   │   ├── main_eda.py
│   │   ├── metadata_eda.py
│   │   └── utils.py
│   ├── models/                  # Definición de arquitecturas de segmentación
│   │   ├── get_model.py
│   │   ├── resunet3d.py
│   │   ├── segmamba.py
│   │   ├── swin_unetr.py
│   │   └── unet3d.py
│   └── utils/                   # Utilidades comunes del pipeline
│       ├── brats_lesionwise.py
│       ├── brats_regions.py
│       ├── checkpoints.py
│       ├── meters.py
│       └── seed.py
├── scripts/                     # Scripts auxiliares de análisis y visualización
│   ├── analyze_crossval_statistics.py
│   ├── compare_background_results.py
│   ├── create_3d_models_region_grid.py
│   ├── generate_crossval_results.py
│   ├── plot_crossval_results.py
│   └── utils_results.py
├── docs/                        # Documentación adicional
│   └── segmamba_notes.md        # Notas de instalación y compatibilidad de SegMamba
├── requirements.txt             # Dependencias de Python
├── LICENSE                      # Licencia del proyecto
├── NOTICE                       # Atribuciones de código de terceros
└── README.md                    # Documentación principal del repositorio
```

## Dataset

El conjunto de datos no se incluye en este repositorio debido a restricciones de tamaño y licencia.

Los datos utilizados corresponden al reto **BraTS 2024 Challenge - BraTS-GLI** y pueden obtenerse desde la [página oficial de Synapse](https://www.synapse.org/Synapse:syn53708249/wiki/627759).

Una vez descargados, se deben ubicar en el directorio `data/` con una estructura similar a la siguiente:

```text
data/
├── training_data1_v2/
├── training_data_additional/
├── validation_data/
├── BraTS-PTG supplementary demographic information and metadata.xlsx
├── CITATIONS.bib
└── manifest.csv
```

Las rutas exactas del dataset y la configuración de los experimentos se definen en:

```text
src/config.py
```

Antes de ejecutar el proyecto, es necesario comprobar que las rutas indicadas en `src/config.py` coinciden con la estructura local de carpetas.

## Entorno de ejecución

El proyecto se desarrolló y ejecutó en un entorno **WSL con Ubuntu**, utilizando una GPU NVIDIA RTX 5090, CUDA 12.8 y Python 3.12.

El entorno principal utilizado fue:

```text
OS: WSL Ubuntu
Python: 3.12
GPU: NVIDIA RTX 5090
CUDA Toolkit: 12.8
PyTorch: 2.11.0+cu128
```

La instalación general del proyecto se describe en la siguiente sección. Para reproducir todos los experimentos, incluyendo los basados en SegMamba, se deben seguir además las instrucciones específicas de configuración y compilación local descritas en:

```text
docs/segmamba_notes.md
```

## Instalación

Crear y activar un entorno virtual de Python:

```bash
python3 -m venv venv
source venv/bin/activate
```

Actualizar las herramientas básicas de empaquetado:

```bash
python -m pip install --upgrade pip wheel ninja
python -m pip install "setuptools==70.2.0"
```

Instalar las dependencias generales del proyecto:

```bash
pip install -r requirements.txt
```

El archivo `requirements.txt` utiliza la versión de PyTorch con soporte para CUDA 12.8.

> **Nota:** la instalación anterior cubre las dependencias generales del proyecto. Para ejecutar los experimentos basados en SegMamba, es necesario seguir los pasos adicionales descritos en `docs/segmamba_notes.md`, ya que `causal-conv1d` y `mamba-ssm` tuvieron que compilarse localmente para ser compatibles con la NVIDIA RTX 5090.

## Configuración

El archivo principal de configuración es:

```text
src/config.py
```

Este archivo contiene las rutas del proyecto, parámetros de entrenamiento, configuración de modelos y opciones de evaluación utilizadas en el pipeline.

Antes de lanzar un experimento, se recomienda revisar especialmente:

* rutas del dataset;
* carpetas de salida;
* selección del modelo;
* hiperparámetros de entrenamiento;
* configuración de validación cruzada;
* opciones de evaluación.

## Uso

Ejecutar el pipeline completo desde la raíz del proyecto:

```bash
python -m src.main
```

Por defecto, el modo de ejecución es `all`, por lo que se realiza tanto el entrenamiento como la evaluación.

Entrenar y evaluar un fold concreto:

```bash
python -m src.main --fold 1
```

Ejecutar solo el entrenamiento:

```bash
python -m src.main --mode train
```

Ejecutar solo el entrenamiento de un fold concreto:

```bash
python -m src.main --mode train --fold 1
```

Ejecutar solo la evaluación:

```bash
python -m src.main --mode test
```

Ejecutar solo la evaluación de un fold concreto:

```bash
python -m src.main --mode test --fold 1
```

Ejecutar experimentos incluyendo o excluyendo la clase de fondo en la función de pérdida:

```bash
python -m src.main --include-background
python -m src.main --no-include-background
```

Estas opciones también pueden combinarse. Por ejemplo:

```bash
python -m src.main --mode train --fold 1 --include-background
python -m src.main --mode test --fold 1 --no-include-background
```

## Scripts de análisis de resultados

La carpeta `scripts/` contiene utilidades para procesar, agregar y visualizar los resultados experimentales.

Ejemplos:

```bash
python scripts/generate_crossval_results.py
python scripts/analyze_crossval_statistics.py
python scripts/compare_background_results.py
python scripts/plot_crossval_results.py
python scripts/create_3d_models_region_grid.py
```

## Código de terceros y atribución

Este proyecto incluye o adapta código de proyectos externos de código abierto, incluyendo SegMamba y utilidades de métricas de BraTS 2024.

Consultar el archivo `NOTICE` para más detalles sobre las atribuciones.

## Licencia

Este proyecto está licenciado bajo la Apache License 2.0. Consultar el archivo `LICENSE` para más detalles.