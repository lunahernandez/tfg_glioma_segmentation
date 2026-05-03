from pathlib import Path

import torch

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CACHE_ROOT = PROJECT_ROOT / "persistent_cache"

TRAIN_DIRS = [
    DATA_DIR / "training_data1_v2",
    DATA_DIR / "training_data_additional",
]


# Hardware and reproducibility
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Dataset configuration
LABEL_SUFFIX = "seg"

# ROI_SIZE = (96, 96, 96)
ROI_SIZE = (128, 128, 128)
SPACING = (1.0, 1.0, 1.0)

IN_CHANNELS = 4
OUT_CHANNELS = 5

# Experiments
MODEL_NAME = "unet3d"
# MODEL_NAME = "resunet3d"
# MODEL_NAME = "dense_unet_plus"
# MODEL_NAME = "swin_unetr"
# MODEL_NAME = "segmamba"

# Training hyperparameters
BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
NUM_WORKERS = 4

MAX_EPOCHS = 100
VAL_EVERY = 5

WEIGHT_DECAY = 1e-5

if MODEL_NAME == "segmamba":
    LEARNING_RATE = 1e-5
    CLIP_GRAD = True
    GRAD_CLIP_MAX_NORM = 1.0
else:
    LEARNING_RATE = 1e-4
    CLIP_GRAD = False
    GRAD_CLIP_MAX_NORM = None

# Memory and performance
SW_BATCH_SIZE = 2
USE_CHECKPOINT = True

# Cross-validation
N_FOLDS = 5
INNER_VAL_RATIO = 0.1

# Cache
CACHE_VERSION = "v1"
CACHE_NAME = (
    f"brats_{CACHE_VERSION}"
    f"_roi{ROI_SIZE[0]}x{ROI_SIZE[1]}x{ROI_SIZE[2]}"
    f"_sp{SPACING[0]}_{SPACING[1]}_{SPACING[2]}"
)
PERSISTENT_CACHE_DIR = CACHE_ROOT / CACHE_NAME

EXPERIMENT_NAME = (
    f"brats_{MODEL_NAME}_roi{ROI_SIZE[0]}_bs{BATCH_SIZE}"
    f"_nworkers{NUM_WORKERS}_cv{N_FOLDS}_labelmapfix"
)