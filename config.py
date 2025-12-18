# Configuration file for AMD Detection Model

import os

# Dataset paths
DATASET_ROOT = r"D:\AMDNet23 Fundus Image Dataset for  Age-Related Macular Degeneration Disease Detection\AMDNet23 Fundus Image Dataset for  Age-Related Macular Degeneration Disease Detection\AMDNet23 Dataset"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VALID_DIR = os.path.join(DATASET_ROOT, "valid")

# Model configuration
IMG_SIZE = 224  # Standard size for most pretrained models
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
NUM_CLASSES = 4  # amd, cataract, diabetes, normal

# Class names
CLASS_NAMES = ['amd', 'cataract', 'diabetes', 'normal']

# Output directories
OUTPUT_DIR = "outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Create output directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Training parameters
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
