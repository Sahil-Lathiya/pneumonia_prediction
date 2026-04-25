"""
Configuration and paths for Pneumonia Detection Project
"""
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "chest_xray")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
CHARTS_DIR = os.path.join(BASE_DIR, "report_charts")

# Dataset paths
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Model save paths
RESNET_PATH = os.path.join(MODELS_DIR, "best_resnet50.keras")
MOBILENET_PATH = os.path.join(MODELS_DIR, "best_mobilenetv2.keras")
CNN_PATH = os.path.join(MODELS_DIR, "best_custom_cnn.keras")

# Training parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

print("Configuration loaded successfully")
print(f"Dataset: {DATASET_DIR}")
print(f"Models: {MODELS_DIR}")
print(f"Charts: {CHARTS_DIR}")
