"""
Data loading and preprocessing pipeline
Handles image augmentation, batching, and class weight calculation
"""
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from config import (
    BATCH_SIZE,
    IMG_SIZE,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    VALIDATION_SPLIT,
    validate_dataset_layout,
)


def create_data_generators():
    """
    Create data generators for training, validation, and testing
    Training data includes augmentation, test data does not
    """
    validate_dataset_layout()

    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        fill_mode="nearest",
        validation_split=VALIDATION_SPLIT,
    )
    
    # Test data only gets rescaled, no augmentation
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    
    # 80% of train folder for actual training
    train_gen = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        seed=SEED,
        shuffle=True,
        subset="training"
    )
    
    # 20% of train folder for validation during training
    val_gen = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        seed=SEED,
        shuffle=False,
        subset="validation"
    )
    
    # Completely separate test set
    test_gen = test_datagen.flow_from_directory(
        str(TEST_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        seed=SEED,
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen


def compute_class_weights(train_gen):
    """
    Compute class weights to handle imbalance
    Return balanced class weights based on the generated training split.
    """
    labels = train_gen.classes
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weights = dict(enumerate(class_weights_array))
    
    return class_weights


def print_dataset_info(train_gen, val_gen, test_gen):
    """Print dataset statistics"""
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Training images  : {train_gen.n:,}")
    print(f"Validation images: {val_gen.n:,}")
    print(f"Test images      : {test_gen.n:,}")
    print(f"Class mapping    : {train_gen.class_indices}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test the pipeline
    train_gen, val_gen, test_gen = create_data_generators()
    class_weights = compute_class_weights(train_gen)
    print_dataset_info(train_gen, val_gen, test_gen)
    print(f"Class weights: {class_weights}")
