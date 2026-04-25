"""
Training utilities and callbacks for model training
"""
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)


def compile_model(model, learning_rate=1e-4):
    """Compile model with Adam optimizer"""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"]
    )
    return model


def get_callbacks(model_path, patience=5):
    """
    Create training callbacks:
    - EarlyStopping: Stop if validation loss doesn't improve
    - ReduceLROnPlateau: Lower learning rate when stuck
    - ModelCheckpoint: Save best model version
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        )
    ]
    return callbacks


def train_model(model, train_gen, val_gen, epochs, class_weights, model_path):
    """
    Train a model
    
    Args:
        model: Keras model
        train_gen: Training data generator
        val_gen: Validation data generator
        epochs: Number of epochs
        class_weights: Dictionary of class weights
        model_path: Path to save best model
    
    Returns:
        Training history
    """
    callbacks = get_callbacks(model_path)
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    return history
