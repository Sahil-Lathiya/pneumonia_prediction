"""
Model architectures for pneumonia detection
Includes ResNet50, MobileNetV2, and Custom CNN
"""
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, Dense, 
    Dropout, GlobalAveragePooling2D, Input
)


def build_resnet50():
    """
    Build ResNet50 transfer learning model
    - Base: Pretrained on ImageNet (24M+ parameters)
    - Head: Custom classification layers
    """
    resnet_base = ResNet50(weights="imagenet", include_top=False, 
                          input_shape=(224, 224, 3))
    resnet_base.trainable = False  # Freeze base in phase 1
    
    x = resnet_base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=resnet_base.input, outputs=output)
    return model


def build_mobilenetv2():
    """
    Build MobileNetV2 transfer learning model
    - Base: Pretrained on ImageNet (2.4M parameters, lightweight)
    - Head: Custom classification layers
    """
    mobilenet_base = MobileNetV2(weights="imagenet", include_top=False,
                                input_shape=(224, 224, 3))
    mobilenet_base.trainable = False  # Freeze base in phase 1
    
    x = mobilenet_base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=mobilenet_base.input, outputs=output)
    return model


def build_custom_cnn():
    """
    Build custom CNN from scratch
    - 4 convolutional blocks
    - No pretrained weights
    - For comparison with transfer learning models
    """
    model = Sequential([
        Input(shape=(224, 224, 3)),
        
        # Block 1
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        
        # Block 4
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        
        # Classification head
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    
    return model
