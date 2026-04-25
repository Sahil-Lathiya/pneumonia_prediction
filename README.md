# 🫁 Pneumonia Detection using Chest X-Ray Images

A deep learning project that classifies chest X-ray images as **Normal** or **Pneumonia** using state-of-the-art neural networks.

---

## 📋 Quick Overview

This project compares three deep learning models:
- **ResNet50** - Large transfer learning model (24M parameters)
- **MobileNetV2** - Lightweight transfer learning model (2.4M parameters) ⭐ Best performance
- **Custom CNN** - Built from scratch for comparison

All models can detect pneumonia from chest X-ray images with high accuracy.

---

## 🎯 What You Can Do

✅ Train multiple models from scratch  
✅ Evaluate models on test data  
✅ Generate performance charts and reports  
✅ Test on new/unseen images  
✅ Compare model architectures  
✅ Use the best model for inference  

---

## 📁 Project Structure

```
pneumonia_prediction/
├── config.py                 # Configuration and paths
├── data_pipeline.py         # Data loading and preprocessing
├── models.py                # Model architectures
├── train_utils.py           # Training utilities
├── evaluate.py              # Evaluation metrics
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── dataset/                # YOUR DATASET (not included)
│   └── chest_xray/
│       ├── train/
│       ├── val/
│       └── test/
│
├── saved_models/           # Trained models (after training)
│   ├── best_resnet50.keras
│   ├── best_mobilenetv2.keras
│   └── best_custom_cnn.keras
│
└── report_charts/          # Performance graphs (after training)
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── metric_comparison.png
    └── training_history.png
```

---

## 🚀 Getting Started (Simple 3 Steps)

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Dataset

Download the chest X-ray dataset and organize it:

```
dataset/chest_xray/
├── train/
│   ├── NORMAL/        (1,341 images)
│   └── PNEUMONIA/     (3,875 images)
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

**Dataset source:** [Kaggle Chest X-Ray Images Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Step 3: Run Training (Use Jupyter Notebook)

The original `Pnuemonia_detection_project.ipynb` notebook contains all training code. Open it in Jupyter:

```bash
jupyter notebook Pnuemonia_detection_project.ipynb
```

Or run Python scripts directly:

```bash
python -c "from data_pipeline import *; train_gen, val_gen, test_gen = create_data_generators()"
```

---

## 🧠 Models Explained

### ResNet50
- **Type:** Transfer Learning
- **Parameters:** 24M+
- **Speed:** Slower, more memory
- **Accuracy:** ~95%
- **Best for:** Maximum accuracy requirements

### MobileNetV2 ⭐ RECOMMENDED
- **Type:** Transfer Learning
- **Parameters:** 2.4M (10x smaller!)
- **Speed:** Fast, low memory
- **Accuracy:** ~96%
- **Best for:** Production, mobile deployment

### Custom CNN
- **Type:** Built from scratch
- **Parameters:** ~1M
- **Speed:** Fastest
- **Accuracy:** ~92%
- **Best for:** Learning how CNNs work

---

## 📊 Expected Results

After training, you'll get:

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| ResNet50 | ~95% | ~95% | ~96% |
| MobileNetV2 | ~96% | ~96% | ~97% |
| Custom CNN | ~92% | ~92% | ~94% |

Plus performance charts:
- Confusion matrices
- ROC curves
- Training history
- Metric comparison

---

## 💻 Code Structure

### `config.py` - Setup paths and parameters
```python
from config import TRAIN_DIR, MODELS_DIR, IMG_SIZE
```

### `data_pipeline.py` - Load and prepare data
```python
from data_pipeline import create_data_generators, compute_class_weights
train_gen, val_gen, test_gen = create_data_generators()
```

### `models.py` - Define model architectures
```python
from models import build_resnet50, build_mobilenetv2, build_custom_cnn
model = build_mobilenetv2()
```

### `train_utils.py` - Training utilities
```python
from train_utils import compile_model, train_model
history = train_model(model, train_gen, val_gen, 20, class_weights, model_path)
```

### `evaluate.py` - Evaluation functions
```python
from evaluate import get_predictions, calculate_metrics, print_metrics
probs, preds, true_labels = get_predictions(model, test_gen)
```

---

## 🎓 How to Use This Project

### 1. Train Models
- Open `Pnuemonia_detection_project.ipynb` in Jupyter
- Run all cells to train all 3 models
- Models are saved to `saved_models/`

### 2. View Results
- Charts are generated in `report_charts/`
- Open PNG files to see performance metrics
- Each chart shows different aspects of model performance

### 3. Test on New Images
- Add images to `unseen_demo_live/NORMAL/` or `unseen_demo_live/PNEUMONIA/`
- Run the inference cell in the notebook
- Get predictions and risk levels

### 4. Use Best Model for Predictions
```python
from tensorflow.keras.models import load_model
from config import MOBILENET_PATH

model = load_model(MOBILENET_PATH)
# Use model.predict() on new X-ray images
```

---

## ⚙️ System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- 5GB disk space for dataset

**Recommended:**
- Python 3.10+
- 16GB RAM
- GPU (NVIDIA with CUDA support)
- 10GB disk space

---

## 🔧 Installation Details

### For GPU Support (NVIDIA)
```bash
pip install tensorflow[and-cuda]
```

### For CPU Only
```bash
pip install tensorflow
```

---

## 📚 Key Concepts Used

✓ **Transfer Learning** - Using pretrained ImageNet weights  
✓ **Data Augmentation** - Rotation, zoom, flip to increase data variety  
✓ **Class Weight Balancing** - Handle imbalanced classes  
✓ **Early Stopping** - Stop training when validation loss plateaus  
✓ **Fine-tuning** - Carefully update pretrained layers  
✓ **Callback Functions** - Monitor and save best models  

---

## 🐛 Troubleshooting

**Issue:** "CUDA out of memory"
- Solution: Reduce BATCH_SIZE in config.py or use CPU

**Issue:** "Dataset not found"
- Solution: Update TRAIN_DIR, VAL_DIR, TEST_DIR paths in config.py

**Issue:** "GPU not detected"
- Solution: Check NVIDIA drivers and CUDA installation

**Issue:** "Imbalanced training results"
- Solution: Class weights are automatically computed and applied

---

## 📄 License

This project is provided for educational and research purposes.

Dataset source: [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Original paper: *Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning" (2018)*

---

## ❓ Questions?

- Check the Jupyter notebook cells for detailed comments
- Read the docstrings in each Python file
- Review the training output and generated charts

---

**Made for pneumonia detection research and deep learning education** 🔬