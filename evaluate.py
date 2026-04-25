"""
Evaluation utilities for model comparison
Includes metrics calculation and visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, accuracy_score, 
    precision_score, recall_score
)
from tensorflow.keras.models import load_model


def get_predictions(model, test_gen, threshold=0.5):
    """Get predictions from model on test data"""
    test_gen.reset()
    probs = model.predict(test_gen, verbose=0)
    preds = (probs > threshold).astype(int).flatten()
    true_labels = test_gen.classes
    return probs.flatten(), preds, true_labels


def calculate_metrics(true_labels, preds):
    """Calculate accuracy, precision, recall"""
    acc = accuracy_score(true_labels, preds) * 100
    prec = precision_score(true_labels, preds) * 100
    rec = recall_score(true_labels, preds) * 100
    return acc, prec, rec


def print_metrics(model_name, true_labels, preds):
    """Print metrics for a model"""
    acc, prec, rec = calculate_metrics(true_labels, preds)
    print(f"\n{model_name} Results:")
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  Precision : {prec:.2f}%")
    print(f"  Recall    : {rec:.2f}%")
    return acc, prec, rec


def plot_confusion_matrix(true_labels, preds, model_name, ax, color):
    """Plot confusion matrix for a model"""
    cm = confusion_matrix(true_labels, preds)
    cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap=sns.light_palette(color, as_cmap=True),
                ax=ax, xticklabels=["Normal", "Pneumonia"],
                yticklabels=["Normal", "Pneumonia"], cbar=False, annot_kws={"size": 13})
    
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn) * 100
    rec = tp / (tp + fn) * 100
    
    ax.set_title(f"{model_name}\nAccuracy: {acc:.1f}%   Recall: {rec:.1f}%", fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True Label")
    
    return cm


def plot_roc_curve(true_labels, probs, model_name, ax, color):
    """Plot ROC curve for a model"""
    fpr, tpr, _ = roc_curve(true_labels, probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{model_name}  AUC = {roc_auc:.3f}")
    return roc_auc
