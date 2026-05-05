#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PNEUMONIA DETECTION PROJECT - FINAL SUMMARY VISUALIZATIONS
Generates comprehensive visual summaries for Part 11 wrap-up
All outputs saved to report_charts/
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
import os

# Set style
plt.style.use('default')

# Paths
CHARTS_DIR = r"D:\AI & ML\pneumonia_prediction\report_charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

print("=" * 70)
print("  PNEUMONIA DETECTION PROJECT - FINAL SUMMARY VISUALIZATIONS")
print("=" * 70)
print()

# ============================================================================
# 1. JOURNEY SUMMARY CARD
# ============================================================================
print("Creating: 1_journey_summary.png")

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#F5F7FA')

# Title
fig.text(0.5, 0.95, 'Building an AI to Detect Pneumonia', 
         ha='center', fontsize=32, weight='bold', color='#1a3a52')
fig.text(0.5, 0.90, 'MSc Project - 11 Posts. 3 Models. 1 Winner.', 
         ha='center', fontsize=20, color='#2196F3')

# Main boxes - The Journey
box_y = 0.80
box_height = 0.12
box_width = 0.28
x_positions = [0.08, 0.38, 0.68]

journey_items = [
    {
        'title': 'Data Pipeline',
        'content': 'Built from scratch\nClass imbalance handled\n5,856 images processed\nTrain/Val/Test split'
    },
    {
        'title': 'Three Architectures',
        'content': 'ResNet50\nMobileNetV2\nCustom CNN\nBenchmarked fairly'
    },
    {
        'title': 'Winner Selected',
        'content': 'MobileNetV2\n90.54% Accuracy\n96.41% Recall\nAUC 0.971'
    }
]

for x_pos, item in zip(x_positions, journey_items):
    box = FancyBboxPatch((x_pos, box_y - box_height), box_width, box_height,
                         boxstyle="round,pad=0.01", 
                         facecolor='white', edgecolor='#2196F3', linewidth=2.5)
    fig.patches.append(box)
    
    fig.text(x_pos + box_width/2, box_y - 0.015, item['title'],
             ha='center', fontsize=14, weight='bold', color='#1565C0')
    
    fig.text(x_pos + 0.02, box_y - 0.055, item['content'],
             ha='left', fontsize=11, color='#424242')

# Key Learnings Section
fig.text(0.5, 0.58, 'What I Learned', 
         ha='center', fontsize=18, weight='bold', color='#1a3a52')

learnings = [
    "Environment matters as much as code",
    "Class imbalance is a clinical decision",
    "Recall != Accuracy in healthcare",
    "Explainability (Grad-CAM) is mandatory"
]

y_pos = 0.52
for learning in learnings:
    fig.text(0.5, y_pos, learning, ha='center', fontsize=12, 
             color='#2E7D32', weight='bold')
    y_pos -= 0.045

# Next Steps Section
fig.text(0.5, 0.22, 'What\'s Next', 
         ha='center', fontsize=18, weight='bold', color='#1a3a52')

next_steps = [
    "Convert to TensorFlow Lite for mobile deployment",
    "Multi-class diagnosis (bacterial vs viral vs fungal)",
    "Real hospital PACS integration"
]

y_pos = 0.16
for step in next_steps:
    fig.text(0.5, y_pos, step, ha='center', fontsize=12, 
             color='#BF360C', weight='bold')
    y_pos -= 0.045

# Footer
fig.text(0.5, 0.01, "Pneumonia Detection Project | Clinical Decision Support Tool",
         ha='center', fontsize=9, color='#757575', style='italic')

plt.axis('off')
plt.tight_layout()
path = os.path.join(CHARTS_DIR, "1_journey_summary.png")
plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='#F5F7FA')
plt.close()
print("  OK: " + path)
print()

# ============================================================================
# 2. MODEL COMPARISON
# ============================================================================
print("Creating: 2_model_comparison_detailed.png")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#FAFAFA')

fig.text(0.5, 0.97, 'Model Performance Comparison', 
         ha='center', fontsize=28, weight='bold', color='#1a3a52')
fig.text(0.5, 0.93, 'Test Set: 390 images (234 Normal, 156 Pneumonia)', 
         ha='center', fontsize=12, color='#666666', style='italic')

models_info = [
    {
        'name': 'ResNet50',
        'color': '#2196F3',
        'metrics': {
            'Accuracy': '95.64%',
            'Precision': '95.00%',
            'Recall': '98.97%',
            'AUC': '0.9924',
            'Params': '25.6M',
            'Speed': 'Slow'
        }
    },
    {
        'name': 'MobileNetV2',
        'color': '#4CAF50',
        'metrics': {
            'Accuracy': '90.54%',
            'Precision': '89.47%',
            'Recall': '96.41%',
            'AUC': '0.9711',
            'Params': '3.5M',
            'Speed': 'Fast'
        },
        'winner': True
    },
    {
        'name': 'Custom CNN',
        'color': '#FF5722',
        'metrics': {
            'Accuracy': '93.33%',
            'Precision': '95.45%',
            'Recall': '93.27%',
            'AUC': '0.9722',
            'Params': '2.1M',
            'Speed': 'Medium'
        }
    }
]

y_start = 0.85
card_height = 0.28
card_width = 0.30
x_positions = [0.05, 0.35, 0.65]
metric_keys = ['Accuracy', 'Precision', 'Recall', 'AUC', 'Params', 'Speed']

for x_pos, model_data in zip(x_positions, models_info):
    card = FancyBboxPatch((x_pos, y_start - card_height), card_width, card_height,
                          boxstyle="round,pad=0.015",
                          facecolor='white', edgecolor=model_data['color'], 
                          linewidth=3 if model_data.get('winner') else 2)
    fig.patches.append(card)
    
    if model_data.get('winner'):
        fig.text(x_pos + card_width - 0.02, y_start + 0.01, 'WINNER',
                ha='right', fontsize=11, weight='bold', color='#FF9800')
    
    fig.text(x_pos + card_width/2, y_start - 0.025, model_data['name'],
            ha='center', fontsize=14, weight='bold', color=model_data['color'])
    
    y_metric = y_start - 0.055
    for metric_key in metric_keys:
        metric_value = model_data['metrics'][metric_key]
        
        fig.text(x_pos + 0.01, y_metric, metric_key + ":",
                ha='left', fontsize=10, color='#555555', weight='bold')
        
        fig.text(x_pos + card_width - 0.01, y_metric, metric_value,
                ha='right', fontsize=10, color=model_data['color'], weight='bold')
        
        y_metric -= 0.035

# Decision
fig.text(0.5, 0.30, 'Why MobileNetV2?', 
         ha='center', fontsize=16, weight='bold', color='#1a3a52')

reasons = [
    "Excellent recall (96.41%) - catches 96% of pneumonia cases",
    "13x smaller model (3.5M vs 25.6M parameters)",
    "Perfect for mobile deployment (TensorFlow Lite)",
    "Fast inference - suitable for real-time clinical use",
    "Best balance of accuracy, size, and speed"
]

y_pos = 0.24
for reason in reasons:
    fig.text(0.5, y_pos, reason, ha='center', fontsize=11, color='#2E7D32')
    y_pos -= 0.035

plt.axis('off')
plt.tight_layout()
path = os.path.join(CHARTS_DIR, "2_model_comparison_detailed.png")
plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
plt.close()
print("  OK: " + path)
print()

# ============================================================================
# 3. MOBILENETV2 WINNER CARD
# ============================================================================
print("Creating: 3_mobilenetv2_winner_card.png")

fig = plt.figure(figsize=(12, 8))
fig.patch.set_facecolor('#E8F5E9')

main_card = FancyBboxPatch((0.05, 0.1), 0.9, 0.85,
                           boxstyle="round,pad=0.02",
                           facecolor='white', edgecolor='#4CAF50', linewidth=4)
fig.patches.append(main_card)

fig.text(0.5, 0.90, 'WINNER', ha='center', fontsize=60, color='#FFC107')

fig.text(0.5, 0.80, 'MobileNetV2', ha='center', fontsize=36, weight='bold', color='#1B5E20')
fig.text(0.5, 0.75, 'SELECTED ARCHITECTURE', ha='center', fontsize=16, 
         color='#4CAF50', weight='bold', style='italic')

metrics_display = [
    ('90.54%', 'Accuracy'),
    ('96.41%', 'Recall'),
    ('0.971', 'AUC Score')
]

x_pos_metrics = [0.18, 0.50, 0.82]
for x_pos, (value, label) in zip(x_pos_metrics, metrics_display):
    circle = Circle((x_pos, 0.55), 0.10, transform=fig.transFigure,
                   facecolor='#C8E6C9', edgecolor='#4CAF50', linewidth=2)
    fig.patches.append(circle)
    
    fig.text(x_pos, 0.56, value, ha='center', va='center', fontsize=26, 
            weight='bold', color='#1B5E20')
    
    fig.text(x_pos, 0.43, label, ha='center', fontsize=12, color='#2E7D32', weight='bold')

specs_text = "Parameters: 3.5M | Speed: FAST\nThreshold: 0.5 | F1-Score: 92.86%"

fig.text(0.5, 0.32, specs_text, ha='center', fontsize=10,
        color='#424242')

fig.text(0.5, 0.18, "Smallest model with highest recall", ha='center', fontsize=12,
        color='#FF6F00', weight='bold')
fig.text(0.5, 0.14, "Ready for TensorFlow Lite conversion", ha='center', fontsize=12,
        color='#FF6F00', weight='bold')
fig.text(0.5, 0.10, "Suitable for real-time clinical deployment", ha='center', fontsize=12,
        color='#FF6F00', weight='bold')

plt.axis('off')
plt.tight_layout()
path = os.path.join(CHARTS_DIR, "3_mobilenetv2_winner_card.png")
plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='#E8F5E9')
plt.close()
print("  OK: " + path)
print()

# ============================================================================
# 4. KEY LEARNINGS
# ============================================================================
print("Creating: 4_key_learnings.png")

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#F3E5F5')

fig.text(0.5, 0.96, 'Key Learnings', 
         ha='center', fontsize=28, weight='bold', color='#4A148C')
fig.text(0.5, 0.92, 'Unexpected Insights from This Project', 
         ha='center', fontsize=14, color='#6A1B9A', style='italic')

learnings_detailed = [
    {
        'title': 'Environment Matters',
        'content': 'Hardware failures taught me practical lessons.\nInfrastructure is as critical as code.',
        'color': '#2196F3'
    },
    {
        'title': 'Class Imbalance is Clinical',
        'content': 'Choosing class weights over SMOTE was\nabout learning from real pathology.',
        'color': '#4CAF50'
    },
    {
        'title': 'Recall != Accuracy',
        'content': 'ResNet-50 had 98.97% recall but lost.\nContext determines the right metric.',
        'color': '#FF9800'
    },
    {
        'title': 'Explainability is Mandatory',
        'content': 'A 90% model without explanation\nisn\'t deployable. Grad-CAM isn\'t optional.',
        'color': '#E91E63'
    }
]

y_start = 0.83
card_height = 0.18

for idx, learning in enumerate(learnings_detailed):
    y_pos = y_start - (idx * (card_height + 0.03))
    
    card = FancyBboxPatch((0.05, y_pos - card_height), 0.90, card_height,
                          boxstyle="round,pad=0.01",
                          facecolor='white', edgecolor=learning['color'], linewidth=2.5)
    fig.patches.append(card)
    
    fig.text(0.15, y_pos - 0.015, learning['title'],
            fontsize=12, weight='bold', color=learning['color'])
    
    fig.text(0.15, y_pos - 0.075, learning['content'],
            fontsize=10, color='#424242', va='top')

plt.axis('off')
plt.tight_layout()
path = os.path.join(CHARTS_DIR, "4_key_learnings.png")
plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='#F3E5F5')
plt.close()
print("  OK: " + path)
print()

# ============================================================================
# 5. PROJECT STATISTICS
# ============================================================================
print("Creating: 5_project_statistics.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#ECEFF1')
fig.suptitle('Project Statistics Dashboard', fontsize=20, weight='bold', color='#263238')

# Dataset distribution
ax = axes[0, 0]
splits = ['Train', 'Val', 'Test']
normal_counts = [1349, 76, 234]
pneumonia_counts = [3875, 194, 156]

x = np.arange(len(splits))
width = 0.35

ax.bar(x - width/2, normal_counts, width, label='Normal', color='#4CAF50', alpha=0.8)
ax.bar(x + width/2, pneumonia_counts, width, label='Pneumonia', color='#FF5722', alpha=0.8)

ax.set_xlabel('Dataset Split', fontsize=11, weight='bold')
ax.set_ylabel('Count', fontsize=11, weight='bold')
ax.set_title('Dataset Distribution (5,856 total)', fontsize=12, weight='bold', color='#1565C0')
ax.set_xticks(x)
ax.set_xticklabels(splits)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Model Complexity
ax = axes[0, 1]
models = ['ResNet50', 'MobileNetV2', 'Custom CNN']
params = [25.6, 3.5, 2.1]
accuracy = [95.64, 90.54, 93.33]
colors = ['#2196F3', '#4CAF50', '#FF5722']

ax.scatter(params, accuracy, s=[300, 400, 250], c=colors, alpha=0.7, edgecolors='black', linewidth=2)

for i, model in enumerate(models):
    ax.annotate(model, (params[i], accuracy[i]),
               textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, weight='bold')

ax.set_xlabel('Model Parameters (M)', fontsize=11, weight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=11, weight='bold')
ax.set_title('Complexity vs Accuracy', fontsize=12, weight='bold', color='#1565C0')
ax.grid(alpha=0.3)

# Metrics
ax = axes[1, 0]
metric_names = ['Accuracy', 'Precision', 'Recall', 'AUC']
values = [90.54, 89.47, 96.41, 97.11]

ax.barh(metric_names, values, color='#4CAF50', alpha=0.8, edgecolor='#1B5E20', linewidth=2)
ax.set_xlim([85, 100])
ax.set_xlabel('Score (%)', fontsize=11, weight='bold')
ax.set_title('MobileNetV2 Metrics', fontsize=12, weight='bold', color='#1565C0')
ax.grid(axis='x', alpha=0.3)

# Achievements
ax = axes[1, 1]
ax.axis('off')

achievements = [
    "11-part series from concept to deployment",
    "3 architectures benchmarked fairly",
    "Class imbalance handled responsibly",
    "Grad-CAM explainability implemented",
    "Live inference demo ready",
    "96.41% recall achieved",
    "TensorFlow Lite ready",
    "Clinical framework established"
]

y_text = 0.95
for achievement in achievements:
    ax.text(0.1, y_text, achievement, fontsize=10, weight='bold',
           color='#2E7D32', transform=ax.transAxes)
    y_text -= 0.11

ax.set_title('Project Achievements', fontsize=12, weight='bold', color='#1565C0', pad=20)

plt.tight_layout()
path = os.path.join(CHARTS_DIR, "5_project_statistics.png")
plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='#ECEFF1')
plt.close()
print("  OK: " + path)
print()

# ============================================================================
# 6. ETHICS STATEMENT
# ============================================================================
print("Creating: 6_ethics_statement.png")

fig = plt.figure(figsize=(12, 10))
fig.patch.set_facecolor('#FFF9C4')

fig.text(0.5, 0.96, 'Ethics & Responsible AI', 
         ha='center', fontsize=26, weight='bold', color='#F57F17')

main_box = FancyBboxPatch((0.05, 0.08), 0.90, 0.85,
                         boxstyle="round,pad=0.02",
                         facecolor='white', edgecolor='#FBC02D', linewidth=3)
fig.patches.append(main_box)

fig.text(0.5, 0.88, 'DECISION SUPPORT TOOL - NOT AUTONOMOUS AGENT', 
         ha='center', fontsize=13, weight='bold', color='#F57F17')

points = [
    {
        'title': 'EU AI Act Compliance',
        'text': 'Medical AI is high-risk. Built with those requirements.'
    },
    {
        'title': 'Human in the Loop',
        'text': 'Every positive requires radiologist review before action.'
    },
    {
        'title': 'Transparent Limitations',
        'text': '3.59% miss rate: 14/390 patients not caught by model.'
    },
    {
        'title': 'Clinical Explainability',
        'text': 'Grad-CAM shows exactly which regions influenced decision.'
    }
]

y_pos = 0.80
for point in points:
    fig.text(0.15, y_pos + 0.01, point['title'], fontsize=11, weight='bold', color='#E65100')
    
    fig.text(0.15, y_pos - 0.035, point['text'], fontsize=9, color='#424242')
    
    y_pos -= 0.18

fig.text(0.5, 0.04, "No AI is perfect. The human always stays in the loop.",
         ha='center', fontsize=11, style='italic', color='#F57F17', weight='bold')

plt.axis('off')
plt.tight_layout()
path = os.path.join(CHARTS_DIR, "6_ethics_statement.png")
plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='#FFF9C4')
plt.close()
print("  OK: " + path)
print()

# ============================================================================
# 7. ROADMAP
# ============================================================================
print("Creating: 7_roadmap_whats_next.png")

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#E3F2FD')

fig.text(0.5, 0.96, 'What\'s Next - Product Roadmap', 
         ha='center', fontsize=26, weight='bold', color='#0D47A1')

roadmap_phases = [
    {
        'phase': 'Phase 1: Mobile Deployment',
        'items': [
            'Convert to TensorFlow Lite',
            'iOS and Android optimization',
            'Real-time on-device inference',
            'Battery-efficient processing'
        ],
        'color': '#2196F3',
        'position': 0.80
    },
    {
        'phase': 'Phase 2: Multi-Class Diagnosis',
        'items': [
            'Bacterial vs viral vs fungal',
            'Expand to 4-class problem',
            'Antibiotic recommendation',
            'Treatment pathway support'
        ],
        'color': '#4CAF50',
        'position': 0.50
    },
    {
        'phase': 'Phase 3: Hospital Integration',
        'items': [
            'PACS integration',
            'Real hospital systems',
            'Multi-hospital dataset',
            'Clinical validation'
        ],
        'color': '#FF9800',
        'position': 0.20
    }
]

for phase_data in roadmap_phases:
    y_pos = phase_data['position']
    
    phase_box = FancyBboxPatch((0.05, y_pos - 0.12), 0.90, 0.12,
                              boxstyle="round,pad=0.01",
                              facecolor='white', edgecolor=phase_data['color'], linewidth=2.5)
    fig.patches.append(phase_box)
    
    fig.text(0.08, y_pos + 0.035, phase_data['phase'],
            fontsize=13, weight='bold', color=phase_data['color'])
    
    for i, item in enumerate(phase_data['items']):
        fig.text(0.25 + (i % 2) * 0.35, y_pos + 0.02 - (i // 2) * 0.035, item,
                fontsize=9, color='#424242')

metrics_box = FancyBboxPatch((0.1, 0.01), 0.8, 0.08,
                            boxstyle="round,pad=0.01",
                            facecolor='#BBDEFB', edgecolor='#0D47A1', linewidth=2)
fig.patches.append(metrics_box)

fig.text(0.5, 0.065, "Targets: >95% Sensitivity | <2% Miss Rate | <200ms Inference",
        ha='center', fontsize=10, weight='bold', color='#0D47A1')

plt.axis('off')
plt.tight_layout()
path = os.path.join(CHARTS_DIR, "7_roadmap_whats_next.png")
plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='#E3F2FD')
plt.close()
print("  OK: " + path)
print()

# ============================================================================
# 8. EXECUTIVE SUMMARY
# ============================================================================
print("Creating: 8_executive_summary.png")

fig = plt.figure(figsize=(11, 14))
fig.patch.set_facecolor('white')

header_box = FancyBboxPatch((0, 0.92), 1, 0.08,
                           boxstyle="round,pad=0.005",
                           facecolor='#1a3a52', edgecolor='none')
fig.patches.append(header_box)

fig.text(0.5, 0.965, 'PNEUMONIA DETECTION AI', 
         ha='center', fontsize=20, weight='bold', color='white')
fig.text(0.5, 0.935, 'Executive Summary - MSc Project', 
         ha='center', fontsize=11, color='#B0BEC5')

y_pos = 0.88
fig.text(0.05, y_pos, 'WHAT WAS BUILT', fontsize=12, weight='bold', color='#1a3a52')
y_pos -= 0.03
items = [
    "Binary classifier: Pneumonia vs Normal from X-rays",
    "Data pipeline: 5,856 images, class imbalance handled",
    "3 architectures: ResNet50, MobileNetV2, Custom CNN",
    "Grad-CAM explainability for clinical transparency",
    "Live inference demo for real-time prediction"
]
for item in items:
    fig.text(0.07, y_pos, item, fontsize=9, color='#424242')
    y_pos -= 0.035

y_pos -= 0.02
fig.text(0.05, y_pos, 'PERFORMANCE WINNER', fontsize=12, weight='bold', color='#1B5E20')
y_pos -= 0.03
fig.text(0.07, y_pos, 'MobileNetV2: 90.54% Accuracy | 96.41% Recall | AUC 0.971', 
         fontsize=10, weight='bold', color='#2E7D32')
y_pos -= 0.035
fig.text(0.07, y_pos, 'Selected for: Smallest (3.5M) + High recall + Mobile-ready', 
         fontsize=9, color='#424242')

y_pos -= 0.05
fig.text(0.05, y_pos, 'KEY LEARNINGS', fontsize=12, weight='bold', color='#0D47A1')
y_pos -= 0.03
learnings_short = [
    "Environment matters as much as code",
    "Class imbalance is a clinical decision",
    "Recall != Accuracy in healthcare",
    "Explainability is mandatory in production"
]
for learning in learnings_short:
    fig.text(0.07, y_pos, learning, fontsize=9, color='#424242')
    y_pos -= 0.035

y_pos -= 0.02
fig.text(0.05, y_pos, 'ETHICS & RESPONSIBILITY', fontsize=12, weight='bold', color='#F57F17')
y_pos -= 0.03
ethics_points = [
    "Decision support tool, not autonomous agent",
    "Human in the loop - radiologist review required",
    "3.59% miss rate disclosed transparently",
    "EU AI Act compliant for high-risk medical AI"
]
for point in ethics_points:
    fig.text(0.07, y_pos, point, fontsize=9, color='#E65100')
    y_pos -= 0.035

y_pos -= 0.02
fig.text(0.05, y_pos, 'WHAT\'S NEXT', fontsize=12, weight='bold', color='#C62828')
y_pos -= 0.03
next_items = [
    "TensorFlow Lite for mobile deployment",
    "Multi-class diagnosis (bacterial/viral/fungal)",
    "Hospital PACS integration and validation"
]
for item in next_items:
    fig.text(0.07, y_pos, item, fontsize=9, color='#424242')
    y_pos -= 0.035

fig.text(0.5, 0.01, '11 Posts | 3 Models | 1 Winner | Clinical-Grade Solution',
         ha='center', fontsize=10, weight='bold', color='#1565C0', style='italic')

plt.axis('off')
plt.tight_layout()
path = os.path.join(CHARTS_DIR, "8_executive_summary.png")
plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  OK: " + path)
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("  SUCCESS! 8 SUMMARY VISUALIZATIONS CREATED")
print("=" * 70)
print()
print("Generated files:")
print()
print("1. 1_journey_summary.png")
print("2. 2_model_comparison_detailed.png")
print("3. 3_mobilenetv2_winner_card.png")
print("4. 4_key_learnings.png")
print("5. 5_project_statistics.png")
print("6. 6_ethics_statement.png")
print("7. 7_roadmap_whats_next.png")
print("8. 8_executive_summary.png")
print()
print("All files saved to: " + CHARTS_DIR)
print()
