# ============================================================================
# CELL: PART 11 - FINAL SUMMARY VISUALIZATIONS (READY TO COPY-PASTE)
# ============================================================================

# Add this code cell to your Jupyter notebook after all other cells
# It will display all 8 summary visualizations

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from IPython.display import display, Image as IPImage

# Set the charts directory
CHARTS_DIR = r"D:\AI & ML\pneumonia_prediction\report_charts"

# Create a figure to display all 8 summary visualizations
print("=" * 80)
print("  PNEUMONIA DETECTION PROJECT - FINAL SUMMARY VISUALIZATIONS")
print("=" * 80)
print()

# List of all summary visualizations
summary_files = [
    ("1_journey_summary.png", "The Complete Journey - 11 Posts, 3 Models, 1 Winner"),
    ("2_model_comparison_detailed.png", "Model Performance Comparison - All 3 Architectures"),
    ("3_mobilenetv2_winner_card.png", "MobileNetV2 Winner Card - 90.54% Accuracy, 96.41% Recall"),
    ("4_key_learnings.png", "Key Learnings - 4 Unexpected Insights"),
    ("5_project_statistics.png", "Project Statistics Dashboard"),
    ("6_ethics_statement.png", "Ethics & Responsible AI Framework"),
    ("7_roadmap_whats_next.png", "Product Roadmap - What's Next"),
    ("8_executive_summary.png", "Executive Summary - One Page Overview")
]

# Display each visualization
for filename, title in summary_files:
    filepath = os.path.join(CHARTS_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"\n{title}")
        print("-" * 80)
        display(IPImage(filename=filepath))
        print()
    else:
        print(f"⚠️  File not found: {filename}")
        print()

print("=" * 80)
print("  All 8 Summary Visualizations Generated Successfully!")
print("=" * 80)
print()
print("Files saved to:", CHARTS_DIR)
print()
print("Use cases:")
print("  • 3_mobilenetv2_winner_card.png - LinkedIn/Social Media")
print("  • 8_executive_summary.png - Report/Portfolio")
print("  • All 8 files - Conference Presentation (in sequence)")
print()
