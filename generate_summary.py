"""Plot the archived notebook metrics with explicit evidence caveats."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "reported_metrics.json"
OUTPUT_PATH = ROOT / "docs" / "archived_metric_comparison.png"


def load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def main() -> None:
    results = load_results()
    models = results["models"]
    names = [item["model"] for item in models]
    metrics = {
        "Accuracy": [item["accuracy_percent"] for item in models],
        "Precision": [item["precision_percent"] for item in models],
        "Recall": [item["recall_percent"] for item in models],
    }

    x = np.arange(len(names))
    width = 0.24
    colors = ["#285F8F", "#D19A3E", "#A95D75"]
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.20)

    for index, ((metric, values), color) in enumerate(zip(metrics.items(), colors)):
        offset = (index - 1) * width
        bars = ax.bar(x + offset, values, width, label=metric, color=color)
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3, fontsize=8)

    ax.set_title("Archived test metrics by model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Recorded score (%)")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=3, loc="upper center")
    fig.text(
        0.5,
        0.035,
        "Source: archived notebook output; 624 test images; not reproduced in CI; not for medical use.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
