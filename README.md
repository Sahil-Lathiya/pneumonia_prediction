# Pneumonia X-ray Classification Experiment

An MSc deep-learning experiment comparing ResNet50, MobileNetV2, and a custom CNN on the public **Chest X-Ray Images (Pneumonia)** dataset. The repository demonstrates data preparation, transfer learning, class weighting, threshold-based evaluation, and model comparison.

> **Research and education only.** This project is not a medical device, does not provide a diagnosis, and has not been clinically validated.

## What is verified here

The notebook contains code for training and evaluating three binary image classifiers. Its archived local run recorded the following test-set results on 624 images:

| Model | Accuracy | Precision | Recall | Decision threshold |
| --- | ---: | ---: | ---: | ---: |
| ResNet50 | 78.04% | 74.37% | 98.97% | 0.50 |
| MobileNetV2 | 90.54% | 89.31% | 96.41% | 0.50 |
| Custom CNN | 83.49% | 90.65% | 82.05% | 0.35 |

![Grouped bar chart of the three archived metrics for each model](docs/archived_metric_comparison.png)

These are **recorded outputs from one archived local run**, not results reproduced by CI. The dataset and trained model files are not stored in this repository, so the figures cannot be independently rerun from the repository alone. Machine-readable provenance is in [`results/reported_metrics.json`](results/reported_metrics.json), and the full evidence review is in [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md).

## Experiment design

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Task: binary classification (`NORMAL` or `PNEUMONIA`)
- Input size: 224 × 224 pixels
- Training split: deterministic 80/20 split of the dataset's `train` folder
- Test split: the dataset's separate 624-image `test` folder
- Architectures: ImageNet-initialised ResNet50, ImageNet-initialised MobileNetV2, and a custom CNN
- Imbalance handling: class weights
- Random seed: 42 where supported

The dataset's supplied 16-image `val` folder is checked for completeness but is not used for model selection. Validation is instead taken from 20% of the training folder, matching the notebook code.

## Repository layout

```text
.
├── Pnuemonia_detection_project.ipynb  # Cleared, portable experiment notebook
├── config.py                           # Paths and experiment settings
├── data_pipeline.py                    # Training/validation/test generators
├── models.py                           # Three model architectures
├── train_utils.py                      # Compilation and callbacks
├── evaluate.py                         # Evaluation helpers
├── generate_summary.py                 # Honest chart from archived metrics
├── results/reported_metrics.json       # Recorded results and provenance
├── VALIDATION_REPORT.md                # Evidence and limitation review
└── tests/                              # Lightweight integrity tests
```

The dataset, saved models, and generated charts are intentionally excluded from Git.

## Run locally

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the Kaggle dataset and place it at `dataset/chest_xray`, or set an absolute custom path:

```bash
export PNEUMONIA_DATASET_DIR=/path/to/chest_xray
jupyter notebook Pnuemonia_detection_project.ipynb
```

Expected dataset layout:

```text
chest_xray/
├── train/{NORMAL,PNEUMONIA}/
├── val/{NORMAL,PNEUMONIA}/
└── test/{NORMAL,PNEUMONIA}/
```

Generate the archived-results comparison chart without TensorFlow or the dataset:

```bash
python generate_summary.py
```

## Important limitations

- The raw images and trained weights are unavailable in the repository.
- No patient-level identifier audit was possible, so patient overlap cannot be ruled out.
- Duplicate and near-duplicate image checks were not recorded.
- There is no external hospital or demographic validation set.
- Thresholds were selected during local experimentation; no nested validation or calibration analysis is recorded.
- Accuracy, precision, and recall are preserved, but unsupported README AUC claims were removed.
- The data are paediatric chest X-rays; performance must not be generalised to other populations or settings.
- Confidence scores are model outputs, not calibrated medical probabilities.

## Reproducibility status

CI checks syntax, notebook portability, results provenance, and documentation integrity. It deliberately does not train the models because the dataset and weights are not included. A future independently reproducible release should add a data manifest with hashes, environment lock, run metadata, trained model checksums, patient-grouped split verification, and an external evaluation protocol.

## Licence

Code is available under the [MIT License](LICENSE). The external dataset has its own terms and is not covered by this repository's licence.
