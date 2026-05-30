# CHISCO EEG-to-Text Decoding

Deep learning models for open-vocabulary EEG-to-text decoding on the CHISCO dataset, supporting both **stimulus-driven reading** and **imagined speech (recall)** paradigms.

This repository extends the original CHISCO framework with improved EEG encoders, sentence-aligned train/validation/test splits, large-scale retrieval evaluation, semantic analysis, interpretability tools, and systematic ablation studies.

---

# Overview

The goal of this project is to learn a shared embedding space between EEG recordings and natural language sentence embeddings, enabling retrieval of semantically relevant sentences directly from brain activity.

Two paradigms are supported:

- **Read** — participants read presented sentences.
- **Imagine** — participants internally recalled and imagined previously presented sentences.

The primary architecture combines:

- EEGNet-inspired feature extraction
- Spatial attention over electrodes
- Subject-specific adaptation layers
- Residual dilated temporal convolution blocks
- Projection into sentence embedding space

A faithful reproduction of the original CHISCO architecture is also included as a baseline.

---

# Repository Structure

```text
.
├── train_script_with_split.py
├── eval_global_retrieval.py
├── k_afc_evaluation.py
├── k_afc_evaluation_multi.py
├── semantic_hypothesis_testing.py
├── GRADCAM.py
├── subject_variance.py
├── plot_data_scaling.py
├── summarize_augmentation_results.py
│
├── Brain_Module.py
├── model_parts.py
├── CHISCO_baseline_model.py
├── simple_experiment_logger.py
│
├── splits/
│   └── train_val_test_split.json
│
└── ds005170/
```

---

# Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

- PyTorch
- Transformers
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- Pandas
- HuggingFace Transformers

---

# Dataset

This repository assumes access to the CHISCO dataset:

```text
ds005170/
```

Sentence embeddings should be provided as:

```text
ds005170/json/sentence_bert_embs.json
```

with format:

```json
{
  "sentence text": [0.1, 0.2, ...]
}
```

---

# Training

The primary training entry point is:

```bash
python train_script_with_split.py
```

Example:

```bash
python train_script_with_split.py \
  --task read \
  --seed 4 \
  --split-file splits/train_val_test_split.json
```

For imagined speech:

```bash
python train_script_with_split.py \
  --task imagine \
  --seed 4 \
  --split-file splits/train_val_test_split.json
```

---

# Sentence-Aligned Train / Validation / Test Splits

All experiments use a fixed sentence-aligned split:

```text
Train      80%
Validation 10%
Test       10%
```

The same sentence assignment is used across both read and imagine paradigms, ensuring direct comparability between tasks.

Each experiment stores:

```json
{
  "train_sentences": [...],
  "val_sentences": [...],
  "test_sentences": [...]
}
```

inside:

```text
split.json
```

allowing exact reconstruction of evaluation datasets.

---

# Experiment Logging

Training automatically creates experiment folders:

```text
experiments/
└── read/
    └── 2026-04-28_16-23-51__split=sentence_aligned_read_imagine__seed=4/
```

Each run contains:

```text
config.json
split.json
metrics.jsonl
best_checkpoint.pt
best_summary.json
last_checkpoint.pt
```

This enables fully reproducible experiments.

---

# Architectures

## Full Model

The default architecture combines:

- EEGNet-style front-end
- Spatial attention
- Subject adaptation layer
- Residual dilated temporal convolutions
- Projection head

## CHISCO Baseline

A reproduction of the original CHISCO architecture can be trained via:

```bash
python train_script_with_split.py \
  --task read \
  --arch chisco_baseline \
  --split-file splits/train_val_test_split.json
```

or

```bash
python train_script_with_split.py \
  --task imagine \
  --arch chisco_baseline \
  --split-file splits/train_val_test_split.json
```

---

# Global Retrieval Evaluation

Evaluate retrieval against the entire sentence corpus:

```bash
python eval_global_retrieval.py \
  --exp-dir <experiment_folder>
```

Outputs:

```text
global_retrieval.json
```

containing:

```json
{
  "top1": 0.054,
  "top5": 0.353,
  "top10": 0.678,
  "top50": 2.639
}
```

Evaluation is performed on the held-out test set reconstructed from the experiment's saved split file.

---

# K-AFC Evaluation

Evaluate K-way forced-choice retrieval:

```bash
python k_afc_evaluation.py \
  --exp-dir <experiment_folder>
```

Outputs:

```text
k_afc_results.json
k_afc_curve_top5.png
k_afc_curve_top10.png
k_afc_curve_top50.png

k_afc_curve_top5_relative.png
k_afc_curve_top10_relative.png
k_afc_curve_top50_relative.png
```

To aggregate across multiple runs:

```bash
python k_afc_evaluation_multi.py \
  --exp-root <folder>
```

---

# Random Noise Baseline

Evaluate retrieval under random EEG input:

```bash
python test_random_noise.py \
  --exp-root <folder>
```

This establishes empirical chance-level performance under identical evaluation conditions.

---

# Semantic Analysis

Semantic category preservation is evaluated via:

```bash
python semantic_hypothesis_testing.py
```

The analysis tests whether retrieved sentences belong to the same semantic category as the target sentence more frequently than expected under chance.

---

# Interpretability

Grad-CAM analysis:

```bash
python GRADCAM.py \
  --exp-root <folder>
```

Produces:

- Temporal saliency maps
- Channel importance maps
- Topographic scalp visualisations
- Spatiotemporal attribution analyses

for both read and imagine paradigms.

---

# Subject Variance Analysis

To investigate subject-specific structure:

```bash
python subject_variance.py
```

This computes covariance-based representations and compares inter-subject geometry using:

- Correlation embeddings
- Log-Euclidean geometry
- Incremental PCA
- Silhouette analysis

---

# Data Scaling Experiments

Train using fractions of the training set:

```bash
--train-data-fraction 0.1
--train-data-fraction 0.2
...
--train-data-fraction 1.0
```

Example:

```bash
python train_script_with_split.py \
  --task read \
  --train-data-fraction 0.5 \
  --split-file splits/train_val_test_split.json
```

Visualise results:

```bash
python plot_data_scaling.py
```

---

# Data Augmentation

Supported augmentations:

## Gaussian Noise

```bash
--noise-std
```

## Temporal Jitter

```bash
--jitter-max
```

## Channel Dropout

```bash
--channel-dropout-p
```

Disable all augmentation:

```bash
--no-augment-train
```

---

# Reproducibility

All experiments store:

- Configuration
- Random seed
- Dataset split
- Checkpoints
- Validation metrics
- Training curves

Recommended seeds:

```text
4
6
8
10
12
```

for reporting mean performance and 95% confidence intervals.

---

# Example Workflow

Train:

```bash
python train_script_with_split.py \
  --task read \
  --seed 4 \
  --split-file splits/train_val_test_split.json
```

Evaluate retrieval:

```bash
python eval_global_retrieval.py \
  --exp-dir experiments/read/<run>
```

Run K-AFC:

```bash
python k_afc_evaluation.py \
  --exp-dir experiments/read/<run>
```

Run semantic analysis:

```bash
python semantic_hypothesis_testing.py \
  --exp-root experiments/read
```

Generate Grad-CAM visualisations:

```bash
python GRADCAM.py \
  --exp-root experiments/read
```

---

```

In addition, please cite the original CHISCO dataset paper.
