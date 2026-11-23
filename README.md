# CHISCO EEG-to-Text Decoding

Code for decoding sentences from EEG on the [CHISCO dataset](https://github.com/zhangzihan-is-good/Chisco), for both **reading** and **imagined speech** (recall).

An EEG encoder is trained with a contrastive (CLIP-style) loss to map each trial into the embedding space of a frozen Chinese BERT, so that the sentence a participant read or imagined can be retrieved from their brain activity. The repository includes the training code, a re-implementation of the original CHISCO encoder as a baseline, and the evaluation and analysis scripts used in the paper.

## Model

The main encoder combines:

- an EEGNet-style front end (temporal convolution + depthwise spatial convolution),
- spatial attention over sensors, a subject-specific layer, and residual dilated convolution blocks, following the brain module of Défossez et al. (2023), *Decoding speech perception from non-invasive brain recordings*, Nature Machine Intelligence,
- temporal pooling and an MLP projection into the 768-d sentence-embedding space.

Each component can be switched off from the command line for ablations.

## Repository layout

```text
train.py                    Train a model (main entry point)
eeg2text/                   Shared code
  data.py                     Loading CHISCO trials, datasets, augmentation
  model.py                    Full model + build_model(config)
  brain_module.py             BrainModule
  model_parts.py              Front end, spatial attention, subject layer, residual blocks
  chisco_baseline.py          CHISCO baseline encoder
  experiment_logger.py        Run folders, checkpoints, metrics
  eval_utils.py               Helpers shared by evaluation scripts
evaluation/
  global_retrieval.py         Retrieval against the full sentence corpus
  k_afc.py                    K-alternative forced choice
  semantic_enrichment.py      Are retrieved sentences from the target's semantic category?
  random_noise_baseline.py    Empirical chance level with noise input
analysis/
  gradcam.py                  Temporal Grad-CAM and channel / scalp saliency
  subject_variance.py         Between- vs within-subject structure of the EEG
plotting/
  plot_data_scaling.py        Accuracy vs amount of training data
  plot_kafc_curves.py         K-AFC curves aggregated over runs
scripts/
  precompute_embeddings.py    Sentence embeddings used as training targets
splits/
  train_val_test_split.json   Fixed sentence-level train / val / test split
resources/
  montage.csv                 Electrode positions, for topomaps
```

All commands below are run from the repository root.

## Setup

```bash
pip install -r requirements.txt
```

### Data

Download CHISCO ([OpenNeuro ds005170](https://doi.org/10.18112/openneuro.ds005170.v1.1.2)) into `ds005170/`, for example:

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds005170 ds005170
```

The code uses the preprocessed pickles in `ds005170/derivatives/preprocessed_pkl/` and the sentence metadata in `ds005170/json/`. Then compute the sentence-embedding targets (the `[CLS]` vector of `bert-base-chinese` for every sentence):

```bash
python -m scripts.precompute_embeddings
```

This writes `ds005170/json/sentence_bert_embs.json`.

## Data split

All experiments use the fixed split in `splits/train_val_test_split.json`, made at the **sentence** level (80 / 10 / 10 %), so test sentences are never seen during training. The same sentences are held out for reading and imagined speech, so the two tasks are directly comparable.

## Training

```bash
python train.py --task read --seed 4
python train.py --task imagine --seed 4
```

The CHISCO baseline:

```bash
python train.py --task read --arch chisco_baseline
```

Each run gets its own folder, `experiments/eeg2text/<task>/<split>/<timestamp>__split=...__seed=.../`, containing `config.json`, `split.json`, `metrics.jsonl`, `best_checkpoint.pt` (best validation Acc@10), `last_checkpoint.pt` and training curves. At the end of training the best checkpoint is evaluated once on the test split.

We report the mean and 95% CI over seeds 4, 6, 8, 10 and 12.

Useful options (see `python train.py --help` for all):

| Option | Purpose |
|---|---|
| `--no-use-frontend`, `--no-use-eegnet`, `--no-use-brainmodule`, `--no-use-spatial-attn`, `--no-use-subject-layer`, `--no-use-residual-blocks` | Ablations |
| `--train-data-fraction 0.5` | Train on a random subset of the training trials |
| `--noise-std`, `--jitter-max`, `--channel-dropout-p`, `--no-augment-train` | Data augmentation (Gaussian noise, circular time shift, channel dropout) |

## Evaluation

All evaluation scripts use the held-out **test** split recorded in the run's `split.json` and the run's `best_checkpoint.pt`.

**Global retrieval**: rank all 6,567 corpus sentences for each test trial and report top-1/5/10/50 accuracy. Writes `global_retrieval.json` and per-trial results to `global_retrieval_trials.json`.

```bash
python -m evaluation.global_retrieval --exp-dir <run folder>
```

**K-AFC**: the true sentence plus K−1 random distractors; top-k accuracy as K grows. Writes `k_afc_results.json` and curves to the run folder. To aggregate runs into one figure per task:

```bash
python -m evaluation.k_afc --exp-dir <run folder>
python -m plotting.plot_kafc_curves --exp-root experiments/eeg2text
```

**Semantic enrichment**: how often the top-k retrieved sentences share the target's semantic category, compared with chance (hypergeometric null, one-sided test).

```bash
python -m evaluation.semantic_enrichment --exp-root experiments/eeg2text
```

**Random-noise baseline**: the same retrieval with the EEG replaced by Gaussian noise, giving an empirical chance level.

```bash
python -m evaluation.random_noise_baseline --exp-root experiments/eeg2text
```

Scripts that take `--exp-root` search it recursively for run folders, evaluate each one and also save a summary with the mean ± 95% CI per task.

## Analysis

**Grad-CAM**: temporal Grad-CAM on the BrainModule output, and input-gradient saliency per channel and over time, averaged over test trials (and over runs with `--runs-root`). Scalp topomaps additionally need a text file listing the 122 model input channels in order:

```bash
python -m analysis.gradcam --runs-root experiments/eeg2text --channels-txt channels.txt
```

**Subject variance**: each trial is represented by its log-Euclidean channel-correlation matrix; we compare between-subject and within-subject spread for reading vs imagined speech (centroid distances, silhouette scores, bootstrap CIs).

```bash
python -m analysis.subject_variance --outdir covcorr_report
```

**Data scaling**: after training and evaluating runs with different `--train-data-fraction` values,

```bash
python -m plotting.plot_data_scaling --root <folder with those runs>
```

## Citation

If you use this code, please also cite the CHISCO dataset:

> Zhang, Z., Zhao, Y., Bao, Y., & Ding, X. *Chisco: An EEG-based BCI dataset for decoding of imagined speech.* Dataset: [doi:10.18112/openneuro.ds005170.v1.1.2](https://doi.org/10.18112/openneuro.ds005170.v1.1.2)

<!-- Add the BibTeX for this paper here. -->
