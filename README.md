
# EEG-ASD Multi-Domain Learning and Decision Fusion

### A Lightweight Multi-Domain EEG Framework with Decision Fusion for Autism Spectrum Disorder Identification

**Associated Manuscript:** IEEE Access — Access-2026-44649

## Overview

This repository provides a Python implementation of a lightweight multi-domain electroencephalography (EEG) framework for autism spectrum disorder (ASD) identification.

The framework integrates three complementary EEG representations and classification architectures:

1. **PSD-MLP:** Power spectral density features classified using a multilayer perceptron.
2. **CWT-CNN:** Continuous wavelet transform scalograms analyzed using a depthwise separable convolutional neural network.
3. **Edge-Aware GCN:** Functional connectivity and spectral node features processed using a graph convolutional network with explicit edge-feature and node-feature pathways.

The three classification branches are integrated through decision-level fusion strategies, including soft-mean fusion, majority voting, and threshold-aligned margin-weighted fusion.

The implementation supports **eight-fold subject-disjoint cross-validation**, ensuring that EEG segments from the same participant are not distributed across model-development and test partitions within a fold.

The repository also includes subject-level evaluation, prediction alignment, experimental logging, and verification utilities to support reproducible research.

---

## 1. Dataset and Evaluation Protocol

The framework is configured for resting-state EEG recordings from 16 participants:

| Group | Participants |
|---|---:|
| Autism Spectrum Disorder (ASD) | 8 |
| Typically Developing (TD) | 8 |
| Total | 16 |

The EEG recordings contain 16 channels sampled at 256 Hz.

### Subject-Disjoint Cross-Validation

An eight-fold subject-disjoint cross-validation procedure is implemented.

For each fold:

- One ASD subject and one TD subject are held out for testing.
- The remaining 14 participants are used for model development.
- All EEG windows from a participant remain within the same partition.
- Identical fold assignments are used across the three classification branches.
- Data-dependent normalization parameters are estimated using model-development subjects only.
- Predictions from held-out subjects are saved for out-of-fold evaluation.

The implementation contains checks to prevent participant overlap between development and test partitions.

Each participant serves as a held-out test subject exactly once across the eight folds.

### Evaluation Levels

Performance evaluation is supported at two levels:

**Segment-Level Evaluation**

Classification metrics are calculated from the pooled out-of-fold predictions of individual EEG segments.

**Subject-Level Evaluation**

Segment predictions belonging to each held-out participant are aggregated using within-subject majority voting.

The subject-level procedure produces one final classification decision per participant.

---

## 2. EEG Preprocessing

The EEG preprocessing pipeline includes:

1. Importing EEG recordings stored in BCI2000 `.dat` format.
2. Removing the first four seconds of each recording.
3. Applying fourth-order Butterworth band-pass filtering between 1 and 40 Hz.
4. Applying Daubechies-4 discrete wavelet transform (DWT) denoising.
5. Performing independent component analysis (ICA) for artifact attenuation.
6. Segmenting the EEG signals into four-second windows with 50% overlap.
7. Maintaining consistent segment identifiers across the three feature representations.

At a sampling frequency of 256 Hz:

- Window duration: 4 seconds.
- Samples per window: 1,024.
- Window overlap: 50%.
- Window stride: 512 samples.
- Number of EEG channels: 16.

---

## 3. Feature Extraction and Model Architectures

### 3.1 PSD-MLP Branch

The PSD-MLP branch extracts relative spectral power features using Welch's method.

Five frequency bands are considered:

| Frequency Band | Frequency Range |
|---|---|
| Delta | 1–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–40 Hz |

Each EEG segment produces a 16 × 5 spectral representation, which is flattened into an 80-dimensional feature vector.

The MLP architecture consists of:

- Input: 80 spectral features.
- Dense layer: 64 units.
- Dense layer: 32 units.
- Dense layer: 16 units.
- Sigmoid classification output.

Normalization, dropout, and regularization are used to support stable model training.

### 3.2 CWT-CNN Branch

The CWT-CNN branch extracts time-frequency representations from multichannel EEG signals.

Continuous wavelet transform scalograms represent the temporal evolution of spectral activity.

The input tensor has dimensions:

`64 × 128 × 16`

The lightweight CNN comprises:

- Initial Conv2D block with 24 filters.
- Depthwise separable convolutional blocks with 48, 96, and 128 filters.
- Batch normalization and nonlinear activation.
- Pooling and dropout.
- Global average pooling.
- Dense layer with 64 units.
- Sigmoid classification output.

Depthwise separable convolutions are employed to reduce the number of trainable parameters.

### 3.3 Edge-Aware GCN Branch

The edge-aware GCN represents EEG channels as graph nodes and inter-channel statistical relationships as graph edges.

The framework uses:

- 16 EEG nodes.
- Five relative PSD features per node.
- Absolute Pearson correlation for functional connectivity representation.
- Top-k graph sparsification.
- Symmetric adjacency construction.
- Graph normalization and convolution.

The GCN integrates three processing pathways:

**Graph Pathway**

Residual graph convolutional blocks with hidden dimensions of 32, 48, and 64 units.

**Edge Pathway**

A dedicated pathway processing 120 unique pairwise connectivity features through dense layers.

**Node-Feature Pathway**

A dedicated pathway processing the spectral node-feature representation.

The three latent representations are concatenated and processed through a fusion head before binary classification.

The default graph sparsification setting is:

`k = 4`

---

## 4. Decision-Level Fusion

The framework includes three decision-level fusion approaches.

### 4.1 Soft-Mean Fusion

Soft-mean fusion averages the threshold-aligned prediction scores from the three classification branches.

### 4.2 Majority-Vote Fusion

Majority voting combines the binary predictions generated by the three branches.

The final class is determined by the majority of branch-level decisions.

### 4.3 Margin-Weighted Fusion

Margin-weighted fusion assigns weights according to the distance between each threshold-aligned prediction score and the decision boundary.

The default margin-weighting exponent is:

`gamma = 1`

Fusion is performed on matched out-of-fold predictions using consistent subject identifiers, recording identifiers, window indices, class labels, and fold assignments.

No additional trainable fusion parameters are required.

The comparison among fusion strategies is exploratory because no separate validation set is used to select the preferred strategy.

---

## 5. Repository Structure

```text
IEEE_Access_EEG_ASD_Reproducibility/
│
├── config.py
├── preprocessing.py
├── feature_extraction.py
├── align_caches.py
├── splits.py
├── models.py
├── train.py
├── fusion.py
├── evaluate.py
├── verify.py
├── requirements.txt
├── README.md
│
├── configs/
│   ├── manifest.csv
│   └── subject_folds.csv
│
├── tests/
│
└── outputs/
```

### Module Descriptions

| Module | Description |
|---|---|
| `config.py` | Experimental constants, random seeds, and model configurations |
| `preprocessing.py` | EEG loading, filtering, denoising, ICA, and segmentation |
| `feature_extraction.py` | Spectral, time-frequency, and graph feature generation |
| `align_caches.py` | Alignment of feature caches using verified sample identities |
| `splits.py` | Subject-disjoint cross-validation and fold verification |
| `models.py` | PSD-MLP, CWT-CNN, and edge-aware GCN architectures |
| `train.py` | Training, threshold selection, checkpoint saving, and out-of-fold predictions |
| `fusion.py` | Decision-fusion strategies |
| `evaluate.py` | Segment-level and subject-level classification evaluation |
| `verify.py` | Prediction integrity and subject-to-fold validation |
| `tests/` | Software tests using synthetic examples |

---

## 6. Installation

### Step 1: Prepare the Environment

Open a terminal in the project directory.

```bash
cd IEEE_Access_EEG_ASD_Reproducibility
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

TensorFlow and BCI2000 reading dependencies must be compatible with the selected Python environment.

### Step 3: Run Software Tests

```bash
python -m pytest -q tests
```

These tests check software functionality using synthetic examples. They do not independently establish the experimental performance reported in the associated manuscript.

For environment documentation:

```bash
pip freeze > requirements-lock.txt
```

---

## 7. Data Preparation

Two data preparation workflows are supported.

### Route A: Existing Feature Caches

This approach is suitable when the original PSD, CWT, and connectivity feature caches are available.

Each cache must contain the necessary provenance information:

- `y`
- `groups`
- `filepaths`
- `win_index`

Feature alignment is performed using participant identifiers, recording filenames, and window indices.

Run:

```bash
python align_caches.py \
  --psd-cache /path/to/skema1_psd_features_cache.npz \
  --cwt-cache /path/to/skema2_cwt_multichannel_64x128_features_cache.npz \
  --gcn-cache /path/to/skema3_connectivity_graph_features_cache.npz \
  --output outputs/aligned_features.npz
```

The alignment procedure checks sample correspondence and class-label consistency.

If recording identifiers differ among feature caches, a verified identity mapping must be supplied.

Example:

```bash
python align_caches.py \
  --psd-cache /path/to/skema1_psd_features_cache.npz \
  --cwt-cache /path/to/skema2_cwt_multichannel_64x128_features_cache.npz \
  --gcn-cache /path/to/skema3_connectivity_graph_features_cache.npz \
  --identity-map configs/verified_cache_identities.csv \
  --output outputs/aligned_features.npz
```

Missing identifiers and inconsistent sample identities must be resolved before combining the three feature sets.

### Route B: Raw EEG Recordings

This approach generates feature representations directly from BCI2000 EEG recordings.

Prepare a `configs/manifest.csv` file containing:

```csv
filepath,subject_id,y
ASD/recording_01.dat,ASD-01,1
NORMAL/recording_01.dat,TD-01,0
```

The entries above are illustrative. Actual recording paths and participant identities must be verified against the source dataset.

Run feature extraction:

```bash
python feature_extraction.py \
  --data-root /path/to/KAU_dataset \
  --manifest configs/manifest.csv \
  --output outputs/features_reconstructed.npz \
  --k 4
```

The raw-data feature-generation pathway uses the CWT and graph-construction procedures implemented in this repository.

Numerical agreement with feature caches generated through other preprocessing implementations is not assumed.

---

## 8. Subject-Wise Model Training

Train the three classification branches using eight-fold subject-disjoint cross-validation.

For aligned feature caches:

```bash
python train.py \
  --features outputs/aligned_features.npz \
  --fold-map configs/subject_folds.csv \
  --out outputs/subject_wise \
  --expected-windows 4262
```

For features generated from raw EEG recordings:

```bash
python train.py \
  --features outputs/features_reconstructed.npz \
  --fold-map configs/subject_folds.csv \
  --out outputs/new_subjectwise \
  --expected-windows 4262
```

The `--expected-windows` argument provides an additional consistency check against the expected number of segments.

The fold mapping must represent verified participant identities and intended held-out assignments.

If the segment count differs because of recording duration, preprocessing, or dataset differences, the discrepancy must be investigated rather than forcing the expected value.

### Training Configuration

The implementation supports:

- Shared subject-disjoint folds across all three branches.
- Fold-specific normalization using model-development data.
- Fixed random seed configuration.
- Independent model training.
- Branch-specific prediction thresholds.
- Model checkpoint export.
- Training logs.
- Out-of-fold prediction export.

The PSD-MLP uses a fixed threshold of 0.5.

CWT-CNN and edge-aware GCN thresholds are determined using in-sample model-development predictions, without using held-out test labels.

This threshold-selection procedure does not provide the same protection against selection bias as independent inner subject-wise validation.

---

## 9. Decision Fusion and Evaluation

### Step 1: Execute Decision Fusion

```bash
python fusion.py --results outputs/subject_wise
```

The fusion module combines aligned out-of-fold predictions produced by the three branches.

### Step 2: Verify Prediction Integrity

```bash
python verify.py --results outputs/subject_wise
```

The verification procedure checks subject and fold consistency, including the separation of development and test participants.

### Step 3: Examine Evaluation Outputs

The evaluation framework supports the following metrics:

- Accuracy
- Balanced accuracy
- Precision
- Sensitivity
- Specificity
- F1-score
- Matthews correlation coefficient (MCC)
- Area under the ROC curve (AUC)

Subject-level evaluation uses majority voting over the out-of-fold predictions belonging to each participant.

This subject-level aggregation is distinct from majority-vote fusion across the three classification branches.

---

## 10. Reproducibility and Experimental Integrity

The implementation supports the export of experimental artifacts for reproducibility assessment.

These include:

- Subject-wise fold assignments.
- Model-development and held-out subject information.
- Model checkpoints.
- Training logs.
- Branch-specific decision thresholds.
- Out-of-fold prediction scores.
- Segment-level classification decisions.
- Subject-level aggregated predictions.
- Performance evaluation outputs.

Each branch's out-of-fold predictions can include:

```text
filepath
window_index
subject_id
fold
y_true
p_asd
threshold
y_pred
```

The evaluation process supports verification that:

1. Each participant belongs to only one held-out fold.
2. No participant occurs in both development and test partitions within a fold.
3. All three branches use compatible subject and segment identifiers.
4. Fusion operates on matched out-of-fold predictions.
5. Subject-level predictions are aggregated from the corresponding held-out segments.
6. Classification metrics are calculated from the saved predictions and ground-truth labels.

### Manuscript Result Verification

The associated manuscript reports classification performance, architectural comparisons, and computational characteristics of the proposed framework.

To establish numerical reproducibility, the implementation must be executed using verified research data, and the resulting predictions and evaluation outputs must be reconciled with the manuscript tables.

The repository's core execution pipeline supports the main classification and decision-fusion procedures.

Supplementary analyses, including architecture ablation, graph sparsification sensitivity, frequency-band perturbation, and end-to-end timing, require their corresponding verified experimental runs and outputs.

---

## 11. Methodological Considerations

The framework is designed to minimize subject-level information leakage through subject-disjoint data partitioning.

Nevertheless, several considerations remain important:

- The independent participant sample is small.
- Overlapping EEG windows are not statistically independent observations.
- In-sample threshold selection can produce unstable estimates.
- Fusion strategies compared retrospectively do not establish independent superiority.
- EEG functional connectivity can be affected by volume conduction and common-reference effects.
- Performance on a single-source dataset does not establish external clinical generalizability.
- Model-level inference latency does not represent complete end-to-end processing time.

The framework is intended for biomedical signal processing and machine-learning research, rather than standalone clinical diagnosis.

---

## 12. Data Availability and Ethical Considerations

The framework is designed for secondary analysis of resting-state EEG recordings associated with the King Abdulaziz University BCI Laboratory dataset.

Data access, redistribution, and use must comply with the permissions and ethical conditions established by the original data provider.

Participant identifiers used in the implementation must not expose personally identifiable information.

Raw EEG recordings and restricted research datasets should not be uploaded to a public repository without appropriate authorization.

---

## 13. Research Purpose

This repository is provided to support reproducible research in:

- Biomedical signal processing.
- Electroencephalography analysis.
- Autism spectrum disorder identification.
- Multi-domain feature learning.
- Graph neural networks.
- Lightweight deep learning.
- Decision-level fusion.
- Subject-independent EEG classification.

The implementation is intended to facilitate transparent methodological evaluation, independent experimentation, and further development of EEG-based machine-learning approaches.
