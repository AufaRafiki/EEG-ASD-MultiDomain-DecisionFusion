# EEG-ASD IEEE Access — reconstructed subject-wise code

**STATUS: reconstructed from `paper-rineng.ipynb`; NOT the historical source code used to generate the manuscript results.**

This folder is a **new, independently executable `.py` implementation** of the three model families and threshold-aligned fusion in the supplied notebook, replacing the notebook's **window-wise `StratifiedKFold`** with **strict 8-fold subject-disjoint splits**. It includes checks that **no subject occurs in both model-development and test partitions**. The code must be run on verified raw recordings or original feature caches before it can support *new* accuracy claims.

**Scope of verification:** This reconstructed implementation has not yet been demonstrated to reproduce Tables II–IX. Reported numerical results must be checked through an actual subject-wise run, log inspection, and reconciliation with the manuscript before any reproduction claim is made. Changing window-wise splits to subject-wise changes the scientific experiment, often changing accuracy considerably. The old notebook explicitly warns that window-wise results can be optimistic. Its Skema 1 inherited the window-level folds from Skema 2/3. Therefore the old notebook does **not** independently establish the manuscript's 8-fold subject-wise performance.

## Contents

- `config.py` — seed, preprocessing constants, hyperparameters.
- `preprocessing.py` — BCI2000 data reader, Butterworth 1–40 Hz, DWT-assisted ICA, 4 s / 50% windows, remove first 4 s.
- `feature_extraction.py` — raw-to-feature *reconstructed* route: PSD and **new** CWT recipe / connectivity graph construction.
- `align_caches.py` — **recommended** when original `.npz` caches are retained: join Skema 1,2,3 data by `(subject_id, recording filename, window_index)`; reject missing keys and label mismatches. No silent row-position alignment.
- `splits.py` — subject-wise paired folds and evidence CSV.
- `models.py` — model definitions reconstructed from notebook (PSD MLP 64–32–16; multichannel CWT separable CNN; residual GCN 32–48–64 with edge and node pathways).
- `train.py` — train all models on shared folds; fold-training-only normalization; training-only threshold selection for CNN/GCN; export checkpoint `.keras`, per-fold logs, fold-specific thresholds, and OOF results.
- `fusion.py` — soft-mean, majority-vote, margin-weighted fusion over strictly aligned OOF predictions.
- `evaluate.py` — window metrics, fold metrics, **within-subject majority-vote** metrics.
- `verify.py` — check OOF and subject-to-fold integrity.
- `tests/` — tests, synthetic labels only; no research results.

## Required input data and verification

1. **Original EEG raw `.dat` and/or three original `.npz` feature caches**. None were provided in this upload.
2. A **verified subject ID mapping**: `subject_id` must represent a real person, not a filename or a recording ID. If recordings from the same person are present, assign the same ID. Check carefully that the 16 anonymous subjects are actually 16 unique people.
3. If trying to reproduce the manuscript's exact Table III fold assignment, supply `configs/subject_folds.csv` with the true subject IDs and held-out fold IDs; never assume lexical sorting matches the original pairing.
4. Training computer with TensorFlow and the EEG-reading dependency.

## Installation

```bash
cd IEEE_Access_EEG_ASD_Reproducibility
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q tests
```

TensorFlow and BCI2kReader version compatibility must be checked for the selected OS/Python runtime. For true reproducibility, run `pip freeze > requirements-lock.txt` on the successful environment and keep the lock file.

## Route A: reuse historical feature caches (recommended for fidelity)

**Only when all three original caches actually exist and contain provenance columns** `y`, `groups`, `filepaths`, and `win_index`.

```bash
python align_caches.py \
  --psd-cache /path/to/skema1_psd_features_cache.npz \
  --cwt-cache /path/to/skema2_cwt_multichannel_64x128_features_cache.npz \
  --gcn-cache /path/to/skema3_connectivity_graph_features_cache.npz \
  --output outputs/aligned_features.npz

python train.py --features outputs/aligned_features.npz \
  --fold-map configs/subject_folds.csv \
  --out outputs/subject_wise --expected-windows 4262

python fusion.py --results outputs/subject_wise
python verify.py --results outputs/subject_wise
```

If original caches use inconsistent recording identifiers or filename prefixes, **resolve the mapping explicitly**, with a checked mapping log, before alignment. Create an optional CSV with `branch,original_subject_id,canonical_subject_id,original_filename,canonical_filename`, then supply `--identity-map configs/verified_cache_identities.csv` to `align_caches.py`. Each mapping row must be verified against original recording metadata. This program intentionally refuses to guess that two names identify the same recording or person. If the source cache lacks filenames or window indices, it cannot safely be merged with other caches without recovering their sample identities.

## Route B: reconstruct all features directly from BCI2000 .dat

First create `configs/manifest.csv` with three columns:

```
filepath,subject_id,y
ASD/recording_01.dat,ASD-01,1
NORMAL/recording_01.dat,TD-01,0
```

These are **illustrative example rows**, not names of actual recordings. All dataset entries must be mapped by the research team. Multiple recordings from a participant must share the same `subject_id`.

```bash
python feature_extraction.py \
  --data-root /path/to/KAU_dataset \
  --manifest configs/manifest.csv \
  --output outputs/features_reconstructed.npz --k 4

python train.py --features outputs/features_reconstructed.npz \
  --fold-map configs/subject_folds.csv \
  --out outputs/new_subjectwise --expected-windows 4262
python fusion.py --results outputs/new_subjectwise
python verify.py --results outputs/new_subjectwise
```

**Important feature-recipe limitation:** the supplied notebook only *loads* existing CWT and GCN feature caches. It does NOT contain the code that originally constructed them. Consequently, the `feature_extraction.py` implementation uses an explicitly new Morlet CWT definition, 64×128 resampling, absolute Pearson correlations, and symmetric weighted top-k. These may differ materially from the original cache builder and the resulting numbers **must not be represented as reproductions of the original manuscript experiment**. If you still have the exact original cache-generation scripts, use them instead.

## Fold logic and evaluation integrity

- 8 folds, **one ASD and one TD subject held out in each fold** when 8+8 unique subjects are provided.
- 14 development subjects and 2 held-out test subjects per fold.
- No subject's windows enter both train and test in any fold. Subject ID validation and disjointness assertions abort on violations.
- Train-only StandardScaler / channel-wise CWT normalization / train-only node and edge statistics.
- PSD-MLP threshold fixed at 0.5; CNN and GCN threshold chosen on *in-sample* predictions from development subjects, never the held-out test subjects. This imitates the notebook and retains its **in-sample threshold-selection limitation**; an inner subject-wise cross-validation would be scientifically preferable but requires a distinct protocol and results.
- Saved OOF table for every branch: `filepath`, `window_index`, `subject_id`, `fold`, `y_true`, `p_asd`, `threshold`, `y_pred`.
- Fusion uses threshold-centered logits with epsilon=1e-7; margins use gamma=1, **equal branch reliability** (not global-OFF performance weights). Post-hoc selection among fusion schemes is exploratory.
- Subject predictions are **within-subject majority vote of OOF window decisions**, not the per-window majority-vote fusion across models.

## Scientific caution and limitations

The source notebook called its setup **8-fold window-wise CV**, which is not subject-wise and may leak participant-specific information. The new code's results have not been computed in this environment (original EEG, original NPZ caches, TensorFlow runtime not provided). **No claim is made that the new script produces 98.36%, 99.60%, or any other reported accuracy.** Raw preprocessing can also vary due to preprocessing choices, reference/montage differences, ICA convergence, hardware, dependency versions, class imbalance, and participant metadata.

Do not upload private identifiable EEG or restricted data to a public repository. Check KAU dataset permissions and anonymize files. External validation, ablation variants, k-sensitivity re-training, feature perturbation and end-to-end time benchmark from the revised paper require additional verified scripts/results: **they are not generated merely by running this core reproduction package.**
