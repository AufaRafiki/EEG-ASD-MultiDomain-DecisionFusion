
# Implementation and Reproducibility

## Overview

This repository provides a Python implementation of the multi-domain EEG classification framework described in the associated IEEE Access manuscript.

The implementation integrates PSD-MLP, CWT-CNN, and Edge-Aware GCN architectures with decision-level fusion and eight-fold subject-disjoint cross-validation.

## Subject-Wise Validation

The evaluation pipeline implements the following procedures:

- Eight-fold subject-disjoint cross-validation.
- One ASD and one TD participant held out per fold.
- Consistent fold assignments across all three classification branches.
- Strict separation of participants between model-development and test partitions.
- Training-data-only feature normalization and threshold selection.
- Out-of-fold prediction generation for each classification branch.
- Subject-level classification through within-subject majority voting.
- Automated verification of participant and prediction integrity.

## Model Implementation

The repository includes the following components:

- EEG preprocessing and segmentation.
- Spectral feature extraction using relative power spectral density.
- Time-frequency representation using continuous wavelet transforms.
- Functional connectivity and graph construction.
- Three lightweight classification architectures.
- Soft-mean, majority-vote, and margin-weighted decision fusion.
- Segment-level and subject-level performance evaluation.
- Model checkpoint and experimental output management.

## Experimental Verification

The evaluation pipeline provides fold assignments, training logs, prediction scores, classification decisions, and performance metrics to facilitate reproducibility assessment.

Numerical consistency with the manuscript must be verified using the corresponding research dataset and experimental outputs.

The raw-data feature-generation pathway implements the preprocessing and feature-extraction procedures specified in this repository. Differences in feature-generation settings may affect numerical agreement with previously generated feature sets.

Additional analyses, including architecture ablation, graph sparsification sensitivity, frequency-band perturbation, and end-to-end computational benchmarking, require their corresponding experimental procedures and verified results.

## Research Scope

This implementation supports transparent evaluation of the proposed multi-domain EEG framework under subject-disjoint cross-validation.

The framework is intended for biomedical signal processing and machine-learning research. Its applicability to independent clinical populations requires further validation using larger and more diverse datasets.
