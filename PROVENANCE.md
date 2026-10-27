# Provenance and scope of conversion

**Source:** `paper-rineng.ipynb` supplied by the author.

- Cell 3: PSD–MLP, raw BCI2000 preprocessing, PSD extraction, 8-fold master window-wise CV (master sourced from CWT or GCN OOF).
- Cell 7: CWT–CNN, `StratifiedKFold` at window level on a pre-existing NPZ cache.
- Cell 10: Edge-aware GCN, `StratifiedKFold` at window level on a pre-existing graph NPZ cache.
- Cell 12: OOF fusion on aligned windows; no verification that subjects are held out as a group.
- Cells 4, 8, 11: post-training explainability and model complexity analyses; **not ported** to this focused core CV training pipeline.

**Material changes:** all three models share subject-wise held-out test folds; explicit subject-index integrity checks, fold evidence and checkpoint artifacts; fail-closed metadata-based cache joins; no reliance on older window-wise OOF fold IDs; no use of outer-test labels in threshold selection. A new raw CWT/graph feature builder is supplied because the notebook does not provide original builders; it is explicitly a reconstruction, not an exact transcription.

**Not verified:** the original underlying dataset's participant IDs, per-channel EEG ordering, original CWT scaling and resize, original top-k adjacency derivation, training output values, neural-network parameter totals in the current TensorFlow version, timing claims, feature perturbation and ablation results, and whether the old notebook's OOF predictions agree with the article. Independent researchers must verify these before presenting this reconstructed pipeline as reproducible evidence for the manuscript.
