"""Explicit experiment settings adapted from paper-rineng.ipynb.

This is a reconstructed subject-wise pipeline, NOT the historical training code
that produced published/reported results. See README.md and PROVENANCE.md.
"""
from dataclasses import dataclass

SEED = 42
FS = 256
N_CHANNELS = 16
REMOVE_FIRST_SECONDS = 4
WINDOW_SECONDS = 4
OVERLAP = .50
LOWCUT, HIGHCUT = 1., 40.
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 40)}

@dataclass(frozen=True)
class Settings:
    n_folds: int = 8
    seed: int = SEED
    psd_epochs: int = 50
    psd_batch: int = 16
    psd_lr: float = 5e-4
    cwt_epochs: int = 100
    cwt_batch: int = 32
    cwt_lr: float = 1e-5
    gcn_epochs: int = 50
    gcn_batch: int = 32
    gcn_lr: float = 3e-4
    weight_decay: float = 1e-4
    dropout_gcn: float = .35
    k: int = 4
    threshold_metric: str = "balanced_accuracy"  # on in-sample train predictions

    def as_dict(self):
        from dataclasses import asdict
        return asdict(self)
