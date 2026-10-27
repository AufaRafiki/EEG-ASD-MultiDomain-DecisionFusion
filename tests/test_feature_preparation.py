import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.signal import butter,sosfiltfilt
from feature_extraction import psd_matrix,pairwise_correlation
from preprocessing import segment_recording

def test_psd_and_windows_without_ica():
    fs=256
    rng=np.random.default_rng(9)
    signal=rng.normal(size=(16,10*fs))
    windows=segment_recording(signal)
    assert windows.shape==(4,16,4*fs)
    psd=psd_matrix(windows[0]);assert psd.shape==(16,5)
    assert np.isfinite(psd).all()
    corr=pairwise_correlation(windows[0]);assert corr.shape==(16,16)
    assert np.allclose(corr,corr.T)
    assert np.allclose(np.diag(corr),1)
