"""Port of source notebook BCI2000 reader, bandpass, DWT-assisted ICA, windows.

Potentially expensive per-recording ICA. The notebook assumed 256 Hz instead
of reading it from metadata. Here sampling rate MUST be user-verified.
"""
import numpy as np
from scipy import signal
from scipy.integrate import trapezoid
from scipy.stats import kurtosis
from config import FS,N_CHANNELS,REMOVE_FIRST_SECONDS,WINDOW_SECONDS,OVERLAP

def load_bci2000_dat(path,fs=FS):
    try:from BCI2kReader import BCI2kReader as module
    except ImportError as ex:raise ImportError('Install compatible BCI2kReader and verify .dat reader API') from ex
    cls=getattr(module,'BCI2kReader',module)
    obj=cls(str(path))
    x=np.asarray(obj.signals,dtype=np.float64)
    if x.ndim!=2:raise ValueError('Not two-dimensional EEG')
    if x.shape[0]!=N_CHANNELS and x.shape[1]==N_CHANNELS:x=x.T
    if x.shape[0]!=N_CHANNELS:raise ValueError(f'Expected {N_CHANNELS} channels; got {x.shape}')
    x=x[:,int(fs*REMOVE_FIRST_SECONDS):]
    if x.shape[1]<4*fs:raise ValueError('Recording shorter than one window')
    return np.nan_to_num(x)

def dwt_denoise_1d(x,threshold_scale=.8):
    import pywt
    coeff=pywt.wavedec(np.asarray(x,dtype=np.float64),'db4')
    detail=coeff[-1]
    sigma=np.median(np.abs(detail-np.median(detail)))/.6745 if detail.size else 0
    t=threshold_scale*sigma*np.sqrt(2*np.log(len(x)))
    clean=[coeff[0]]+[pywt.threshold(c,value=t,mode='soft') for c in coeff[1:]]
    return pywt.waverec(clean,'db4')[:len(x)]

def high_frequency_ratio(z,fs=FS):
    f,p=signal.welch(z,fs=fs,nperseg=min(len(z),fs*2))
    total=trapezoid(p,f)+1e-12
    mask=f>=25
    high=trapezoid(p[mask],f[mask]) if mask.any() else 0
    return float(high/total)

def preprocess_recording(raw,fs=FS,seed=42):
    from sklearn.decomposition import FastICA
    b,a=signal.butter(4,[1/(fs/2),40/(fs/2)],btype='bandpass')
    x=signal.filtfilt(b,a,raw,axis=1)
    x_dwt=np.vstack([dwt_denoise_1d(ch) for ch in x])
    xx=x_dwt.T
    m=xx.mean(axis=0,keepdims=True);s=xx.std(axis=0,keepdims=True)+1e-8
    z=(xx-m)/s
    ica=FastICA(n_components=N_CHANNELS,random_state=seed,max_iter=1000,tol=1e-4,whiten='unit-variance')
    try:component=ica.fit_transform(z);mix=ica.mixing_
    except Exception as exc:
        import warnings;warnings.warn(f'ICA failure, fallback to DWT only: {exc}',RuntimeWarning)
        return x_dwt
    indicators=np.array([[abs(kurtosis(component[:,i],fisher=True)),
                           high_frequency_ratio(component[:,i],fs),
                           np.max(np.abs(component[:,i]))] for i in range(component.shape[1])])
    indicators=np.nan_to_num(indicators)
    zz=(indicators-indicators.mean(axis=0))/(indicators.std(axis=0)+1e-8)
    score=zz.sum(axis=1)
    drop=np.where(score>score.mean()+2*score.std())[0]
    if len(drop)>2:drop=drop[np.argsort(score[drop])[-2:]]
    component[:,drop]=0
    return ((component@mix.T)*s+m).T

def segment_recording(x,fs=FS):
    samples=int(WINDOW_SECONDS*fs);step=int(samples*(1-OVERLAP))
    win=[x[:,j:j+samples] for j in range(0,x.shape[1]-samples+1,step)]
    return np.asarray(win,dtype=np.float32) if win else np.empty((0,N_CHANNELS,samples),dtype=np.float32)
