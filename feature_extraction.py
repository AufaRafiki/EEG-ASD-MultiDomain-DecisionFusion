"""PSD port, reconstructed CWT and top-k connectivity features.

IMPORTANT: Uploaded notebook does NOT include original generation of CWT and
GCN .npz caches. CWT definition here is a transparent RECONSTRUCTION, not
confirmed identical to the caches used for historical experiments.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import trapezoid
from config import FS,N_CHANNELS,LOWCUT,HIGHCUT,BANDS
from preprocessing import load_bci2000_dat,preprocess_recording,segment_recording
from feature_utils import topk_adjacency,edges_from_matrix

def psd_matrix(w,fs=FS):
    f,p=signal.welch(w,fs=fs,nperseg=min(w.shape[1],fs*2),axis=1)
    total=(f>=LOWCUT)&(f<=HIGHCUT)
    total_power=trapezoid(p[:,total],f[total],axis=1)+1e-12
    a=[]
    for lo,hi in BANDS.values():
        ix=(f>=lo)&(f<hi)
        a.append(trapezoid(p[:,ix],f[ix],axis=1)/total_power if ix.any() else np.zeros(w.shape[0]))
    return np.log1p(np.stack(a,axis=1)).astype(np.float32)

def cwt_scalogram(w,fs=FS,size=(64,128)):
    """Explicit, nonhistorical CWT recipe: Morlet, |CWT|, resample time to 128."""
    import pywt
    from scipy.signal import resample
    freqs=np.geomspace(1.,40.,num=size[0])
    scales=pywt.central_frequency('morl')*fs/freqs
    out=np.empty((*size,w.shape[0]),dtype=np.float32)
    for ch in range(w.shape[0]):
        coefs,_=pywt.cwt(w[ch],scales,'morl',sampling_period=1/fs)
        out[:,:,ch]=resample(np.abs(coefs),size[1],axis=1).astype(np.float32)
    return out

def pairwise_correlation(w):
    c=np.corrcoef(w)
    c=np.nan_to_num(c,nan=0.,posinf=0.,neginf=0.)
    c=np.abs(c).astype(np.float32)
    np.fill_diagonal(c,1.)
    return c

def generate(data_root,manifest_file,out_path, k=4):
    manifest=pd.read_csv(manifest_file,dtype={'subject_id':str})
    needed={'filepath','subject_id','y'}
    if not needed.issubset(manifest.columns):raise ValueError(f'Manifest needs {needed}')
    if manifest.subject_id.isna().any() or manifest.subject_id.astype(str).str.strip().eq('').any():
        raise ValueError('Every .dat must have a verified subject ID')
    if manifest.groupby('subject_id')['y'].nunique().gt(1).any():
        raise ValueError('Subject class contradiction')
    if not set(manifest.y.unique())<={0,1}:raise ValueError('Use TD=0 ASD=1')
    psd=[]; cwt=[]; node=[]; adj=[]; edges=[];ys=[];groups=[];files=[];idxs=[]
    root=Path(data_root).resolve()
    for _,row in manifest.iterrows():
        path=(root/str(row.filepath)).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise FileNotFoundError(path)
        raw=load_bci2000_dat(path)
        windows=segment_recording(preprocess_recording(raw))
        for j,w in enumerate(windows):
            n=psd_matrix(w)
            corr=pairwise_correlation(w)
            psd.append(n.reshape(-1));node.append(n)
            cwt.append(cwt_scalogram(w));adj.append(topk_adjacency(corr,k=k)[0])
            edges.append(edges_from_matrix(corr[None])[0])
            ys.append(int(row.y));groups.append(str(row.subject_id))
            files.append(str(row.filepath));idxs.append(j)
        print(f'{row.subject_id}: {path.name}: {len(windows)} windows')
    if not ys:raise ValueError('No EEG windows generated')
    result={'X_psd':np.array(psd,np.float32),'X_cwt':np.array(cwt,np.float32),
       'X_nodes':np.array(node,np.float32),'A_graph':np.array(adj,np.float32),
       'X_edges':np.array(edges,np.float32),'y':np.array(ys,int),
       'groups':np.array(groups,str),'filepaths':np.array(files,str),
       'win_index':np.array(idxs,int)}
    target=Path(out_path);target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,**result)
    print('Saved:',target,'samples:',len(ys))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',required=True);ap.add_argument('--manifest',required=True)
    ap.add_argument('--output',default='outputs/features_reconstructed.npz')
    ap.add_argument('--k',type=int,default=4)
    args=ap.parse_args();generate(args.data_root,args.manifest,args.output,args.k)
