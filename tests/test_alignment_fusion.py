import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from align_caches import align
from fusion import run


def test_cache_alignment_and_fusion(tmp_path):
    rng=np.random.default_rng(7)
    names=np.array([f'subject{i}' for i in range(16)])
    label=np.array([1]*8+[0]*8)
    n=48
    group=np.repeat(names,3)
    y=np.repeat(label,3)
    files=np.array([f'{name}.dat' for name in group])
    idx=np.tile(np.arange(3),16)
    perm=rng.permutation(n)
    common=dict(y=y,groups=group,filepaths=files,win_index=idx)
    psd=tmp_path/'psd.npz'
    cwt=tmp_path/'cwt.npz'
    gcn=tmp_path/'gcn.npz'
    np.savez_compressed(psd,**common,X_psd=np.zeros((n,80),np.float32))
    np.savez_compressed(cwt,**{k:v[perm] for k,v in common.items()},X_cwt=np.ones((n,64,128,16),np.float32)[perm])
    np.savez_compressed(gcn,**common,X_nodes=np.zeros((n,16,5),np.float32),
                        A_graph=np.eye(16,dtype=np.float32)[None].repeat(n,axis=0),
                        X_edges=np.ones((n,120),np.float32))
    output=tmp_path/'aligned.npz'
    result=align(psd,cwt,gcn,output)
    assert output.exists()
    assert np.array_equal(result['y'],y)
    assert result['X_cwt'].shape==(48,64,128,16)
    # Distinct OOF records from 3 branches with same held-out keys.
    for b in ['psd','cwt','gcn']:
        df=pd.DataFrame(dict(filepath=files,window_index=idx,subject_id=group,
           fold=np.repeat(np.arange(1,9),6),y_true=y,p_asd=np.where(y==1,.9,.1),threshold=.5))
        # fold map here is arbitrary; fusion only ensures branches agree on fold.
        df.to_csv(tmp_path/f'{b}_oof_predictions.csv',index=False)
    res=run(tmp_path)
    assert len(res['soft_mean'])==n
    assert (tmp_path/'soft_mean_subject_predictions.csv').exists()


def test_fail_closed_on_mismatched_sample_key(tmp_path):
    n=2
    m=dict(y=np.array([0,1]),groups=np.array(['T','A']),filepaths=np.array(['x.dat','y.dat']),
           win_index=np.array([0,0]))
    p=tmp_path/'p.npz';c=tmp_path/'c.npz';g=tmp_path/'g.npz'
    np.savez(p,**m,X_psd=np.zeros((n,80)))
    mm=dict(m);mm['filepaths']=np.array(['x.dat','bad.dat'])
    np.savez(c,**mm,X_cwt=np.zeros((n,64,128,16)))
    np.savez(g,**m,X_nodes=np.zeros((n,16,5)),X_edges=np.zeros((n,120)))
    with pytest.raises(ValueError,match='identities differ'):
        align(p,c,g,tmp_path/'aligned.npz')
