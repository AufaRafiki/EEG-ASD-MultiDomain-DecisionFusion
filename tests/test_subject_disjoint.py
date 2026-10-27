import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from splits import make_subject_folds
from feature_utils import topk_adjacency,normalized_adjacency,edges_from_matrix
from fusion import centered
from evaluate import metrics,subject_vote


def dataset():
    subjects=np.array([f'ASD-{i:02d}' for i in range(1,9)]+[f'TD-{i:02d}' for i in range(1,9)])
    ids=np.repeat(subjects,4)
    y=np.repeat([1]*8+[0]*8,4)
    return ids,np.array(y)


def test_eight_subject_disjoint_folds():
    groups,y=dataset()
    f,summary=make_subject_folds(groups,y)
    assert len(summary)==16
    for fold in range(1,9):
        assert len(set(groups[f==fold]))==2
        assert not (set(groups[f==fold]) & set(groups[f!=fold]))
        assert set(y[f==fold])=={0,1}


def test_reject_subject_duplicate_labels():
    groups,y=dataset();y[0]=0
    with pytest.raises(ValueError):make_subject_folds(groups,y)


def test_reject_subject_invalid_count():
    groups,y=dataset()
    with pytest.raises(ValueError):make_subject_folds(groups[:-4],y[:-4])


def test_no_external_fold_subject():
    groups,y=dataset()
    f,summary=make_subject_folds(groups,y)
    bad=summary[['subject_id','fold']].copy();bad.loc[0,'subject_id']='UNKNOWN'
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'f.csv';bad.to_csv(p,index=False)
        with pytest.raises(ValueError):make_subject_folds(groups,y,fold_file=p)


def test_topk_graph_and_scores():
    rng=np.random.default_rng(42)
    c=np.abs(rng.random((2,16,16)));c=(c+c.transpose(0,2,1))/2
    a=topk_adjacency(c,k=4)
    assert np.allclose(a,a.transpose(0,2,1))
    assert np.allclose(np.diagonal(a,axis1=1,axis2=2),0)
    assert np.isfinite(normalized_adjacency(a)).all()
    assert edges_from_matrix(c).shape==(2,120)
    assert np.allclose(centered(np.array([.2,.6,.8]),np.array([.2,.6,.8])),.5)


def test_subject_vote():
    groups,y=dataset()
    folds,_=make_subject_folds(groups,y)
    predictions=np.array(y)
    df=pd.DataFrame({'subject_id':groups,'fold':folds,'y_true':y,
                     'y_pred':predictions,'p_asd':predictions.astype(float)})
    res=subject_vote(df,'demo')
    assert len(res)==16
    assert (res.y_true==res.y_pred).all()
