"""Metrics, in-sample training threshold, and within-subject majority vote."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             matthews_corrcoef, confusion_matrix)


def choose_train_threshold(y_train, p_train, metric='balanced_accuracy'):
    """Uses only DEVELOPMENT predictions. This is in-sample selection,
    NOT nested CV and may remain optimistic; never use held-out test labels.
    """
    y=np.asarray(y_train,dtype=int)
    p=np.asarray(p_train,dtype=float)
    from sklearn.metrics import roc_curve
    roc_thr=roc_curve(y,p)[2]
    candidates=np.unique(np.r_[roc_thr[np.isfinite(roc_thr)],
           [.001,.002,.005,.01,.02,.03,.04,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.97,.98,.99,.995,.999]])
    best_thr,best_score=.5,-np.inf
    for t in candidates:
        yp=(p>=t).astype(int)
        score=balanced_accuracy_score(y,yp) if metric=='balanced_accuracy' else f1_score(y,yp,zero_division=0)
        if score>best_score: best_thr,best_score=float(t),float(score)
    return best_thr


def metrics(y_true,score,pred=None,threshold=.5):
    y=np.asarray(y_true,dtype=int); p=np.asarray(score,dtype=float)
    pr=(p>=threshold).astype(int) if pred is None else np.asarray(pred,dtype=int)
    tn,fp,fn,tp=confusion_matrix(y,pr,labels=[0,1]).ravel()
    result={'n':len(y),'accuracy':accuracy_score(y,pr),'balanced_accuracy':balanced_accuracy_score(y,pr),
            'precision':precision_score(y,pr,zero_division=0),
            'sensitivity':recall_score(y,pr,zero_division=0),
            'specificity':float(tn/(tn+fp)) if tn+fp else float('nan'),
            'f1':f1_score(y,pr,zero_division=0),
            'mcc':matthews_corrcoef(y,pr),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
    result['auc']=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan')
    return result


def subject_vote(oof, method):
    res=[]
    for sid,g in oof.groupby('subject_id',sort=True):
        if g.y_true.nunique()!=1 or g.fold.nunique()!=1:
            raise ValueError(f'Subject {sid} has inconsistent labels/folds')
        votes=int(g['y_pred'].sum()); count=len(g)
        res.append(dict(method=method,subject_id=sid,fold=int(g.fold.iloc[0]),y_true=int(g.y_true.iloc[0]),
                        n_windows=count,asd_votes=votes,td_votes=count-votes,
                        y_pred=int(votes/count>=.5),p_asd_mean=float(g.p_asd.mean())))
    return pd.DataFrame(res)


def export_evaluation(oof,out,method):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    oof=oof.sort_values(['fold','subject_id','filepath','window_index']).reset_index(drop=True)
    if oof.duplicated(['filepath','window_index']).any():
        raise ValueError('Duplicate out-of-fold record keys')
    oof.to_csv(out/f'{method}_oof_predictions.csv',index=False)
    subject=subject_vote(oof,method)
    subject.to_csv(out/f'{method}_subject_predictions.csv',index=False)
    met=metrics(oof.y_true, oof.p_asd, oof.y_pred)
    pd.DataFrame([dict(method=method,level='pooled_window',**met)]).to_csv(out/f'{method}_metrics.csv',index=False)
    fold_rows=[dict(fold=int(f),**metrics(g.y_true,g.p_asd,g.y_pred)) for f,g in oof.groupby('fold')]
    pd.DataFrame(fold_rows).to_csv(out/f'{method}_fold_metrics.csv',index=False)
    sm=metrics(subject.y_true, subject.p_asd_mean,subject.y_pred)
    pd.DataFrame([dict(method=method,level='subject',**sm)]).to_csv(out/f'{method}_subject_metrics.csv',index=False)
    return met
