"""Align true window-wise OOF outputs and compute three no-fit fusion methods.

Uses threshold-aligned scores from notebook (epsilon 1e-7, gamma=1).
Reports exploratory post-hoc fusion comparisons, not confirmatory superiority.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from evaluate import export_evaluation

EPS=1e-7

def logit(p):
    p=np.clip(np.asarray(p,dtype=float),EPS,1-EPS)
    return np.log(p/(1-p))

def centered(p,t):
    v=np.clip(logit(p)-logit(t),-50,50)
    return 1/(1+np.exp(-v))

def run(root,out=None):
    root=Path(root);out=Path(out or root)
    frames=[]; key=['filepath','window_index']
    for branch in ('psd','cwt','gcn'):
        df=pd.read_csv(root/f'{branch}_oof_predictions.csv')
        required=set(key+['fold','subject_id','y_true','p_asd','threshold'])
        if not required<=set(df):raise ValueError(f'{branch}: missing {required-set(df)}')
        if df.duplicated(key).any():raise ValueError(f'{branch} duplicate window key')
        frames.append(df)
    base=frames[0].copy()
    if len(set(base.subject_id))!=16:print('Warning: dataset does not have 16 subjects; inspect requested cohort')
    for i,branch in enumerate(('psd','cwt','gcn')):
        df=frames[i]
        lookup=base[key].merge(df,how='left',on=key,validate='one_to_one',indicator=True)
        if len(df)!=len(base) or not lookup._merge.eq('both').all():
            raise ValueError(f'Window identity mismatch for {branch}')
        for c in ('fold','subject_id','y_true'):
            if not np.array_equal(base[c].astype(str),lookup[c].astype(str)):
                raise ValueError(f'Inconsistent {c} in {branch}')
        base[f'p_{branch}']=lookup.p_asd.to_numpy()
        base[f't_{branch}']=lookup.threshold.to_numpy()
        base[f'q_{branch}']=centered(base[f'p_{branch}'],base[f't_{branch}'])
    q=np.column_stack([base[f'q_{b}'] for b in ('psd','cwt','gcn')])
    scores={}
    scores['soft_mean']=np.mean(q,axis=1)
    scores['majority_vote']=np.mean(q>=.5,axis=1)
    weight=np.power(2*np.abs(q-.5),1.)
    scores['margin_weighted']=np.divide((weight*q).sum(axis=1),weight.sum(axis=1),
         out=scores['soft_mean'].copy(),where=weight.sum(axis=1)>EPS)
    for name,score in scores.items():
        oof=base[['filepath','window_index','fold','subject_id','y_true']].copy()
        oof['p_asd']=score;oof['threshold']=.5;oof['y_pred']=(score>=.5).astype(int)
        result=export_evaluation(oof,out,name)
        print(f'{name}: {result["accuracy"]:.5f}; {result["tn"]} TN, {result["fp"]} FP, {result["fn"]} FN, {result["tp"]} TP')
    return scores

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',default='outputs/subject_wise')
    p.add_argument('--output',default=None)
    args=p.parse_args();run(args.results,args.output)
