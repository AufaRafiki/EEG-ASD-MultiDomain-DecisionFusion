"""Read-only quality checks for a subject-wise output folder."""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from evaluate import metrics

def verify(result_dir,expected_windows=4262):
    root=Path(result_dir)
    mapping=pd.read_csv(root/'subject_fold_map.csv',dtype={'subject_id':str})
    if len(mapping)!=16 or mapping.subject_id.nunique()!=16:raise AssertionError('Expected 16 unique subjects')
    if mapping.groupby(['fold','y']).size().ne(1).any():raise AssertionError('Each fold must hold 1 ASD + 1 TD')
    frames=[]
    for branch in ['psd','cwt','gcn']:
        f=root/f'{branch}_oof_predictions.csv'
        if not f.is_file():continue
        df=pd.read_csv(f,dtype={'subject_id':str})
        if len(df)!=expected_windows:raise AssertionError(f'{branch} expected {expected_windows}, got {len(df)}')
        if df.duplicated(['filepath','window_index']).any():raise AssertionError(f'{branch} duplicate windows')
        source=df[['subject_id','fold']].drop_duplicates()
        got=source.merge(mapping[['subject_id','fold']],on=['subject_id','fold'],how='outer',indicator=True)
        if not got._merge.eq('both').all():raise AssertionError(f'{branch} OOF fold discrepancy')
        for fold,g in df.groupby('fold'):
            hold=set(mapping.loc[mapping.fold==fold,'subject_id'])
            if set(g.subject_id)!=hold:raise AssertionError(f'{branch} fold {fold} wrong held-out subjects')
        print(branch,metrics(df.y_true,df.p_asd,df.y_pred))
        frames.append((branch,df))
    if not frames:raise ValueError('No OOF prediction files in directory')
    a=frames[0][1]
    for branch,b in frames[1:]:
        keys=['filepath','window_index']
        ab=a[keys+['subject_id','fold','y_true']].merge(b[keys+['subject_id','fold','y_true']],on=keys,how='outer',indicator=True,suffixes=('_a','_b'))
        if not ab._merge.eq('both').all():raise AssertionError(f'{branch} row alignment mismatch')
        for col in ('subject_id','fold','y_true'):
            if not (ab[col+'_a'].astype(str)==ab[col+'_b'].astype(str)).all():
                raise AssertionError(f'{branch} disagreement on {col}')
    print('PASSED: no repeated samples across fold outputs; held-out subjects disjoint; aligned OOF.')
    return True

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',required=True);p.add_argument('--expected-windows',type=int,default=4262)
    args=p.parse_args();verify(args.results,args.expected_windows)
