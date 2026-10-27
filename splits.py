"""Strict subject-disjoint 8-fold construction. No window-wise CV anywhere."""
from pathlib import Path
import numpy as np
import pandas as pd


def _subjects(groups, y):
    df = pd.DataFrame({'subject_id': np.asarray(groups).astype(str), 'y': np.asarray(y, int)})
    if df['subject_id'].str.strip().eq('').any():
        raise ValueError('Blank subject_id: cannot guarantee subject independence')
    if df.groupby('subject_id')['y'].nunique().gt(1).any():
        raise ValueError('A subject appears with conflicting ASD/TD labels')
    return df.drop_duplicates('subject_id').sort_values('subject_id').reset_index(drop=True)


def make_subject_folds(groups, y, n_folds=8, fold_file=None):
    """Return row fold IDs, and one row/subject mapping.

    - If fold_file is provided it must have subject_id,fold and match all subjects.
    - Otherwise sort IDs *within classes*, pair one ASD with one TD per fold.
    - For the manuscript configuration: 8 ASD + 8 TD => 8 folds.
    """
    sub = _subjects(groups, y)
    if fold_file:
        given = pd.read_csv(fold_file, dtype={'subject_id': str})
        if not {'subject_id','fold'} <= set(given.columns):
            raise ValueError('fold_file must contain subject_id,fold')
        if given.subject_id.duplicated().any():
            raise ValueError('Duplicate subject in fold_file')
        if set(given.subject_id) != set(sub.subject_id):
            raise ValueError('Fold file IDs do not match feature subject IDs exactly')
        fmap = sub.merge(given[['subject_id','fold']], on='subject_id', validate='one_to_one')
        fmap['fold'] = fmap['fold'].astype(int)
    else:
        pos = sorted(sub.loc[sub.y == 1, 'subject_id'])
        neg = sorted(sub.loc[sub.y == 0, 'subject_id'])
        if len(pos) != n_folds or len(neg) != n_folds:
            raise ValueError(f'Auto 1 ASD + 1 TD per fold needs {n_folds} unique subjects of each class, got ASD={len(pos)}, TD={len(neg)}. Provide valid fold_file for different design.')
        ids = {sid: i+1 for i,(a,b) in enumerate(zip(pos,neg)) for sid in (a,b)}
        fmap = sub.assign(fold=sub.subject_id.map(ids))
    if set(fmap.fold) != set(range(1, n_folds+1)):
        raise ValueError('Folds must be consecutive IDs 1..n_folds')
    counts = fmap.groupby(['fold','y']).size().unstack(fill_value=0)
    for f in range(1, n_folds+1):
        if counts.loc[f,0] != 1 or counts.loc[f,1] != 1:
            raise ValueError(f'Fold {f}: expected exactly 1 ASD and 1 TD test subject')
    row_folds = pd.Series(np.asarray(groups).astype(str)).map(dict(zip(fmap.subject_id, fmap.fold)))
    if row_folds.isna().any():
        raise ValueError('Unassigned samples')
    row_folds = row_folds.to_numpy(dtype=int)
    for f in range(1,n_folds+1):
        tr,te = row_folds != f, row_folds == f
        if set(np.asarray(groups)[tr]).intersection(set(np.asarray(groups)[te])):
            raise AssertionError('SUBJECT LEAKAGE')
        if set(np.unique(np.asarray(y)[tr])) != {0,1}:
            raise ValueError(f'Fold {f}: training needs two classes')
    return row_folds, fmap.sort_values(['fold','y'], ascending=[True,False]).reset_index(drop=True)


def save_fold_evidence(groups, y, row_folds, fmap, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    counts = (pd.DataFrame({'subject_id':np.asarray(groups).astype(str),'y':y,'fold':row_folds})
              .groupby(['fold','subject_id','y']).size().rename('test_windows').reset_index())
    counts.to_csv(out/'exact_subject_wise_fold_assignments.csv',index=False)
    fmap.to_csv(out/'subject_fold_map.csv',index=False)
