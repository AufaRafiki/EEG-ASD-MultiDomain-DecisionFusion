"""Subject-disjoint training runner for 3 models; no test-subject leakage.

Typical:
 python train.py --features outputs/aligned_features.npz --out outputs/experiment
 python train.py --features outputs/aligned_features.npz --fold-map configs/subject_folds.csv ...
"""
from pathlib import Path
import argparse, json, gc, sys, platform
import numpy as np
import pandas as pd
from config import Settings
from splits import make_subject_folds,save_fold_evidence
from feature_utils import gcn_inputs_from_raw
from evaluate import choose_train_threshold, export_evaluation


def load_features(p):
    with np.load(p,allow_pickle=False) as d:
        required={'X_psd','X_cwt','X_nodes','A_graph','X_edges','y','groups','filepaths','win_index'}
        if not required <= set(d.files):raise ValueError('Missing features: '+str(sorted(required-set(d.files))))
        a={k:d[k] for k in required}
    n=len(a['y'])
    for k,v in a.items():
        if len(v)!=n:raise ValueError(f'{k}: samples not aligned')
    if not np.isfinite(a['X_psd']).all() or not np.isfinite(a['X_cwt']).all() or not np.isfinite(a['X_nodes']).all() or not np.isfinite(a['A_graph']).all():
        raise ValueError('Input contains NaN/Inf; must correct feature extraction, no silent imputation')
    if a['X_psd'].ndim!=2 or a['X_psd'].shape[1]!=80:raise ValueError('Expected PSD 80 features')
    if a['X_cwt'].shape[1:]!=(64,128,16):raise ValueError('Expected CWT 64x128x16')
    if a['X_nodes'].shape[1:]!=(16,5):raise ValueError('Expected node input 16x5')
    if a['A_graph'].shape[1:]!=(16,16):raise ValueError('Expected adjacency 16x16')
    if a['X_edges'].shape[1:]!=(120,):raise ValueError('Expected 120 edge features')
    if not set(np.unique(a['y']))=={0,1}:raise ValueError('Expected both classes 0=TD,1=ASD')
    if pd.DataFrame({'fp':a['filepaths'].astype(str),'wi':a['win_index']}).duplicated().any():
        raise ValueError('Duplicate window keys')
    return a


def _psd_inputs(a,train_idx):
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler().fit(a['X_psd'][train_idx])
    return scaler.transform(a['X_psd']).astype(np.float32),dict(mean=scaler.mean_,scale=scaler.scale_)


def _cwt_inputs(a,train_idx):
    x=a['X_cwt']
    # Notebook: mean/std computed over all TRAIN windows/time/freq, per channel
    mean=x[train_idx].mean(axis=(0,1,2),keepdims=True)
    std=x[train_idx].std(axis=(0,1,2),keepdims=True)+1e-8
    return ((x-mean)/std).astype(np.float32),dict(mean=mean,std=std)


def run(features, out_dir,fold_file=None,settings=None,expected_windows=0,branches=None):
    cfg=settings or Settings()
    branches=branches or ['psd','cwt','gcn']
    invalid=set(branches)-{'psd','cwt','gcn'}
    if invalid:raise ValueError('Unknown branch '+str(invalid))
    a=load_features(features)
    if expected_windows and len(a['y'])!=expected_windows:
        raise ValueError(f'Expected {expected_windows} windows but got {len(a["y"])}')
    y=a['y'].astype(int);groups=a['groups'].astype(str)
    folds,fmap=make_subject_folds(groups,y,cfg.n_folds,fold_file)
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    save_fold_evidence(groups,y,folds,fmap,out)
    (out/'run_config.json').write_text(json.dumps(dict(settings=cfg.as_dict(),features_path=str(features),
      fold_map_path=str(fold_file),n_windows=len(y),n_subjects=len(np.unique(groups)),
      numpy_version=np.__version__,python=platform.python_version(),branches=branches,
      audit_status='reconstructed_subject_disjoint_run_pending_manuscript_reconciliation'),indent=2))
    import tensorflow as tf
    from models import psd_mlp,cwt_cnn,edge_aware_gcn
    all_results={}
    for branch in branches:
        rows=[]
        for fold in range(1,cfg.n_folds+1):
            tr=np.flatnonzero(folds!=fold); te=np.flatnonzero(folds==fold)
            if not set(groups[tr]).isdisjoint(groups[te]):raise RuntimeError('SUBJECT LEAKAGE')
            if len(np.unique(y[te]))!=2:raise ValueError('Each test fold must contain ASD and TD')
            tf.keras.backend.clear_session();gc.collect()
            tf.keras.utils.set_random_seed(cfg.seed+fold)
            d=out/branch/f'fold_{fold:02d}';d.mkdir(parents=True,exist_ok=True)
            if branch=='psd':
                inputs,stats=_psd_inputs(a,tr)
                model=psd_mlp(80,cfg.psd_lr)
                epochs,batch=cfg.psd_epochs,cfg.psd_batch
                def batch_data(ii):return inputs[ii]
                threshold=.5
            elif branch=='cwt':
                inputs,stats=_cwt_inputs(a,tr)
                model=cwt_cnn(tuple(a['X_cwt'].shape[1:]),cfg.cwt_lr)
                epochs,batch=cfg.cwt_epochs,cfg.cwt_batch
                def batch_data(ii):return inputs[ii]
                threshold=None
            else:
                inp,stats=gcn_inputs_from_raw(a['X_nodes'],a['A_graph'],a['X_edges'],tr,te)
                model=edge_aware_gcn(tuple(a['X_nodes'].shape[1:]),tuple(a['A_graph'].shape[1:]),
                     tuple(a['X_edges'].shape[1:]),cfg.gcn_lr,cfg.weight_decay,cfg.dropout_gcn)
                epochs,batch=cfg.gcn_epochs,cfg.gcn_batch
                def batch_data(ii):return [x[ii] for x in inp]
                threshold=None
            np.savez_compressed(d/'train_only_normalization.npz',**stats)
            # Avoid file overwrites and discourage early stopping driven by test fold.
            class_count=np.bincount(y[tr],minlength=2)
            weights={0:len(tr)/(2*class_count[0]),1:len(tr)/(2*class_count[1])}
            history=model.fit(batch_data(tr),y[tr],epochs=epochs,batch_size=batch,
                verbose=2,class_weight=weights,callbacks=[tf.keras.callbacks.TerminateOnNaN(),
                   tf.keras.callbacks.CSVLogger(str(d/'training_log.csv'))])
            model.save(d/'model.keras')
            if threshold is None:
                # Matches notebook IN-SAMPLE threshold selection (limitation explicitly noted).
                p_tr=model.predict(batch_data(tr),batch_size=batch,verbose=0).reshape(-1)
                threshold=choose_train_threshold(y[tr],p_tr,cfg.threshold_metric)
            p_te=model.predict(batch_data(te),batch_size=batch,verbose=0).reshape(-1)
            (d/'threshold.json').write_text(json.dumps(dict(threshold=float(threshold),
               selection='fixed .5' if branch=='psd' else 'development in-sample predictions; no held-out labels',
               test_subjects=sorted(np.unique(groups[te]).tolist()),n_train_windows=len(tr),
               n_test_windows=len(te),train_subjects=sorted(np.unique(groups[tr]).tolist()),
               params=int(model.count_params())),indent=2))
            if set(groups[tr]) & set(groups[te]):raise RuntimeError('Leakage before write')
            frame=pd.DataFrame(dict(fold=fold,filepath=a['filepaths'][te].astype(str),
                 window_index=a['win_index'][te].astype(int),subject_id=groups[te],
                 y_true=y[te],p_asd=p_te,threshold=float(threshold),
                 y_pred=(p_te>=threshold).astype(int)))
            frame.to_csv(d/'test_window_predictions.csv',index=False)
            rows.append(frame)
            del model,history
            if branch == 'gcn':
                del inp
            else:
                del inputs
            gc.collect()
        all_oof=pd.concat(rows,ignore_index=True)
        if len(all_oof)!=len(y):raise RuntimeError('OOF count mismatch')
        met=export_evaluation(all_oof,out,branch)
        all_results[branch]=met
        print(f'{branch} subject-wise OOF acc={met["accuracy"]:.4f}, subjects={len(np.unique(groups))}')
    return all_results

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features',required=True)
    ap.add_argument('--out',default='outputs/subject_wise')
    ap.add_argument('--fold-map',help='CSV subject_id,fold; recommended to reproduce specific test assignments')
    ap.add_argument('--expected-windows',type=int,default=4262)
    ap.add_argument('--branches',nargs='+',choices=['psd','cwt','gcn'],default=['psd','cwt','gcn'])
    x=ap.parse_args();run(x.features,x.out,x.fold_map,expected_windows=x.expected_windows,branches=x.branches)
