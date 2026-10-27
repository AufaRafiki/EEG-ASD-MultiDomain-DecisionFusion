"""Build one auditable aligned NPZ from PSD, CWT, GCN feature caches.

Does NOT align by row number or by y alone. Fail closed if sample keys absent.
Notebook's raw CWT/GCN cache-building cells were NOT supplied; original cache
files must be supplied separately for most faithful reconstruction.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from feature_utils import edges_from_matrix, matrix_from_edges

KEYS = ('subject_id','filepath','window_index')

def _get(d,options,required=True):
    for name in options:
        if name in d:return d[name]
    if required:raise KeyError(f'Missing cache keys {options}; actual={list(d)}')
    return None

def _extract(cache,branch):
    with np.load(cache,allow_pickle=True) as d:
        data={k:d[k] for k in d.files}
    y=np.asarray(_get(data,['y','labels','label']),dtype=int).ravel()
    group=np.asarray(_get(data,['groups','subject_ids','subjects','subject_id'])).astype(str).ravel()
    fp=np.asarray(_get(data,['filepaths','filepath','files'])).astype(str).ravel()
    wi=np.asarray(_get(data,['win_index','window_index','window_indices'])).astype(int).ravel()
    # Filepath must come from the underlying recording; never fallback to row index.
    table=pd.DataFrame({'subject_id':group,'filepath':[Path(x).name for x in fp],
                        'window_index':wi,'y':y})
    if not all(len(a)==len(y) for a in (group,fp,wi)):
        raise ValueError(f'{branch}: metadata length differs')
    if table.duplicated(list(KEYS)).any():
        raise ValueError(f'{branch}: duplicated file/window identity; inspect original cache')
    if table.subject_id.str.strip().eq('').any() or table.filepath.str.strip().eq('').any():
        raise ValueError(f'{branch}: subject ID/filepath missing. Cannot align reliably.')
    if branch=='psd':
        values={'X_psd':np.asarray(_get(data,['X_psd']),dtype=np.float32)}
    elif branch=='cwt':
        values={'X_cwt':np.asarray(_get(data,['X_cwt']),dtype=np.float32)}
    else:
        node=np.asarray(_get(data,['X_nodes','X_node','X_nodes_raw','node_features','X_graph_nodes','X_node_features','X_connectivity_nodes']),dtype=np.float32)
        a=_get(data,['A_graph','A','adjacency','adj','X_adj','A_conn','A_connectivity','connectivity_matrix','connectivity_graph'],False)
        e=_get(data,['X_edges','edge_features','edges','X_edge'],False)
        if a is None and e is None:raise ValueError('GCN cache missing adjacency and edges')
        if a is not None:
            a=np.asarray(a,dtype=np.float32)
            if a.ndim==4 and a.shape[-1]==1:a=a[...,0]
            if a.ndim==2 and a.shape==(16,16):a=np.repeat(a[None],len(y),axis=0)
            if a.ndim==2 and a.shape[1]==120:a=matrix_from_edges(a)
            if a.shape!=(len(y),16,16):raise ValueError(f'Adjacency shape unsupported: {a.shape}')
        if e is None:
            e=edges_from_matrix(a)
        else:e=np.asarray(e,dtype=np.float32)
        if a is None:a=matrix_from_edges(e)
        values={'X_nodes':node,'A_graph':a,'X_edges':e}
    for k,x in values.items():
        if x.shape[0]!=len(y):raise ValueError(f'{branch} {k} sample count mismatch')
    return table,values


def _apply_verified_identity_map(table, kind, path):
    """Optional explicit aliases for the same recording/participant across caches.
    CSV: branch,original_subject_id,canonical_subject_id,original_filename,canonical_filename
    """
    if path is None:
        return table
    aliases=pd.read_csv(path,dtype=str).fillna('')
    needed={'branch','original_subject_id','canonical_subject_id','original_filename','canonical_filename'}
    if not needed <= set(aliases.columns):
        raise ValueError(f'Identity-map CSV must include {needed}')
    aliases=aliases.loc[aliases.branch==kind]
    if aliases.duplicated(['original_subject_id','original_filename']).any():
        raise ValueError(f'{kind}: ambiguous alias entries')
    renamed=table.copy()
    mapping={(r.original_subject_id,r.original_filename):(r.canonical_subject_id,r.canonical_filename)
             for r in aliases.itertuples()}
    original=renamed[['subject_id','filepath']].apply(tuple,axis=1)
    target=[mapping.get(x,x) for x in original]
    renamed['subject_id']=[x[0] for x in target]
    renamed['filepath']=[x[1] for x in target]
    if renamed.duplicated(list(KEYS)).any():
        raise ValueError(f'{kind}: aliasing caused duplicated identities')
    return renamed


def align(psd,cwt,gcn,output,identity_map=None):
    sources={'psd':psd,'cwt':cwt,'gcn':gcn};tables={};vals={}
    for kind,path in sources.items():
        tables[kind],vals[kind]=_extract(path,kind)
        tables[kind]=_apply_verified_identity_map(tables[kind],kind,identity_map)
    ref=tables['psd'];key=list(KEYS)
    ref_index=pd.MultiIndex.from_frame(ref[key])
    result={'y':ref.y.to_numpy(dtype=np.int32),
       'groups':ref.subject_id.to_numpy(dtype=str),
       'filepaths':ref.filepath.to_numpy(dtype=str),
       'win_index':ref.window_index.to_numpy(dtype=np.int32)}
    for kind in sources:
        tb=tables[kind]
        if set(map(tuple,tb[key].to_numpy())) != set(map(tuple,ref[key].to_numpy())):
            raise ValueError(f'{kind}: window identities differ from PSD cache. No positional fallback allowed.')
        lookup=pd.MultiIndex.from_frame(tb[key])
        pos=lookup.get_indexer(ref_index)
        if np.any(pos<0):raise AssertionError(f'{kind}: unmatched records')
        if not np.array_equal(tb.y.to_numpy()[pos], ref.y.to_numpy()):
            raise ValueError(f'{kind}: label disagreement after joining keys')
        for name,arr in vals[kind].items():result[name]=arr[pos]
    n=len(result['y'])
    if len(np.unique(result['groups']))<2:raise ValueError('Subject IDs invalid')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**result)
    print(f'Aligned {n} windows across 3 branches => {output}')
    print('Unique subjects:',len(np.unique(result['groups'])))
    print('CLASS counts TD/ASD:',np.bincount(result['y'],minlength=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--psd-cache',required=True)
    p.add_argument('--cwt-cache',required=True)
    p.add_argument('--gcn-cache',required=True)
    p.add_argument('--output',default='outputs/aligned_features.npz')
    p.add_argument('--identity-map',help='Optional VERIFIED per-cache alias CSV; see README')
    a=p.parse_args();align(a.psd_cache,a.cwt_cache,a.gcn_cache,a.output,a.identity_map)
