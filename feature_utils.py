"""Shared feature transforms, model-input conditioning, adjacency creation."""
import numpy as np

def topk_adjacency(abs_corr,k=4):
    """Symmetric weighted top-k, preserving unmasked full matrix separately.
    Assumption for reconstructed raw-input route; historical cache builder missing.
    """
    c=np.nan_to_num(np.abs(np.asarray(abs_corr,dtype=np.float32)))
    if c.ndim==2:c=c[None]
    n=c.shape[1]
    if c.shape[1:]!=(n,n) or not 1<=k<n:raise ValueError('Invalid correlation shape or k')
    weights=c.copy();idx=np.arange(n);weights[:,idx,idx]=-np.inf
    keep=np.zeros_like(weights,dtype=bool)
    sel=np.argpartition(weights,-k,axis=-1)[:,:,-k:]
    np.put_along_axis(keep,sel,True,axis=-1)
    keep=np.logical_or(keep,keep.transpose(0,2,1))
    result=np.where(keep,c,0);result[:,idx,idx]=0
    return result.astype(np.float32)


def normalized_adjacency(A):
    a=np.abs(np.asarray(A,dtype=np.float32)).copy()
    if a.ndim!=3:raise ValueError('Expected N x C x C adjacency')
    a=(a+a.transpose(0,2,1))/2
    i=np.arange(a.shape[1]);a[:,i,i]=1.
    deg=a.sum(axis=-1)
    inv=1/np.sqrt(deg+1e-8)
    return (a*inv[:,:,None]*inv[:,None,:]).astype(np.float32)


def edges_from_matrix(A):
    a=np.asarray(A)
    return a[:,*np.triu_indices(a.shape[1],k=1)].astype(np.float32)


def matrix_from_edges(edges,n_channels=16):
    e=np.asarray(edges,dtype=np.float32)
    if e.ndim!=2 or e.shape[1]!=n_channels*(n_channels-1)//2:
        raise ValueError('Expected 120 upper-triangular edge features')
    a=np.zeros((len(e),n_channels,n_channels),dtype=np.float32)
    i,j=np.triu_indices(n_channels,1)
    a[:,i,j]=e;a[:,j,i]=e
    return a


def gcn_inputs_from_raw(node,adj,edge,tr,te):
    """Train-only normalization, matching the notebook's axis definitions."""
    from numpy import mean,std
    nm=node[tr].mean(axis=(0,1),keepdims=True)
    ns=node[tr].std(axis=(0,1),keepdims=True)+1e-8
    em=edge[tr].mean(axis=0,keepdims=True)
    es=edge[tr].std(axis=0,keepdims=True)+1e-8
    node_scaled=((node-nm)/ns).astype('float32')
    edge_scaled=((edge-em)/es).astype('float32')
    adj_scaled=normalized_adjacency(adj)
    stats=dict(node_mean=nm,node_std=ns,edge_mean=em,edge_std=es)
    return [node_scaled,adj_scaled,edge_scaled],stats
