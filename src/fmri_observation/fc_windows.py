"""Declared measurement operators; prepared window-by-edge input, no imaging pipeline."""
import numpy as np

def window_indices(count, mode):
    if count < 8: raise ValueError("Eight windows required")
    if mode == "full": return np.arange(count)
    if mode == "first8": return np.arange(8)
    if mode == "middle8": return np.arange((count-8)//2, (count-8)//2+8)
    if mode == "last8": return np.arange(count-8, count)
    if mode == "spread8": return np.arange(8)*(count-1)//7
    raise ValueError(mode)

def dispersion(edges, mode="full"):
    x=np.asarray(edges,dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all(): raise ValueError("Finite window-by-edge array required")
    return x[window_indices(len(x),mode)].std(axis=0,ddof=1)

def order_features(edges, lag=3):
    x=np.asarray(edges,dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or not 1 <= lag < len(x): raise ValueError("Invalid trajectory or lag")
    sd=x.std(axis=0)
    if np.any(sd <= 1e-12): raise ValueError("Constant edge is unsupported")
    w=len(x);z=(x-x.mean(axis=0))/sd
    persistence=1-(w-1)/w*np.mean((z[lag:]-z[:-lag])**2,axis=0)/2
    cross=z[:-lag].T@z[lag:]/(w-lag)
    return persistence,(cross-cross.T)[np.triu_indices(x.shape[1],1)]
