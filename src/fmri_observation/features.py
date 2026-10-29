"""Static and sliding-window FC on an already prepared seven-network field."""
import numpy as np
from .connectivity import corr,disp,ei


def extract(timeseries, tr_seconds):
    x=np.asarray(timeseries,dtype=float)
    if x.ndim!=2 or x.shape[1]!=7 or not np.isfinite(x).all():
        raise ValueError('Expected a finite samples-by-seven network field')
    if not np.isfinite(tr_seconds) or tr_seconds<=0:
        raise ValueError('Positive TR in seconds required')
    width=max(8,round(60/tr_seconds));stride=max(1,round(30/tr_seconds))
    starts=list(range(0,len(x)-width+1,stride))
    if len(starts)<2:raise ValueError('Dispersion requires at least two full windows')
    return {'static':corr(x)[ei],'dispersion':disp(x,width,starts),
            'window_samples':width,'stride_samples':stride,'starts':starts}
