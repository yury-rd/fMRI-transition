import numpy as np

def alpha_candidates(y,cut):
    z=y[:cut];sd=z.std(axis=0,ddof=0)
    if not np.all(np.isfinite(sd)) or np.any(sd<=0):raise ValueError('invalid training channel scale')
    centered=(z-z.mean(axis=0))/sd
    lag=(centered[1:]*centered[:-1]).sum(0)/np.maximum((centered[:-1]**2).sum(0),1e-12)
    d=y.shape[1]
    return {'none':np.zeros(d),'scalar_035':np.full(d,.35),
            'scalar_065':np.full(d,.65),'lag1':np.clip(lag,.1,.8)}
