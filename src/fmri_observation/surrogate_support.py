import numpy as np
from . import surrogates as m

def measurement(x):return {'SampEn':m.sampen_fast(x),'MFDFA':m.mfdfa(x),'Higuchi':m.higuchi(x)}

def vectors(z):
    e=z['SampEn'];return {'SampEn':np.array([e['value']]) if e['status']=='defined' else None,
                         'MFDFA_shape':m.mfshape(z['MFDFA']),'Higuchi_shape':m.hshape(z['Higuchi'])}
