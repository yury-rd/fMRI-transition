"""Five controlled observation and proxy worlds from the completed panel."""
import hashlib
import numpy as np
from .score import residual_operator,aggregate_networks
from . import score as h1
NETWORKS=['Cont','Default','DorsAttn','Limbic','SalVentAttn','SomMot','Vis']
COUNTS=[52,91,46,26,47,77,61]
WORLDS={'null':[0.,.25,.5,.75,1.],
        'observation_amplitude':[0.,.5,1.,2.,4.],
        'proxy_quality':[0.,.25,.5,.75,1.],
        'latent_sharing':[0.,.2,.4,.6,.8],
        'proxy_contamination':[0.,.5,1.,2.,4.]}

def membership():
    return np.asarray([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],dtype=int)

def array_hash(x):return hashlib.sha256(np.ascontiguousarray(x,dtype=np.float64).tobytes()).hexdigest()

def ar_from_innovations(e,phi):
    e=np.asarray(e,dtype=float)
    if e.ndim!=2 or len(e)<2 or not np.isfinite(e).all() or not abs(phi)<1:raise ValueError('Finite innovations and stationary phi required')
    y=np.empty_like(e);y[0]=e[0]
    sigma=np.sqrt(1.-phi*phi)
    for t in range(1,len(e)):y[t]=phi*y[t-1]+sigma*e[t]
    return y

def ar(n,k,phi,seed):
    return ar_from_innovations(np.random.default_rng(int(seed)).standard_normal(size=(n,k),dtype=np.float64),phi)

def bank(n,seed):
    idx=membership()
    z=ar(n,7,.5,seed);u=ar(n,1,.5,seed+1);v=ar(n,1,.5,seed+2);h=ar(n,1,.5,seed+3)
    extra=ar(n,4,.5,seed+4);motion=ar(n,6,.3,seed+5)
    derivative=np.vstack((np.zeros((1,6)),np.diff(motion,axis=0)))
    m24=np.column_stack((motion,derivative,motion**2,derivative**2))
    within=.3*ar(n,400,.5,seed+6)
    for j in range(7):within[:,idx==j]-=within[:,idx==j].mean(axis=1,keepdims=True)
    coefficients=.01*np.random.default_rng(int(seed+7)).standard_normal(size=(24,400),dtype=np.float64)
    trend=np.linspace(-1.,1.,n,dtype=float)[:,None]
    design=np.column_stack((trend,m24))
    base=residual_operator(design,operator_id='SYNTHETIC_BASE',include_intercept=True)
    return dict(n=n,seed=int(seed),membership=idx,z=z,u=u,v=v,h=h,extra=extra,m24=m24,
                within=within,motion_field=m24@coefficients,design=design,base=base)

def world(b,name,level):
    z,u,v,h=b['z'],b['u'],b['v'],b['h']
    if name=='null':signal=z;artifact=np.zeros_like(z);proxy=v
    elif name=='observation_amplitude':signal=z;artifact=np.broadcast_to(level*u,z.shape);proxy=u+.5*v
    elif name=='proxy_quality':
        if not 0<=level<=1:raise ValueError('q outside support')
        signal=z;artifact=np.broadcast_to(u,z.shape);proxy=np.sqrt(level)*u+np.sqrt(1-level)*v
    elif name=='latent_sharing':
        if not 0<=level<1:raise ValueError('r outside support')
        signal=np.sqrt(1-level)*z+np.sqrt(level)*h;artifact=np.broadcast_to(u,z.shape);proxy=u+.5*v
    elif name=='proxy_contamination':
        signal=np.sqrt(.75)*z+.5*h;artifact=np.broadcast_to(u,z.shape);proxy=u+level*h+.5*v
    else:raise ValueError('Unknown world')
    proxies=np.column_stack((proxy,b['extra']))
    clean_observed=signal+artifact
    parcels=clean_observed[:,b['membership']]+b['within']+b['motion_field']
    a5=residual_operator(np.column_stack((b['design'],proxies)),operator_id='SYNTHETIC_A5',include_intercept=True)
    fields={};energies={};errors=[]
    for label,op in [('BASE',b['base']),('A5',a5)]:
        fields[label]=aggregate_networks(op.apply(parcels),b['membership'])
        s=op.apply(signal);a=op.apply(artifact)
        errors.append(float(np.max(abs(fields[label]-s-a))))
        energies[label]={'signal':float(np.sum(s*s)),'artifact':float(np.sum(a*a)),
                         'cross':float(2*np.sum(s*a)),'total':float(np.sum(fields[label]**2))}
    e0,e1=energies['BASE'],energies['A5']
    diagnostics={'rank_BASE':b['base'].rank,'rank_A5':a5.rank,
        'full400_projection_aggregation_error':max(errors),
        'energy':energies,'signal_retention_A5_over_BASE':e1['signal']/e0['signal'],
        'artifact_retention_A5_over_BASE':e1['artifact']/e0['artifact'] if e0['artifact']>0 else None,
        'acquired_parcels_sha256':array_hash(parcels),'A5_proxies_sha256':array_hash(proxies),
        'signal_sha256':array_hash(signal),'BASE_C7_sha256':array_hash(fields['BASE'])}
    return dict(fields=fields,signal=signal,artifact=artifact,proxies=proxies,parcels=parcels,
                a5=a5,diagnostics=diagnostics,alternative_signal=clean_observed if name=='observation_amplitude' else None)

def score(x,seed,cache):
    key=(array_hash(x),int(seed))
    if key not in cache:cache[key]=h1.fullscore(x,seed_start=int(seed),repetitions=199,batch_size=16)
    return cache[key]

def summarize_replicates(values):
    a=np.asarray(values,dtype=float)
    if a.ndim!=2 or a.shape[0]!=16 or not np.isfinite(a).all():raise ValueError('16 complete replicate curves required')
    diff=np.diff(a,axis=1)
    return {'mean':a.mean(axis=0).tolist(),'simulation_SE':(a.std(axis=0,ddof=1)/np.sqrt(16)).tolist(),
            'adjacent_mean_difference':diff.mean(axis=0).tolist(),
            'adjacent_simulation_SE':(diff.std(axis=0,ddof=1)/np.sqrt(16)).tolist(),
            'replicates':16,'scope':'Mean response and MC standard error across independent generator replicates, paired across levels; no p/CI or patient sampling inference.'}
