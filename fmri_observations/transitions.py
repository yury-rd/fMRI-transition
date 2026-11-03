"""Frozen synthetic generators and train-only conditional scoring."""
import time
import numpy as np
from . import scoring,learner,observation,var_high_precision

def simulate(case,cfg):
    """Historical generator definitions, fresh registered seeds; no target data."""
    n=case['N'];world=case['world'];amp=case['amplitude']
    rng=np.random.default_rng(case['seed']+1000*n)
    truth=None
    if world in ('cyclic','reversible'):
        trans=np.asarray(cfg['generators'][world]['transition'],float)
        mu=np.asarray(cfg['generators']['means'],float);mu-=mu.mean(axis=0)
        truth=np.empty(n,int);truth[0]=rng.integers(3)
        for t in range(1,n):truth[t]=rng.choice(3,p=trans[truth[t-1]])
        x=amp*mu[truth]+.60*rng.standard_normal((n,7))
    elif world=='common_ar':
        h=np.zeros(n);e=rng.standard_normal(n)
        for t in range(1,n):h[t]=.90*h[t-1]+e[t]
        x=h[:,None]*np.asarray([.8,1.,1.2,.7,1.1,.9,1.3])+.7*rng.standard_normal((n,7))
    elif world=='rotating_ar':
        theta=.23;rho=.97
        rot=rho*np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        z=np.zeros((n,2));e=.30*rng.standard_normal((n,2))
        for t in range(1,n):z[t]=rot@z[t-1]+e[t]
        angles=np.linspace(0,2*np.pi,7,endpoint=False)
        load=np.stack([np.cos(angles),np.sin(angles)],axis=1)
        x=z@load.T+.45*rng.standard_normal((n,7))
    else:raise ValueError(world)
    return x,truth

def observe(x,cfg,val_end):
    alpha=np.asarray(cfg['observation_alpha'])
    stationary=np.empty_like(x);stationary[0]=x[0]
    change=np.empty_like(x);change[0]=x[0]
    for t in range(1,len(x)):
        stationary[t]=alpha*stationary[t-1]+(1-alpha)*x[t]
        a=alpha if t<val_end else alpha[::-1]
        change[t]=a*change[t-1]+(1-a)*x[t]
    units=np.asarray(cfg['unit_scale'])
    return {'identity':x.copy(),'stationary_filter':stationary,
            'filter_change_at_test':change,'unit_change':stationary*units}

def hmm_candidate(y,alpha,k,seed,cut,val_end):
    t=time.perf_counter();v,jac=scoring.transform(y,alpha,cut)
    fit=learner.fit(v[:cut],k,seed,iterations=20);model=fit['model']
    frames=scoring.hmm_frames(v,model)+jac
    reset_test=scoring.hmm_frames(v[val_end:],model)+jac[val_end:]
    reset_val=scoring.hmm_frames(v[cut:val_end],model)+jac[cut:val_end]
    stat=learner.stationary(model['trans'])
    val=float(frames[cut:val_end].mean());test=float(frames[val_end:].mean())
    vals=[*frames,float(learner.flux(model['trans'],stat)),*stat]
    if not np.all(np.isfinite(vals)):raise ValueError('nonfinite HMM outputs')
    # Finite causal prefix check: a future suffix is not used by forward scoring.
    prefix=scoring.hmm_frames(v[:val_end],model)+jac[:val_end]
    if not np.array_equal(prefix,frames[:val_end]):raise ValueError('forward prefix changed')
    return {'status':'defined','family':'HMM','states':k,'alpha':alpha,
            'validation':val,'test':test,'frame_logdensity':frames,
            'reset_validation':float(reset_val.mean()),'reset_test':float(reset_test.mean()),
            'reset_minus_conditioned_frames':reset_test-frames[val_end:],
            'model':model,'stationary':stat,'flux':learner.flux(model['trans'],stat),
            'trace':fit['trace'],'trace_decreases':int(np.sum(np.diff(fit['trace']) < -1e-10)),
            'initial_labels':fit['initial_labels'],
            'free_parameter_count':(k-1)+k*(k-1)+2*k*y.shape[1],
            'elapsed_seconds':time.perf_counter()-t}

def var_candidate(y,alpha,order,cut,val_end):
    t=time.perf_counter()
    out=var_high_precision.score(y,alpha,cut,order,[cut,val_end])
    frames=np.array(out['per_frame'][order:],float)
    if not np.all(np.isfinite(frames)):raise ValueError('nonfinite VAR outputs')
    # Scores are returned per frame; means below use the same binary64 summary
    # as HMM. The qualified MP means are retained as an additional check.
    full=out['per_frame'];val=float(np.mean(full[cut:val_end]));test=float(np.mean(full[val_end:]))
    if abs(test-out['mean_by_start'][val_end])>1e-10:raise ValueError('VAR mean conversion accuracy')
    d=y.shape[1]
    return {'status':'defined','family':'VAR','order':order,'alpha':alpha,'validation':val,'test':test,
            'frame_logdensity':full,'model':out,'free_parameter_count':d*(1+order*d)+d*(d+1)//2,
            'elapsed_seconds':time.perf_counter()-t}

def select(fits,ids):
    defined=[i for i in ids if fits[i]['status']=='defined']
    if not defined:return None
    # Exact ties preserve prospective menu order. Failed candidates are visible.
    return max(defined,key=lambda i:fits[i]['validation'])
