"""Matched nested PLS/ridge; every transform is fitted within training rows."""
import numpy as np
from . import learning as legacy
ALPHAS=[.1,1.,10.,100.]

def fixture():
    rng=np.random.Generator(np.random.PCG64(732401))
    s=rng.normal(size=(48,21));e=rng.normal(size=(48,21));d=.3*s+.7*e
    s[:,20]=2.;d[:,20]=-1.
    cn=np.column_stack([rng.normal(size=(48,2)),np.ones(48)])
    cat=np.array([['A' if i%2==0 else 'B',['x','y','z'][i%3]] for i in range(48)],dtype='<U12')
    cat[0,1]='only-test'
    y=.3*cn[:,0]-.2*cn[:,1]+.7*s[:,0]-.4*d[:,1]+.25*s[:,2]**2+.4*rng.normal(size=48)
    return {'static':s,'dispersion':d,'combined':np.column_stack([s,d]),'clinical_numeric':cn,'categories':cat,'target':y}

def candidate_profile(x,tr,te,y):
    u,v,pars=legacy.prof([x],np.array(tr),np.array(te),y[np.array(tr)])
    names=['MRI_mean','MRI_sd','projection_mean','projection_matrix','projection_variance']
    return {'train_scores':u,'test_scores':v,'parameters':dict(zip(names,pars))}

def candidate_model(data,s,rep):
    y=data['target'];cn=data['clinical_numeric'];cat=data['categories'];x=None if rep=='clinical' else data[rep]
    cache=[];transforms=[]
    for inn in s['inner']:
        ii=np.array(inn['train']);jj=np.array(inn['test'])
        a,b,cats=legacy.covariates(cn,cat,ii,jj)
        pr=None if x is None else candidate_profile(x,ii,jj,y)
        cache.append((ii,jj,a,b,pr));transforms.append({'categories':cats,'MRI':pr})
    candidates=[]
    for k in ([0] if x is None else [2,5,10]):
        for alpha in ALPHAS:
            losses=[];inner=[]
            for ii,jj,a,b,pr in cache:
                aa,bb=(a,b) if x is None else (np.column_stack([a,pr['train_scores'][:,:k]]),np.column_stack([b,pr['test_scores'][:,:k]]))
                pred,pars=legacy.ridge(aa,y[ii],bb,alpha)
                losses.extend(abs(pred-y[jj]));inner.append({'prediction':pred,'ridge_parameters':pars})
            candidates.append({'loss':float(np.mean(losses)),'k':k,'alpha':alpha,'inner':inner})
    best=min(candidates,key=lambda z:(z['loss'],z['k'],z['alpha']))
    tr=np.array(s['train']);te=np.array(s['test']);a,b,cats=legacy.covariates(cn,cat,tr,te)
    pr=None if x is None else candidate_profile(x,tr,te,y)
    aa,bb=(a,b) if x is None else (np.column_stack([a,pr['train_scores'][:,:best['k']]]),np.column_stack([b,pr['test_scores'][:,:best['k']]]))
    pred,pars=legacy.ridge(aa,y[tr],bb,best['alpha'])
    return {'inner_transforms':transforms,'candidates':candidates,'selected':{'k':best['k'],'alpha':best['alpha']},
            'outer_transform':{'categories':cats,'MRI':pr},'outer_prediction':pred,'outer_ridge_parameters':pars}
