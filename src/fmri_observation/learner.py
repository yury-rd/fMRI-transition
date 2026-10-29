"""Source-defined diagonal Gaussian HMM; no top-level execution."""
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

def posterior(x, model):
    pi,trans,mean,var=[np.asarray(model[k],dtype=float) for k in ['pi','trans','mean','var']]
    n,d=x.shape;k=len(pi)
    b=-.5*(d*np.log(2*np.pi)+np.log(var).sum(1)[None,:]
            +((x[:,None,:]-mean[None,:,:])**2/var[None,:,:]).sum(2))
    with np.errstate(divide='ignore'):
        lp=np.log(trans);lpi=np.log(pi)
    la=np.empty((n,k));la[0]=lpi+b[0]
    for t in range(1,n):la[t]=b[t]+logsumexp(la[t-1,:,None]+lp,axis=0)
    ll=float(logsumexp(la[-1]))
    lb=np.zeros((n,k))
    for t in range(n-2,-1,-1):lb[t]=logsumexp(lp+b[t+1][None,:]+lb[t+1][None,:],axis=1)
    gamma=np.exp(la+lb-ll)
    xi=np.exp(la[:-1,:,None]+lp[None,:,:]+b[1:,None,:]+lb[1:,None,:]-ll).sum(0)
    return ll,gamma,xi

def update(x,gamma,xi):
    k=gamma.shape[1]
    count=gamma.sum(0)+1e-12
    pi=np.maximum(gamma[0],1e-8);pi/=pi.sum()
    trans=xi+1.5/k;trans/=trans.sum(1)[:,None]
    mean=gamma.T@x/count[:,None]
    var=np.maximum(gamma.T@(x*x)/count[:,None]-mean*mean,.05**2)
    return {'pi':pi,'trans':trans,'mean':mean,'var':var}

def stationary(trans):
    k=len(trans);a=trans.T-np.eye(k);a[-1]=1;b=np.zeros(k);b[-1]=1
    return np.linalg.solve(a,b)

def stationary_historical_iteration(trans):
    v=np.full(len(trans),1/len(trans))
    for n in range(10000):
        nxt=v@trans
        if np.max(abs(nxt-v))<1e-13:return nxt,n+1
        v=nxt
    raise ValueError('stationary iteration did not converge within historical limit')

def flux(trans,stat):
    return float(.5*sum(abs(stat[i]*trans[i,j]-stat[j]*trans[j,i])
                         for i in range(len(stat)) for j in range(i+1,len(stat))))

def fit(train,k,seed,iterations=20):
    labels=KMeans(n_clusters=k,init='k-means++',n_init=8,max_iter=300,tol=1e-4,
                  algorithm='lloyd',random_state=seed,copy_x=True).fit_predict(train)
    if len(np.unique(labels))!=k:raise ValueError('empty initial cluster')
    trans=np.full((k,k),.2/(k-1));np.fill_diagonal(trans,.8)
    model={'pi':np.full(k,1/k),'trans':trans,
           'mean':np.stack([train[labels==i].mean(0) for i in range(k)]),
           'var':np.ones((k,train.shape[1]))}
    trace=[]
    for _ in range(iterations):
        ll,gamma,xi=posterior(train,model);trace.append(ll)
        if not np.all(np.isfinite(gamma)) or np.max(abs(gamma.sum(1)-1))>=1e-6:
            raise ValueError('posterior normalization outside historical guard')
        model=update(train,gamma,xi)
    ll,gamma,_=posterior(train,model);trace.append(ll)
    return {'model':model,'trace':trace,'train_gamma':gamma,'initial_labels':labels}
