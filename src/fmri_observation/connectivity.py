"""Input is a prepared N-by-7 residual record; no nuisance refit is performed here."""
import numpy as np
ei=np.triu_indices(7,1)
def corr(x):
 z=x-x.mean(0);s=np.sqrt((z*z).sum(0));assert (s>0).all();return z.T@z/np.outer(s,s)
def phase(x,seed):
 f=np.fft.rfft(x,axis=0);rng=np.random.default_rng(seed);a=np.exp(1j*rng.uniform(0,2*np.pi,len(f)));a[0]=1
 if len(x)%2==0:a[-1]=rng.choice([-1.,1.])
 return np.fft.irfft(f*a[:,None],n=len(x),axis=0)
def disp(x,w,starts):return np.stack([corr(x[s:s+w])[ei] for s in starts]).std(0,ddof=1)
def geometry(n,w):
 assert n>=4*w,('UNSUPPORTED_LENGTH',n,w)
 return {**{k:list((np.arange(4)*w+s).astype(int)) for k,s in [('first4',0),('middle4',(n-4*w)//2),('last4',n-4*w)]},'spread4':np.floor(np.linspace(0,n-w,4)).astype(int).tolist()}
