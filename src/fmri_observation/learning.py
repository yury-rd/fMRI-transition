import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
def standard(a):
 mu=a.mean(0);sd=a.std(0);sd=np.where(sd>1e-12,sd,1.);return (a-mu)/sd,mu,sd
def ridge(a,y,b,alpha):
 aa,mu,sd=standard(a);ym=y.mean();coef=np.linalg.solve(aa.T@aa+alpha*np.eye(aa.shape[1]),aa.T@(y-ym));return ((b-mu)/sd)@coef+ym,(mu,sd,coef,np.array([ym]))
def prof(blocks,tr,te,target=None):
 a=[];b=[];mus=[];sds=[]
 for x in blocks:
  z,mu,sd=standard(x[tr]);w=np.sqrt(x.shape[1]);a.append(z/w);b.append((x[te]-mu)/sd/w);mus.append(mu);sds.append(sd)
 a=np.concatenate(a,1);b=np.concatenate(b,1);k=min(10,len(tr)-1,a.shape[1])
 if target is None:
  pca=PCA(n_components=k,svd_solver='full');u=pca.fit_transform(a);v=pca.transform(b);center=pca.mean_;rotation=pca.components_
 else:
  singular=np.linalg.svd(a-a.mean(0),compute_uv=False)
  if not len(singular) or singular[0]==0 or np.count_nonzero(singular>singular[0]*1e-10)<k:raise ValueError('UNSUPPORTED_PLS_RANK')
  pls=PLSRegression(n_components=k,scale=False,max_iter=1000,tol=1e-8);pls.fit(a,target);u=pls.transform(a);v=pls.transform(b);center=pls._x_mean;rotation=pls.x_rotations_.T
 return u,v,(np.concatenate(mus),np.concatenate(sds),center,rotation,u.var(0))
def covariates(num,cat,tr,te):
 aa=[num[tr]];bb=[num[te]];names=[]
 for j in range(cat.shape[1]):
  for c in sorted(set(cat[tr,j])):
   aa.append((cat[tr,j]==c).astype(float)[:,None]);bb.append((cat[te,j]==c).astype(float)[:,None]);names.append(str(j)+':'+c)
 return np.concatenate(aa,1),np.concatenate(bb,1),names
