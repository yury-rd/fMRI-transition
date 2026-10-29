"""Same ridge VAR model evaluated throughout at 50 decimal digits.

Uses Cholesky solves/factor density; independent reference uses LU/determinant.
Starts from raw observed binary inputs, not already rounded standardized data.
"""
import mpmath as mp

def score(y,alpha,cut,order,starts):
    with mp.workdps(50):
        yy=[[mp.mpf(float(x)) for x in row] for row in y]
        aa=[mp.mpf(float(x)) for x in alpha]
        n,d=len(yy),len(yy[0])
        z=[yy[0]]+ [[(yy[t][j]-aa[j]*yy[t-1][j])/(1-aa[j]) for j in range(d)] for t in range(1,n)]
        center=[mp.fsum(row[j] for row in z[:cut])/cut for j in range(d)]
        scale=[mp.sqrt(mp.fsum((row[j]-center[j])**2 for row in z[:cut])/cut) for j in range(d)]
        if any(s<=0 for s in scale):raise ValueError('nonpositive train scale')
        v=[[(row[j]-center[j])/scale[j] for j in range(d)] for row in z]
        jac=-mp.fsum(mp.log(s) for s in scale)-mp.fsum(mp.log(1-a) for a in aa)
        features=[[1]+[v[t-lag][j] for lag in range(1,order+1) for j in range(d)] for t in range(order,n)]
        x=mp.matrix(features[:cut-order]);target=mp.matrix(v[order:cut])
        penalty=mp.eye(x.cols);penalty[0,0]=0
        gram=x.T*x+penalty;right=x.T*target;coef=mp.matrix(x.cols,d)
        for j in range(d):
            sol=mp.cholesky_solve(gram,right[:,j],tol=None)
            for i in range(x.cols):coef[i,j]=sol[i]
        residual=target-x*coef;cov=residual.T*residual/x.rows+mp.mpf('.0001')*mp.eye(d)
        lower=mp.cholesky(cov,tol=None)
        logdet=2*mp.fsum(mp.log(lower[j,j]) for j in range(d))
        scores=[None]*order
        for t in range(order,n):
            pred=mp.matrix([features[t-order]])*coef
            err=[v[t][j]-pred[j] for j in range(d)]
            whiten=[]
            for j in range(d):whiten.append((err[j]-mp.fsum(lower[j,k]*whiten[k] for k in range(j)))/lower[j,j])
            q=mp.fsum(x*x for x in whiten)
            scores.append(-(d*mp.log(2*mp.pi)+logdet+q)/2+jac)
        return {'mean_by_start':{b:float(mp.fsum(scores[b:])/(n-b)) for b in starts},
                'per_frame':[float(x) if x is not None else None for x in scores],
                'coefficient':[[float(coef[i,j]) for j in range(d)] for i in range(coef.rows)],
                'innovation_covariance':[[float(cov[i,j]) for j in range(d)] for i in range(d)],
                'precision_decimal_digits':50}
