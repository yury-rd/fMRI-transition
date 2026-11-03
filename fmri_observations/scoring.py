"""Conditional frame densities in original observation units."""
import numpy as np
from scipy.special import logsumexp

def transform(y, alpha, cut):
    z = y.copy()
    z[1:] = (y[1:] - alpha * y[:-1]) / (1 - alpha)
    mu = z[:cut].mean(axis=0)
    sd = z[:cut].std(axis=0, ddof=0)
    if not np.all(sd > 0):
        raise ValueError("zero training scale")
    v = (z - mu) / sd
    jac = np.full(len(y), -np.log(sd).sum())
    jac[1:] -= np.log1p(-alpha).sum()
    return v, jac

def safe_log(p):
    out = np.full_like(p, -np.inf, dtype=float)
    np.log(p, out=out, where=p > 0)
    return out

def hmm_frames(v, model):
    pi, trans = np.asarray(model['pi']), np.asarray(model['trans'])
    mu, var = np.asarray(model['mean']), np.asarray(model['var'])
    emission = -.5 * (v.shape[1] * np.log(2 * np.pi) + np.log(var).sum(1)
                     + ((v[:, None] - mu) ** 2 / var).sum(2))
    lp, lt = safe_log(pi), safe_log(trans)
    scores = []
    for t, e in enumerate(emission):
        joint = lp + e if t == 0 else logsumexp(lp[:, None] + lt, axis=0) + e
        inc = logsumexp(joint)
        lp = joint - inc
        scores.append(float(inc))
    return np.asarray(scores)

def var_frames(v, cut, order):
    d = v.shape[1]
    x = np.asarray([[1., *np.concatenate([v[t-lag] for lag in range(1, order+1)])]
                    for t in range(order, len(v))])
    tx = x[:cut-order]
    penalty = np.eye(1 + order*d)
    penalty[0, 0] = 0
    coef = np.linalg.solve(tx.T @ tx + penalty, tx.T @ v[order:cut])
    resid = v[order:cut] - tx @ coef
    cov = resid.T @ resid / len(resid) + 1e-4 * np.eye(d)
    sign, ld = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError('nonpositive innovation covariance')
    err = v[order:] - x @ coef
    quad = np.sum(err * np.linalg.solve(cov, err.T).T, axis=1)
    return np.r_[np.full(order, np.nan), -.5 * (d*np.log(2*np.pi) + ld + quad)]
