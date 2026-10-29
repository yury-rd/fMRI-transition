"""Delayed Gaussian coupling with independent prehistory."""
import numpy as np

def finite_matrix(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError('A nonempty finite real matrix is required')
    return a

def lag_world(innovations, alpha):
    bank = finite_matrix(innovations)
    if bank.shape[1] != 7 or len(bank) < 2 or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Need (N+1)-by-7 innovations and alpha in [0,1]')
    x = bank[1:].copy()
    x[:, 1] = float(alpha)*bank[:-1, 0] + np.sqrt(1-float(alpha)**2)*bank[1:, 1]
    return x
