"""Anonymous measurement core: joint projection, C7 delay energy and phase reference.

NumPy is the only third-party dependency. No data loading, clinical inference,
or nuisance-component selection is included. Inputs must already share retained
sample support. See README.md for exact conventions and qualification scope.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
FloatArray = NDArray[np.float64]
WINDOW = 10
CHANNELS = 7



def _validated_field(values: ArrayLike, window: int) -> FloatArray:
    field = np.asarray(values, dtype=np.float64)
    if field.ndim != 2 or field.shape[0] < 3 or field.shape[1] < 2:
        raise ValueError("values must be a samples-by-channels matrix with at least 3 samples and 2 channels")
    if not np.all(np.isfinite(field)):
        raise ValueError("values must have common finite support")
    if not isinstance(window, (int, np.integer)) or not 2 <= int(window) <= field.shape[0] - 1:
        raise ValueError("window must be an integer in [2, samples-1]")
    return field


def normalize_channels(values: ArrayLike, *, center: bool, scale: str) -> FloatArray:
    """Apply an explicit channel coordinate convention.

    ``scale`` is mandatory and is either ``none`` or ``unit-rms``.  Unit RMS is
    computed after the requested centering.  Constant channels fail closed.
    """
    field = np.asarray(values, dtype=np.float64)
    if field.ndim != 2 or not np.all(np.isfinite(field)):
        raise ValueError("values must be a finite matrix")
    if not isinstance(center, (bool, np.bool_)):
        raise ValueError("center must be explicitly Boolean")
    if scale not in {"none", "unit-rms"}:
        raise ValueError("scale must be explicitly 'none' or 'unit-rms'")
    result = field.copy()
    if center:
        result -= result.mean(axis=0, keepdims=True)
    if scale == "unit-rms":
        rms = np.sqrt(np.mean(np.square(result), axis=0))
        if np.any(rms <= np.finfo(np.float64).eps):
            raise ValueError("unit-rms scaling is undefined for a constant channel")
        result /= rms
    return result


def scalar_hankel(values: ArrayLike, window: int) -> FloatArray:
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or series.size < 3 or not np.all(np.isfinite(series)):
        raise ValueError("values must be a finite vector with at least 3 samples")
    if not isinstance(window, (int, np.integer)) or not 2 <= int(window) <= series.size - 1:
        raise ValueError("window must be an integer in [2, samples-1]")
    return np.lib.stride_tricks.sliding_window_view(series, int(window)).T.copy()


def joint_block_hankel(values: ArrayLike, window: int) -> FloatArray:
    """Return the (channels*window)-by-K synchronous vector-delay matrix."""
    field = _validated_field(values, window)
    return np.vstack([scalar_hankel(field[:, channel], int(window))
                      for channel in range(field.shape[1])])


@dataclass(frozen=True)
class OrthogonalResidualOperator:
    operator_id: str
    basis: np.ndarray
    rank: int
    samples: int

    def apply(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape[0] != self.samples:
            raise ValueError("operator and values have different sample counts")
        return matrix - self.basis @ (self.basis.T @ matrix)

    def matrix(self) -> np.ndarray:
        return np.eye(self.samples) - self.basis @ self.basis.T


def _finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite matrix with at least two rows")
    return matrix


def residual_operator(
    design: np.ndarray,
    *,
    operator_id: str,
    include_intercept: bool = True,
    tolerance_multiplier: float = 1.0,
) -> OrthogonalResidualOperator:
    """Construct a span-normalized joint orthogonal residual projector."""
    matrix = _finite_matrix(design, "design")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    raw_scale = np.maximum(1.0, np.max(np.abs(matrix), axis=0))
    threshold = np.finfo(np.float64).eps * np.sqrt(matrix.shape[0]) * raw_scale
    norms = np.linalg.norm(centered, axis=0)
    numerical = centered[:, norms > threshold]
    if numerical.shape[1]:
        numerical = numerical / norms[norms > threshold]
    if include_intercept:
        numerical = np.column_stack([numerical, np.ones(matrix.shape[0])])
    _, singular_values, vh = np.linalg.svd(numerical.T, full_matrices=False)
    cutoff = (
        tolerance_multiplier
        * np.finfo(np.float64).eps
        * max(numerical.shape)
        * float(singular_values[0])
        if singular_values.size
        else 0.0
    )
    rank = int(np.sum(singular_values > cutoff))
    return OrthogonalResidualOperator(operator_id, vh[:rank].T, rank, matrix.shape[0])


def _matrix(values: ArrayLike, *, columns: int | None = None) -> FloatArray:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 16 or data.shape[1] < 2:
        raise ValueError("a samples-by-features matrix with at least 16 rows is required")
    if columns is not None and data.shape[1] != columns:
        raise ValueError(f"expected {columns} columns")
    if not np.all(np.isfinite(data)):
        raise ValueError("input must be finite")
    return data


def aggregate_networks(values: ArrayLike, membership: ArrayLike) -> FloatArray:
    data = _matrix(values)
    labels = np.asarray(membership)
    if labels.ndim != 1 or len(labels) != data.shape[1]:
        raise ValueError("one network label per feature is required")
    names = np.unique(labels)
    if len(names) < 2 or any(np.count_nonzero(labels == name) == 0 for name in names):
        raise ValueError("at least two nonempty networks are required")
    return np.column_stack([data[:, labels == name].mean(axis=1) for name in names])


def _field(values):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=CHANNELS or x.shape[0]<18 or not np.isfinite(x).all():
        raise ValueError('Need common finite N-by-7 field with N>=18')
    normalize_channels(x,center=True,scale='unit-rms')
    return x


def _normalized_hankel(x):
    return joint_block_hankel(normalize_channels(x,center=True,scale='unit-rms'),WINDOW)


def _batch_stats(fields):
    j=np.stack([_normalized_hankel(x) for x in fields])
    gram=j@j.transpose(0,2,1)
    eigenvalues=np.linalg.eigvalsh(gram,UPLO='L')
    trace=np.trace(gram,axis1=1,axis2=2)
    fraction=eigenvalues[:,-2:].sum(axis=1)/trace
    if not np.isfinite(fraction).all() or np.any(fraction<0) or np.any(fraction>1+1e-12):
        raise RuntimeError('Invalid leading-two fraction')
    channel_trace=np.sum(np.square(j).reshape(len(j),CHANNELS,WINDOW,j.shape[-1]),axis=(2,3))
    return fraction,trace,channel_trace


def rho2(values):
    """Source-equivalent top2 singular-energy fraction; no row recentering."""
    return float(_batch_stats([_field(values)])[0][0])


def _surrogate_from_fft(ft,n,seed):
    if not isinstance(seed,(int,np.integer)) or int(seed)<0:
        raise ValueError('seed must be a nonnegative integer')
    phase=np.random.default_rng(int(seed)).uniform(0.0,2*np.pi,size=ft.shape)
    phase[0,:]=0.0
    if n%2==0: phase[-1,:]=0.0
    return np.fft.irfft(ft*np.exp(1j*phase),n=n,axis=0,norm='backward')


def independent_phase_surrogate(values,*,seed):
    x=_field(values)
    return _surrogate_from_fft(np.fft.rfft(x,n=len(x),axis=0,norm='backward'),len(x),seed)


def fullscore(values,*,seed_start,repetitions=199,batch_size=16):
    """Keep all surrogate scores and MC diagnostics, no participant inference."""
    x=_field(values)
    if not isinstance(seed_start,(int,np.integer)) or int(seed_start)<0:
        raise ValueError('seed_start must be a nonnegative integer')
    if repetitions!=199: raise ValueError('This measurement requires exactly 199 surrogates')
    if not isinstance(batch_size,int) or batch_size<1: raise ValueError('Invalid batch size')
    observed,observed_trace,observed_channel_trace=_batch_stats([x])
    ft=np.fft.rfft(x,n=len(x),axis=0,norm='backward')
    scores=[];traces=[];channel_traces=[];max_spectral_error=0.0
    spectral_scale=max(float(np.max(abs(ft))),np.finfo(float).tiny)
    for start in range(0,repetitions,batch_size):
        fields=[_surrogate_from_fft(ft,len(x),int(seed_start)+r) for r in range(start,min(start+batch_size,repetitions))]
        ss,tt,cc=_batch_stats(fields)
        scores.extend(ss.tolist());traces.extend(tt.tolist());channel_traces.extend(cc.tolist())
        transformed=np.fft.rfft(np.stack(fields),n=len(x),axis=1,norm='backward')
        max_spectral_error=max(max_spectral_error,float(np.max(abs(abs(transformed)-abs(ft)[None,:,:]))/spectral_scale))
    scores=np.asarray(scores);traces=np.asarray(traces);channel_traces=np.asarray(channel_traces)
    sd=float(np.std(scores,ddof=1));mean=float(np.mean(scores))
    ratios=channel_traces/observed_channel_trace[0]
    return {'observed_rho2':float(observed[0]),'surrogate_rho2':scores.tolist(),
            'surrogate_mean':mean,'excess':float(observed[0]-mean),'surrogate_std':sd,
            'montecarlo_standard_error':sd/np.sqrt(repetitions),
            'first100_minus_last99':float(np.mean(scores[:100])-np.mean(scores[100:])),
            'surrogate_rho2_min':float(scores.min()),'surrogate_rho2_max':float(scores.max()),
            'observed_hankel_trace':float(observed_trace[0]),
            'surrogate_hankel_trace_min':float(traces.min()),'surrogate_hankel_trace_max':float(traces.max()),
            'observed_channel_hankel_traces':observed_channel_trace[0].tolist(),
            'surrogate_channel_trace_ratio_min':ratios.min(axis=0).tolist(),
            'surrogate_channel_trace_ratio_max':ratios.max(axis=0).tolist(),
            'maximum_relative_marginal_spectrum_error':max_spectral_error,
            'seed_start':int(seed_start),'surrogates':repetitions,'samples':len(x),'channels':CHANNELS,'window':WINDOW}
