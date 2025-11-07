from __future__ import annotations
import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationLearn, harmonizationApply

# Safe guard: pyriemann may be missing; we keep CSD optional and explicit.
_HAS_PYRIEMANN = True
try:
    from pyriemann.utils.mean import mean_riemann
    from pyriemann.utils.base import logm, expm
except Exception:
    _HAS_PYRIEMANN = False

from numpy.linalg import cholesky, inv

def combat_vector_features(df: pd.DataFrame, feature_cols: list[str], batch_col: str, covars: list[str] | None = None):
    """ComBat on numeric columns; returns adjusted DataFrame and model."""
    covars = covars or []
    X = df[feature_cols].to_numpy()
    cov = df[covars].copy() if covars else pd.DataFrame(index=df.index)
    cov = cov.assign(batch=df[batch_col])
    model = harmonizationLearn(X, cov)
    Xh = harmonizationApply(X, cov, model)
    out = df.copy()
    out[feature_cols] = Xh
    return out, model

def combat_riemann(Cs: list[np.ndarray], batch: list[str], covars: pd.DataFrame | None = None):
    """
    Log-map SPD CSDs to tangent space, run ComBat, exp-map back.
    Requires pyriemann; raises RuntimeError if unavailable.
    """
    if not _HAS_PYRIEMANN:
        raise RuntimeError("pyriemann not installed; install pyriemann==0.9 or disable CSD harmonisation.")

    Cs = np.asarray(Cs)
    Cref = mean_riemann(Cs)
    L = cholesky(Cref); Linv = inv(L)

    # log-map
    Y = []
    for Ci in Cs:
        Zi = Linv @ Ci @ Linv.T
        Yi = logm(Zi)
        Y.append(Yi)
    Y = np.stack(Y)  # (n, ch, ch)

    # vectorize upper triangle
    tri = np.triu_indices(Y.shape[1])
    Yv = Y[:, tri[0], tri[1]]
    cov = covars.copy() if covars is not None else pd.DataFrame(index=range(len(Yv)))
    cov = cov.assign(batch=batch)

    model = harmonizationLearn(Yv, cov)
    Yv_h = harmonizationApply(Yv, cov, model)

    # rebuild symmetric matrices
    Yh = np.zeros_like(Y)
    Yh[:, tri[0], tri[1]] = Yv_h
    Yh[:, tri[1], tri[0]] = Yv_h

    # exp-map back
    Cs_h = []
    for Yi in Yh:
        Zi = expm(Yi)
        Ci = L @ Zi @ L.T
        Ci = (Ci + Ci.T) / 2.0
        Ci += np.eye(Ci.shape[0]) * 1e-9
        Cs_h.append(Ci)
    return Cs_h
