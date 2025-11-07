from __future__ import annotations
import typing as _t
import numpy as _np
import pandas as _pd
from neuroHarmonize import harmonizationLearn, harmonizationApply

def neuroCombat(
    dat: _t.Union[_np.ndarray, _pd.DataFrame],
    covars: _t.Union[_pd.DataFrame, dict],
    eb: bool = True,
    parametric: bool = True,
    **kwargs,
):
    """
    Legacy API shim for neuroCombat → backed by neuroHarmonize.

    Returns:
      harmonized: np.ndarray
      model: dict-like model learned by harmonizationLearn
    """
    # Normalize inputs
    if isinstance(dat, _pd.DataFrame):
        X = dat.to_numpy()
    else:
        X = _np.asarray(dat)
    if isinstance(covars, dict):
        cov = _pd.DataFrame(covars)
    else:
        cov = _pd.DataFrame(covars)

    # neuroHarmonize supports eb/parametric flags; pass through if available
    model = harmonizationLearn(X, cov, eb=eb, parametric=parametric)
    Xh = harmonizationApply(X, cov, model)
    return Xh, model

def neuroCombatFromTraining(
    dat: _t.Union[_np.ndarray, _pd.DataFrame],
    model: dict,
    covars: _t.Union[_pd.DataFrame, dict],
):
    """
    Legacy apply-only function. Mirrors old behavior by calling harmonizationApply.
    """
    if isinstance(dat, _pd.DataFrame):
        X = dat.to_numpy()
    else:
        X = _np.asarray(dat)
    if isinstance(covars, dict):
        cov = _pd.DataFrame(covars)
    else:
        cov = _pd.DataFrame(covars)
    Xh = harmonizationApply(X, cov, model)
    return Xh
