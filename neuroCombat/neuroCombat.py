from __future__ import annotations
import typing as _t
import numpy as _np
import pandas as _pd
from neuroHarmonize import harmonizationLearn, harmonizationApply

# ---------- helpers ----------
def _to_dataframe(x) -> _pd.DataFrame:
    if isinstance(x, _pd.DataFrame):
        return x.copy()
    if isinstance(x, dict):
        return _pd.DataFrame(x)
    return _pd.DataFrame(_np.asarray(x))

def make_design_matrix(
    covars: _t.Union[_pd.DataFrame, dict],
    batch_col: str = "batch",
    categorical_cols: _t.Sequence[str] | None = None,
    continuous_cols: _t.Sequence[str] | None = None,
    drop_first: bool = True,
    add_intercept: bool = True,
) -> _pd.DataFrame:
    df = _to_dataframe(covars)
    if batch_col not in df.columns:
        raise ValueError(f"'{batch_col}' column is required in covariates for ComBat/neuroHarmonize.")
    if categorical_cols is None and continuous_cols is None:
        categorical_cols = [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name == "category"]
        continuous_cols = [c for c in df.columns if c not in categorical_cols]
    categorical_cols = list(categorical_cols or [])
    continuous_cols = list(continuous_cols or [])
    if batch_col not in categorical_cols:
        categorical_cols = [batch_col] + categorical_cols
    X_cat = _pd.get_dummies(df[categorical_cols].astype("category"), drop_first=drop_first)
    X_cont = df[continuous_cols].apply(_pd.to_numeric, errors="coerce") if continuous_cols else _pd.DataFrame(index=df.index)
    X = _pd.concat([X_cat, X_cont], axis=1)
    if add_intercept and "intercept" not in X.columns:
        X.insert(0, "intercept", 1.0)
    return X

# ---------- legacy API shims ----------
def neuroCombat(
    dat: _t.Union[_np.ndarray, _pd.DataFrame],
    covars: _t.Union[_pd.DataFrame, dict],
    eb: bool = True,
    parametric: bool = True,
    **kwargs,
):
    X = _np.asarray(dat.values if isinstance(dat, _pd.DataFrame) else dat)
    cov = _to_dataframe(covars)
    batch = kwargs.pop("batch", None)
    if batch is not None and "batch" not in cov.columns:
        cov = cov.copy()
        cov["batch"] = _np.asarray(batch).ravel()
    if "batch" not in cov.columns:
        raise ValueError("Covariates must include a 'batch' column for ComBat/neuroHarmonize.")
    model = harmonizationLearn(X, cov, eb=eb, parametric=parametric)
    Xh = harmonizationApply(X, cov, model)
    return Xh, model

def neuroCombatFromTraining(
    dat: _t.Union[_np.ndarray, _pd.DataFrame],
    model: dict,
    covars: _t.Union[_pd.DataFrame, dict],
):
    X = _np.asarray(dat.values if isinstance(dat, _pd.DataFrame) else dat)
    cov = _to_dataframe(covars)
    if "batch" not in cov.columns:
        raise ValueError("Covariates must include a 'batch' column for ComBat/neuroHarmonize.")
    return harmonizationApply(X, cov, model)

def adjust_data_final(
    dat: _t.Union[_np.ndarray, _pd.DataFrame],
    covars: _t.Union[_pd.DataFrame, dict],
    estimates_or_model: dict,
):
    X = _np.asarray(dat.values if isinstance(dat, _pd.DataFrame) else dat)
    cov = _to_dataframe(covars)
    if "batch" not in cov.columns:
        raise ValueError("Covariates must include a 'batch' column for ComBat/neuroHarmonize.")
    return harmonizationApply(X, cov, estimates_or_model)

def find_parametric_adjustments(
    dat: _t.Union[_np.ndarray, _pd.DataFrame],
    covars: _t.Union[_pd.DataFrame, dict],
    eb: bool = True,
    parametric: bool = True,
    **kwargs,
) -> dict:
    X = _np.asarray(dat.values if isinstance(dat, _pd.DataFrame) else dat)
    cov = _to_dataframe(covars)
    batch = kwargs.pop("batch", None)
    if batch is not None and "batch" not in cov.columns:
        cov = cov.copy()
        cov["batch"] = _np.asarray(batch).ravel()
    if "batch" not in cov.columns:
        raise ValueError("Covariates must include a 'batch' column for ComBat/neuroHarmonize.")
    model = harmonizationLearn(X, cov, eb=eb, parametric=parametric)
    return model

# ---------- legacy EB helper shims ----------
def aprior(s2: _t.Union[_np.ndarray, list, _pd.Series]) -> tuple[float, float]:
    """
    Legacy: method-of-moments estimate of Inverse-Gamma(alpha, beta) for a set of
    batch variance estimates s2. Returns (alpha, beta). Safe fallback used rarely.
    """
    arr = _np.asarray(s2, dtype=float).ravel()
    if arr.size < 2 or _np.allclose(arr.var(ddof=1), 0.0):
        # Degenerate case: weakly-informative prior
        alpha = 2.0
        beta = max(arr.mean(), 1e-6) * (alpha - 1.0)
        return float(alpha), float(beta)
    m = arr.mean()
    v = arr.var(ddof=1)
    alpha = (m**2 / v) + 2.0
    beta = m * (alpha - 1.0)
    return float(alpha), float(beta)

def bprior(s2: _t.Union[_np.ndarray, list, _pd.Series]) -> float:
    """
    Legacy companion returning just the beta parameter for convenience.
    """
    _, beta = aprior(s2)
    return float(beta)
