from __future__ import annotations
import typing as _t
import numpy as _np
import pandas as _pd
from neuroHarmonize import harmonizationLearn, harmonizationApply

def _to_dataframe(x) -> _pd.DataFrame:
    if isinstance(x, _pd.DataFrame):
        return x.copy()
    if isinstance(x, dict):
        return _pd.DataFrame(x)
    # assume array-like
    return _pd.DataFrame(_np.asarray(x))

def make_design_matrix(
    covars: _t.Union[_pd.DataFrame, dict],
    batch_col: str = "batch",
    categorical_cols: _t.Sequence[str] | None = None,
    continuous_cols: _t.Sequence[str] | None = None,
    drop_first: bool = True,
    add_intercept: bool = True,
) -> _pd.DataFrame:
    """
    Legacy-style design-matrix builder.
    Produces a numeric matrix with one-hot for categorical covars and raw values for continuous covars.
    neuroHarmonize does not strictly require a design matrix, but some older pipelines import this symbol.
    """
    df = _to_dataframe(covars)

    if batch_col not in df.columns:
        raise ValueError(f"'{batch_col}' column is required in covariates for ComBat/neuroHarmonize.")

    # Default heuristics: everything non-numeric is categorical except explicit continuous_cols
    if categorical_cols is None and continuous_cols is None:
        categorical_cols = [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name == "category"]
        continuous_cols = [c for c in df.columns if c not in categorical_cols]

    categorical_cols = list(categorical_cols or [])
    continuous_cols = list(continuous_cols or [])

    # Ensure batch is treated as categorical
    if batch_col not in categorical_cols:
        categorical_cols = [batch_col] + categorical_cols

    # One-hot encode categoricals (including batch)
    X_cat = _pd.get_dummies(df[categorical_cols].astype("category"), drop_first=drop_first)

    # Keep continuous columns as-is (ensure numeric)
    X_cont = df[continuous_cols].apply(_pd.to_numeric, errors="coerce") if continuous_cols else _pd.DataFrame(index=df.index)

    # Combine
    X = _pd.concat([X_cat, X_cont], axis=1)

    if add_intercept and "intercept" not in X.columns:
        X.insert(0, "intercept", 1.0)

    return X

def neuroCombat(
    dat: _t.Union[_np.ndarray, _pd.DataFrame],
    covars: _t.Union[_pd.DataFrame, dict],
    eb: bool = True,
    parametric: bool = True,
    **kwargs,
):
    """
    Legacy API shim for neuroCombat → backed by neuroHarmonize.
    Returns (harmonized, model) to match prior expectations.
    Accepts optional `batch=` in kwargs and injects it into covars if provided.
    """
    X = _np.asarray(dat.values if isinstance(dat, _pd.DataFrame) else dat)
    cov = _to_dataframe(covars)

    # Support legacy signature neuroCombat(dat, covars, batch=...)
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
    """
    Legacy apply-only function. Mirrors old behavior via harmonizationApply.
    """
    X = _np.asarray(dat.values if isinstance(dat, _pd.DataFrame) else dat)
    cov = _to_dataframe(covars)
    if "batch" not in cov.columns:
        raise ValueError("Covariates must include a 'batch' column for ComBat/neuroHarmonize.")
    return harmonizationApply(X, cov, model)
