from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def site_variance_ratio(df: pd.DataFrame, feature_cols: list[str], batch_col: str) -> float:
    """
    Ratio of between-site variance to total variance for a single feature (first col used).
    """
    feat = feature_cols[0]
    x = df[feat].to_numpy()
    groups = df[batch_col].astype("category").cat.codes.to_numpy()
    grand = x.mean()
    # Between-group variance
    var_b = 0.0
    for g in np.unique(groups):
        mask = groups == g
        var_b += mask.mean() * (x[mask].mean() - grand) ** 2
    var_t = x.var() + 1e-12
    return float(var_b / var_t)

def preservation_delta(df_pre: pd.DataFrame, df_post: pd.DataFrame, y: str, x: str) -> float:
    """
    Change in slope of y~x regression pre vs post harmonization.
    """
    def slope(a, b):
        lr = LinearRegression()
        X = a[[b]].to_numpy()
        Y = a[y].to_numpy()
        lr.fit(X, Y)
        return float(lr.coef_[0])
    s1 = slope(df_pre, x)
    # Align columns for post
    b = df_post.copy()
    if x not in b.columns and x in df_pre.columns:
        b[x] = df_pre[x]
    s2 = slope(b, x)
    return float(s2 - s1)
