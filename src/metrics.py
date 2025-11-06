from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm

def site_variance_ratio(df: pd.DataFrame, feature_cols: list[str], site_col: str = "site") -> float:
    """Mean ratio of between-site variance / total variance across features."""
    ratios = []
    for c in feature_cols:
        grp = df.groupby(site_col)[c].mean()
        between = grp.var(ddof=1)
        total = df[c].var(ddof=1)
        ratios.append(0.0 if (total is None or total == 0) else float(between / total))
    return float(np.mean(ratios)) if ratios else np.nan

def preservation_delta(pre: pd.DataFrame, post: pd.DataFrame, y: str, x: str) -> float:
    """|Δβ| change in slope for univariate regression y ~ x."""
    Xp = sm.add_constant(pre[x], has_constant="add"); yp = pre[y]
    Xa = sm.add_constant(post[x], has_constant="add"); ya = post[y]
    b0 = sm.OLS(yp, Xp).fit().params.get(x, np.nan)
    b1 = sm.OLS(ya, Xa).fit().params.get(x, np.nan)
    return float(abs(b1 - b0))
