import pandas as pd
from src.metrics import site_variance_ratio

def test_site_variance_ratio_shape(tmp_path):
    df = pd.DataFrame({"alpha":[1.0,2.0,1.5,2.2],
                       "site":["A","A","B","B"]})
    r = site_variance_ratio(df.rename(columns={"alpha":"feat"}), ["feat"], "site")
    assert r >= 0.0
