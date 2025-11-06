import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def boxplot_by_site(df: pd.DataFrame, col: str, out: str | Path):
    fig, ax = plt.subplots(figsize=(6,4))
    df.boxplot(column=col, by="site", ax=ax)
    ax.set_title(f"{col} by site"); ax.figure.suptitle("")
    ax.set_xlabel("Site"); ax.set_ylabel(col)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
