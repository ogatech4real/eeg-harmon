from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    root: Path
    bids_root: Path
    derivatives: Path
    figures: Path
    reports: Path

def default_paths(root: str | Path = ".") -> Paths:
    r = Path(root).resolve()
    return Paths(
        root=r,
        bids_root=r / "data",
        derivatives=r / "derivatives",
        figures=r / "figures",
        reports=r / "reports",
    )
