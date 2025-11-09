from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    root: Path
    bids_root: Path
    derivatives: Path
    figures: Path
    reports: Path

def default_paths(root: str | Path) -> Paths:
    root = Path(root).resolve()
    return Paths(
        root=root,
        bids_root=root / "bids",
        derivatives=root / "derivatives",
        figures=root / "figures",
        reports=root / "reports",
    )
