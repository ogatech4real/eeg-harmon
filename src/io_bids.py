from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import mne
from mne_bids import BIDSPath, read_raw_bids

def validate_bids_root(bids_root: str | Path) -> dict:
    p = Path(bids_root)
    return {
        "is_bids": (p / "dataset_description.json").exists(),
        "root": str(p),
    }

def discover_subject_task(bids_root: str | Path) -> tuple[str, str]:
    root = Path(bids_root)
    subs = sorted([d.name.replace("sub-", "") for d in root.glob("sub-*") if d.is_dir()])
    if not subs:
        raise ValueError("No subjects found in BIDS root.")
    subject = subs[0]
    eeg_dir = root / f"sub-{subject}" / "eeg"
    tasks = []
    for f in eeg_dir.glob(f"sub-{subject}_*eeg.*"):
        # parse task-<name>
        parts = f.name.split("_")
        for part in parts:
            if part.startswith("task-"):
                tasks.append(part.replace("task-", ""))
    task = tasks[0] if tasks else "rest"
    return subject, task

def load_raw(bids_root: str | Path, subject: str, task: str) -> mne.io.BaseRaw:
    bids_path = BIDSPath(root=bids_root, subject=subject, task=task, datatype="eeg", suffix="eeg")
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    return raw

def get_per_epoch_site_vector(bids_root: str | Path, subject: str, n_epochs: int) -> list[str] | None:
    # read participants.tsv, map subject->site
    part = Path(bids_root) / "participants.tsv"
    if not part.exists():
        return None
    df = pd.read_csv(part, sep="\t")
    pid = f"sub-{subject}"
    row = df.loc[df["participant_id"] == pid]
    if row.empty or "site" not in df.columns:
        return None
    site = str(row["site"].iloc[0] or "").strip() or "siteA"
    return [site] * n_epochs
