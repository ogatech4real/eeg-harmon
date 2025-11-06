from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids, get_subjects, get_tasks

# --- Core loader --------------------------------------------------------------
def load_raw(bids_root: str | Path, subject: str, task: str):
    """Load a BIDS EEG Raw using MNE-BIDS."""
    bids_root = Path(bids_root)
    bp = BIDSPath(subject=subject, task=task, datatype="eeg", root=bids_root)
    raw = read_raw_bids(bp, verbose=False)
    return raw

# --- Convenience: auto-discovery & validation ---------------------------------
def discover_subject_task(bids_root: str | Path) -> Tuple[str, str]:
    """Pick the first available subject and task (default to 'rest' if task list is empty)."""
    subs = get_subjects(bids_root)
    if not subs:
        raise FileNotFoundError("No subjects found in BIDS root.")
    tasks = get_tasks(bids_root, subject=subs[0])
    task = tasks[0] if tasks else "rest"
    return subs[0], task

def validate_bids_root(bids_root: str | Path) -> dict:
    """Shallow validation that a BIDS dataset is present."""
    p = Path(bids_root)
    ok = (p / "dataset_description.json").exists()
    return {"bids_root": str(p), "is_bids": ok}

# --- Optional: site labels from metadata --------------------------------------
def get_site_label_for_subject(bids_root: str | Path, subject: str) -> Optional[str]:
    """
    Try to infer a site label for a subject from participants.tsv (column 'site').
    Returns None if not present; the pipeline will fall back to a stub.
    """
    p = Path(bids_root) / "participants.tsv"
    if p.exists():
        df = pd.read_csv(p, sep="\t")
        if {"participant_id", "site"} <= set(df.columns):
            pid = f"sub-{subject}"
            row = df[df["participant_id"] == pid]
            if not row.empty:
                return str(row["site"].iloc[0])
    return None

def get_per_epoch_site_vector(bids_root: str | Path, subject: str, n_epochs: int) -> List[str] | None:
    """
    If you have per-recording site info elsewhere (e.g., scans.tsv/acq), wire it here.
    For now we return a single site label replicated for all epochs if available.
    """
    site = get_site_label_for_subject(bids_root, subject)
    if site is None:
        return None
    return [site] * int(n_epochs)
