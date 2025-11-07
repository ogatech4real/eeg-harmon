# src/io_bids.py
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import mne
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals


def validate_bids_root(bids_root: Path | str) -> dict:
    """Lightweight BIDS root validation."""
    bids_root = Path(bids_root)
    dd = bids_root / "dataset_description.json"
    return {
        "root": str(bids_root.resolve()),
        "is_bids": dd.exists(),
        "has_participants": (bids_root / "participants.tsv").exists(),
    }


def discover_subject_task(bids_root: Path | str) -> tuple[str | None, str | None]:
    """
    Return the first available subject and task in the dataset.
    Uses mne-bids 0.15 API: get_entity_vals(bids_root, entity_key=...).
    """
    bids_root = Path(bids_root)
    try:
        subjects = get_entity_vals(bids_root=bids_root, entity_key="subject") or []
    except TypeError:
        # Fallback if signature differs
        subjects = get_entity_vals(bids_root, "subject") or []

    try:
        tasks = get_entity_vals(bids_root=bids_root, entity_key="task") or []
    except TypeError:
        tasks = get_entity_vals(bids_root, "task") or []

    subj = subjects[0] if len(subjects) else None
    task = tasks[0] if len(tasks) else None
    return subj, task


def load_raw(
    bids_root: Path | str,
    subject: str,
    task: str,
    session: str | None = None,
    run: str | int | None = None,
    datatype: str = "eeg",
) -> mne.io.BaseRaw:
    """
    Load a single Raw using BIDSPath + read_raw_bids.
    Assumes EEG; extend as needed for MEG/iEEG.
    """
    bids_root = Path(bids_root)
    bp = BIDSPath(
        root=bids_root,
        subject=str(subject),
        task=str(task),
        session=str(session) if session else None,
        run=str(run) if run is not None else None,
        datatype=datatype,
        suffix="eeg",
    )
    raw = read_raw_bids(bp, verbose=False)
    return raw


def get_per_epoch_site_vector(
    bids_root: Path | str,
    subject: str,
    n_epochs: int,
    site_column_candidates: list[str] | None = None,
) -> list[str] | None:
    """
    Heuristic site label retrieval.
    - Look for participants.tsv → row for subject → any of ['site','site_id','siteID','center','lab'].
    - If found, repeat per epoch. If not, return None.
    """
    bids_root = Path(bids_root)
    site_column_candidates = site_column_candidates or ["site", "site_id", "siteID", "center", "lab"]

    pfile = bids_root / "participants.tsv"
    if not pfile.exists():
        return None

    try:
        dfp = pd.read_csv(pfile, sep="\t")
    except Exception:
        return None

    # Standard BIDS key is "participant_id" like "sub-01"
    subj_keys = {"participant_id", "participant", "subject", "sub"}
    subj_col = next((c for c in dfp.columns if c in subj_keys), None)
    if subj_col is None:
        # Try to coerce participant_id if absent
        subj_col = "participant_id"

    # Normalise subject identifiers (with or without "sub-")
    def _norm(s: str) -> str:
        s = str(s)
        return s if s.startswith("sub-") else f"sub-{s}"

    try:
        row = dfp.loc[dfp[subj_col].astype(str).map(_norm) == _norm(subject)]
    except Exception:
        # If subject column missing, just take the first row
        row = dfp.iloc[:1]

    if row.empty:
        return None

    site_val = None
    for cand in site_column_candidates:
        if cand in dfp.columns:
            sv = row.iloc[0][cand]
            if pd.notna(sv) and str(sv).strip():
                site_val = str(sv).strip()
                break

    if site_val is None:
        return None

    return [site_val] * int(n_epochs)
