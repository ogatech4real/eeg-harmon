from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import re
import shutil
import pandas as pd
import mne
from mne_bids import BIDSPath, write_raw_bids

@dataclass
class BidsifyResult:
    bids_root: Path
    subject: str
    task: str

# Public API ------------------------------------------------
def bidsify(
    input_path: str | Path,
    out_root: str | Path,
    subject: str | None = None,
    task: str | None = None,
    datatype: str = "eeg",
    overwrite: bool = True,
) -> BidsifyResult:
    """
    Convert a single non-BIDS EEG file (.edf/.bdf/.vhdr/.set/.fif/.cnt/.mff)
    into a minimal BIDS dataset under <out_root>/auto_bids/.
    Auto-resamples to 128 Hz and converts to float32 for compact size.
    """
    input_path = Path(input_path)
    out_root = Path(out_root)
    _assert_supported(input_path)

    subj, tsk = _infer_subject_task(input_path.name)
    subject = subject or subj or "01"
    task = task or tsk or "rest"

    raw = _read_raw(input_path)
    # memory-friendly slimming
    if raw.info["sfreq"] > 128:
        raw.resample(128, npad="auto")
    raw.load_data()
    raw.apply_function(lambda x: x.astype("float32"))
    raw.set_eeg_reference("average", projection=False)
    raw.filter(1.0, 40.0, fir_design="firwin", verbose=False)

    bids_root = out_root / "auto_bids"
    if overwrite and bids_root.exists():
        shutil.rmtree(bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)

    bp = BIDSPath(root=bids_root, subject=subject, task=task, datatype=datatype, suffix="eeg")
    write_raw_bids(raw, bp, overwrite=True, format="FIF", verbose=False)
    _ensure_participants(bids_root, subject)
    return BidsifyResult(bids_root=bids_root, subject=subject, task=task)

def is_eeg_file(p: Path) -> bool:
    return p.suffix.lower() in {".edf", ".bdf", ".vhdr", ".set", ".fif", ".cnt", ".mff"}

# Helpers ---------------------------------------------------
def _assert_supported(p: Path):
    if p.suffix.lower() == ".vhdr":
        for ext in (".vmrk", ".eeg"):
            if not p.with_suffix(ext).exists():
                raise FileNotFoundError(f"BrainVision header found but companion '{ext}' is missing.")
    exts = {".edf", ".bdf", ".vhdr", ".set", ".fif", ".cnt", ".mff"}
    if p.suffix.lower() not in exts:
        raise ValueError(f"Unsupported EEG format: '{p.suffix}'. Supported: {sorted(exts)}")

def _read_raw(p: Path) -> mne.io.BaseRaw:
    sfx = p.suffix.lower()
    if sfx == ".edf":
        return mne.io.read_raw_edf(p, preload=False, verbose=False)
    if sfx == ".bdf":
        return mne.io.read_raw_bdf(p, preload=False, verbose=False)
    if sfx == ".vhdr":
        return mne.io.read_raw_brainvision(p, preload=False, verbose=False)
    if sfx == ".set":
        return mne.io.read_raw_eeglab(p, preload=False, verbose=False)
    if sfx == ".fif":
        return mne.io.read_raw_fif(p, preload=False, verbose=False)
    if sfx == ".cnt":
        return mne.io.read_raw_cnt(p, preload=False, verbose=False)
    if sfx == ".mff":
        return mne.io.read_raw_egi(p, preload=False, verbose=False)
    raise ValueError(f"Cannot read file: {p}")

def _infer_subject_task(filename: str) -> tuple[str | None, str | None]:
    m = re.search(r"sub-([A-Za-z0-9]+)", filename)
    subject = m.group(1) if m else None
    m = re.search(r"task-([A-Za-z0-9]+)", filename)
    task = m.group(1) if m else None
    if task is None:
        for t in ("rest", "rs", "oddball", "p300", "visual", "auditory"):
            if re.search(rf"\b{t}\b", filename.lower()):
                task = "rest" if t in ("rest", "rs") else t
                break
    if subject is None:
        m = re.search(r"(?:sub|s|p|id|pt)[-_]?(\d{1,3})", filename.lower())
        if m:
            subject = m.group(1)
    return subject, task

def _ensure_participants(bids_root: Path, subject: str):
    part_file = bids_root / "participants.tsv"
    if part_file.exists():
        try:
            df = pd.read_csv(part_file, sep="\t")
            if "site" not in df.columns:
                df["site"] = ""
            if (df["participant_id"] == f"sub-{subject}").sum() == 0:
                df = pd.concat([df, pd.DataFrame([{"participant_id": f"sub-{subject}", "site": ""}])], ignore_index=True)
            df.to_csv(part_file, sep="\t", index=False)
            return
        except Exception:
            pass
    df = pd.DataFrame([{"participant_id": f"sub-{subject}", "age": "", "sex": "", "site": ""}])
    df.to_csv(part_file, sep="\t", index=False)
