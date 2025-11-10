from __future__ import annotations
from pathlib import Path
import shutil
import zipfile
import json
import traceback
import time
import sys
import platform
import tempfile
import requests
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import mne

from src.config import default_paths
from src.io_bids import (
    load_raw,
    discover_subject_task,
    validate_bids_root,
    get_per_epoch_site_vector,
)
from src.preproc import basic_preproc
from src.features import bandpowers, cross_spectra, erp_peaks
from src.harmonize import combat_vector_features, combat_riemann
from src.metrics import site_variance_ratio, preservation_delta
from src.reporting import write_markdown, create_results_bundle
from src.utils.bidsify import bidsify, is_eeg_file

class PipelineError(Exception):
    ...

# ---------------- utilities ----------------
def _unzip_if_needed(upload: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if upload.suffix.lower() == ".zip":
        target = dest_dir / upload.stem
        if target.exists():
            shutil.rmtree(target)
        with zipfile.ZipFile(upload, "r") as z:
            z.extractall(target)
        return target
    return upload

def _download_stream(url: str, out_path: Path, chunk=8 * 1024 * 1024) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for cb in r.iter_content(chunk_size=chunk):
                if cb:
                    f.write(cb)
    return out_path

def _write_run_meta(subject: str, task: str, out_dir: Path):
    meta = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "subject": subject,
        "task": task,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

def _is_zip_download(url: str) -> bool:
    """
    Heuristics to decide if a URL returns a ZIP even if it has no .zip suffix.
    - Handles OpenNeuro snapshot/versions /download endpoints.
    - Falls back to HEAD Content-Type.
    """
    u = urlparse(url)
    host = (u.netloc or "").lower()
    path = (u.path or "").lower()

    # OpenNeuro patterns (both legacy and current)
    if "openneuro.org" in host and path.endswith("/download"):
        # e.g., /crn/datasets/dsXXXXXX/snapshots/1.0.0/download
        # or    /datasets/dsXXXXXX/versions/1.0.0/download
        return True

    # If it actually ends with .zip
    if path.endswith(".zip"):
        return True

    # HEAD check
    try:
        r = requests.head(url, allow_redirects=True, timeout=20)
        ct = (r.headers.get("Content-Type") or "").lower()
        # Some servers report generic octet-stream; treat as zip for dataset endpoints
        if "zip" in ct or "application/x-zip-compressed" in ct:
            return True
        if "application/octet-stream" in ct and path.endswith("/download"):
            return True
    except Exception:
        # If HEAD fails, be conservative (not a zip) unless /download
        return path.endswith("/download")

    return False

# --------------- orchestrator ---------------
def run_pipeline(
    dataset_path: str,
    subject: str | None = None,
    task: str | None = None,
    out_root: str = ".",
    include_erp: bool = True,
    include_csd: bool = False,
    target_sfreq: int = 128,
    max_seconds: int = 300,
) -> dict:
    """
    End-to-end run with robust ingest:
    - dataset_path: BIDS folder OR .zip OR URL OR a single EEG file path
    - subject/task: optional overrides; otherwise auto-discover/infer
    - outputs: derivatives/, figures/, reports/, plus results bundle ZIP
    """
    P = default_paths(out_root)
    for d in [P.bids_root, P.derivatives, P.figures, P.reports]:
        Path(d).mkdir(parents=True, exist_ok=True)

    try:
        # 0) Ingest: URL / File / ZIP / Folder
        ds_path = None
        sp = Path(dataset_path)

        if dataset_path.startswith(("http://", "https://")):
            tmp_dir = Path(tempfile.mkdtemp())
            if _is_zip_download(dataset_path):
                # Stream to temp zip, then unzip to BIDS
                local_zip = tmp_dir / "dataset.zip"
                _download_stream(dataset_path, local_zip)
                ds_path = _unzip_if_needed(local_zip, P.bids_root)
            else:
                # Assume a single EEG file → bidsify
                local_file = tmp_dir / "remote_input"
                _download_stream(dataset_path, local_file)
                res = bidsify(local_file, P.bids_root, subject=subject, task=task)
                ds_path, subject, task = res.bids_root, res.subject, res.task

        elif sp.is_file():
            if sp.suffix.lower() == ".zip":
                ds_path = _unzip_if_needed(sp, P.bids_root)
            elif is_eeg_file(sp):
                res = bidsify(sp, P.bids_root, subject=subject, task=task)
                ds_path, subject, task = res.bids_root, res.subject, res.task
            else:
                raise PipelineError(f"Unsupported file: {sp.name}")

        elif sp.is_dir():
            ds_path = sp
        else:
            raise PipelineError("Input not found or unsupported.")

        # 1) Validate BIDS
        v = validate_bids_root(ds_path)
        if not v["is_bids"]:
            raise PipelineError("Dataset is not BIDS; auto-BIDSify did not succeed.")

        # 2) Auto-discover subject/task if not provided
        if not subject or not task:
            dsub, dtask = discover_subject_task(ds_path)
            subject = subject or dsub
            task = task or dtask

        # Record run metadata + params
        _write_run_meta(subject, task, Path(P.reports))
        params = {
            "include_erp": include_erp,
            "include_csd": include_csd,
            "target_sfreq": target_sfreq,
            "max_seconds": max_seconds,
        }
        (Path(P.reports) / "params.json").write_text(json.dumps(params, indent=2))

        # 3) Load & preprocess → epochs (fixed-length if no events)
        raw = load_raw(ds_path, subject=subject, task=task)
        epochs, info = basic_preproc(
            raw,
            target_sfreq=target_sfreq,
            max_seconds=max_seconds,
        )
        if isinstance(epochs, mne.io.BaseRaw):
            epochs = mne.make_fixed_length_epochs(epochs, duration=2.0, preload=True)

        n_epochs = len(epochs)

        # 4) Spectral features (vector)
        bands = {"alpha": (8, 12), "beta": (13, 30)}
        df_bp = bandpowers(epochs, bands)  # columns: epoch, band, value
        df_wide = df_bp.pivot(index="epoch", columns="band", values="value").reset_index()

        # Site labels
        site_vec = get_per_epoch_site_vector(ds_path, subject, n_epochs)
        if site_vec is None or len(site_vec) != n_epochs:
            site_vec = ["siteA"] * n_epochs
            site_note = "Site labels not found; used fallback."
        else:
            site_note = "Site labels inferred from participants.tsv."
        df_wide["site"] = site_vec

        # KPI pre
        feat = "alpha" if "alpha" in df_wide.columns else [c for c in df_wide.columns if c not in {"epoch", "site"}][0]
        pre_ratio = site_variance_ratio(df_wide.rename(columns={feat: "feat"}), ["feat"], "site")

        # 5) Vector ComBat Harmonisation
        df_h, _ = combat_vector_features(
            df_wide.rename(columns={feat: "feat"}), feature_cols=["feat"], batch_col="site", covars=[]
        )
        post_ratio = site_variance_ratio(df_h, ["feat"], "site")

        # Save vector features
        out_feat = Path(P.derivatives / "features" / "spectral.parquet")
        out_feat.parent.mkdir(parents=True, exist_ok=True)
        df_wide.to_parquet(out_feat, index=False)

        out_feat_h = Path(P.derivatives / "harmonized" / "features_harmonized_combat.parquet")
        out_feat_h.parent.mkdir(parents=True, exist_ok=True)
        df_h.to_parquet(out_feat_h, index=False)

        # 6) CSD (Riemannian) — optional
        csd_outputs = {}
        if include_csd:
            Cs = cross_spectra(epochs, fmin=8.0, fmax=30.0, method="multitaper")
            if len(set(site_vec)) == 1:
                batch = ["siteA" if (i % 2 == 0) else "siteB" for i in range(n_epochs)]
            else:
                batch = site_vec
            Cs_h = combat_riemann(Cs, batch=batch, covars=None)
            out_csd_pre = Path(P.derivatives / "harmonized" / "csd" / "csd_pre.npy")
            out_csd_post = Path(P.derivatives / "harmonized" / "csd" / "csd_post_harmonized.npy")
            out_csd_pre.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_csd_pre, np.stack(Cs)); np.save(out_csd_post, np.stack(Cs_h))
            csd_outputs = {"csd_pre": str(out_csd_pre), "csd_post": str(out_csd_post)}

        # 7) ERP (optional; safe if no events—GFP peak fallback in erp_peaks)
        erp_outputs = {}
        if include_erp:
            erp_df = erp_peaks(epochs, tmin=0.25, tmax=0.45)
            out_erp = Path(P.derivatives / "features" / "erp.parquet")
            out_erp.parent.mkdir(parents=True, exist_ok=True)
            erp_df.to_parquet(out_erp, index=False)
            erp_outputs = {"erp": str(out_erp)}

        # 8) Summary + reporting + bundle
        summary = {
            "dataset_root": str(ds_path),
            "subject": subject,
            "task": task,
            "vector_feature": feat,
            "site_info": site_note,
            "site_variance_ratio_pre": float(pre_ratio),
            "site_variance_ratio_post": float(post_ratio),
            "outputs": {
                "spectral": str(out_feat),
                "spectral_harmonized": str(out_feat_h),
                **csd_outputs,
                **erp_outputs,
            },
        }

        # Write report files
        eval_json = Path(P.reports / "eval_summary.json")
        eval_md = Path(P.reports / "eval_summary.md")
        eval_json.write_text(json.dumps(summary, indent=2))
        write_markdown(summary, eval_md)

        # Results bundle ZIP
        bundle_zip = create_results_bundle(P)
        summary["outputs"]["report_markdown"] = str(eval_md)
        summary["outputs"]["report_json"] = str(eval_json)
        summary["outputs"]["results_bundle_zip"] = str(bundle_zip)

        return summary

    except Exception as e:
        Path(P.reports).mkdir(parents=True, exist_ok=True)
        Path(P.reports / "error.log").write_text(traceback.format_exc())
        raise PipelineError(str(e)) from e

if __name__ == "__main__":
    print(json.dumps(run_pipeline("./data/ds_example"), indent=2))
