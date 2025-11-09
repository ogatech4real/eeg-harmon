import streamlit as st
from pathlib import Path
import tempfile
import shutil
import json

# Defer heavy imports so UI renders even if deps fail
run_pipeline = None
PipelineError = Exception
_import_error = None
try:
    from main import run_pipeline as _run_pipeline, PipelineError as _PipelineError
    run_pipeline = _run_pipeline
    PipelineError = _PipelineError
except Exception as e:
    _import_error = e

st.set_page_config(page_title="EEG Harmonisation", layout="centered")
st.title("EEG Harmonisation – One-Click Runner")
st.caption("Upload a BIDS EEG ZIP or a single EEG file (EDF/BDF/BrainVision/EEGLAB/FIF). "
           "Or provide a URL (HTTP/S3 presigned). Click **Run** to generate outputs.")

# --------------------- Ingest Controls ---------------------
tab_zip, tab_file, tab_url = st.tabs(["Upload ZIP", "Upload File", "From URL"])

zip_tmp = None
file_tmp = None
url_val = None

with tab_zip:
    up = st.file_uploader("BIDS ZIP (≤200MB on Cloud recommended)", type=["zip"])
    if up is not None:
        tmpdir = tempfile.mkdtemp()
        zip_tmp = str(Path(tmpdir) / up.name)
        with open(zip_tmp, "wb") as f:
            f.write(up.read())
        st.success(f"Staged: {up.name}")

with tab_file:
    raw = st.file_uploader("Single EEG file (.edf/.bdf/.vhdr/.set/.fif/.cnt/.mff)", type=[
        "edf", "bdf", "vhdr", "set", "fif", "cnt", "mff"
    ])
    if raw is not None:
        tmpdir = tempfile.mkdtemp()
        file_tmp = str(Path(tmpdir) / raw.name)
        with open(file_tmp, "wb") as f:
            f.write(raw.read())
        st.success(f"Staged: {raw.name}")

with tab_url:
    url_val = st.text_input("HTTP/S3 URL to a ZIP or single EEG file")

# --------------------- Run Parameters ---------------------
st.subheader("Run Settings")
col1, col2 = st.columns(2)
with col1:
    include_erp = st.checkbox("Include ERP", value=True)
    include_csd = st.checkbox("Include Riemannian CSD (heavier)", value=False)
with col2:
    target_sfreq = st.number_input("Target sampling rate (Hz)", 64, 512, 128, step=64)
    max_seconds = st.number_input("Max seconds per file (crop)", 60, 3600, 300, step=60)

subject_override = st.text_input("Subject (optional override)", "")
task_override = st.text_input("Task (optional override)", "")

outroot = st.text_input("Output root (server-side path)", "./outputs")

run_btn = st.button("Run Harmonisation", type="primary")

# --------------------- Execution --------------------------
if run_btn:
    if run_pipeline is None:
        st.error("Core modules not loaded. Likely a missing package (e.g., 'mne'). "
                 "See Diagnostics below and fix requirements.")
        st.stop()

    src = zip_tmp or file_tmp or url_val
    if not src:
        st.error("Provide a ZIP, a single EEG file, or a URL.")
        st.stop()

    try:
        summary = run_pipeline(
            dataset_path=src,
            subject=subject_override or None,
            task=task_override or None,
            out_root=outroot,
            include_erp=include_erp,
            include_csd=include_csd,
            target_sfreq=int(target_sfreq),
            max_seconds=int(max_seconds),
        )
        st.success("Done.")
        st.json(summary)

        md_path = Path(summary["outputs"]["report_markdown"])
        if md_path.exists():
            st.download_button("Download one-pager (Markdown)",
                               data=md_path.read_text(),
                               file_name=md_path.name,
                               mime="text/markdown")

        bundle = Path(summary["outputs"]["results_bundle_zip"])
        if bundle.exists():
            st.download_button("Download results bundle (ZIP)",
                               data=bundle.read_bytes(),
                               file_name=bundle.name,
                               mime="application/zip")

        # Quick links to artifacts
        with st.expander("Artifacts"):
            for k, v in summary["outputs"].items():
                st.write(f"**{k}** → `{v}`")

    except PipelineError as e:
        st.error(f"Pipeline failed: {e}")
        logf = Path(outroot) / "reports" / "error.log"
        if logf.exists():
            with st.expander("Error log"):
                st.code(logf.read_text())

# --------------------- Diagnostics ------------------------
st.divider()
with st.expander("Diagnostics"):
    import sys, platform, pkgutil
    st.write("Python:", sys.version)
    st.write("Platform:", platform.platform())
    if _import_error:
        st.warning(f"Deferred import error: {type(_import_error).__name__}: {_import_error}")
    try:
        import mne
        st.write("MNE version:", mne.__version__)
    except Exception as e:
        st.error(f"MNE import failed: {e}")
    try:
        import numpy, scipy, pandas
        st.write("NumPy/Scipy/Pandas:", numpy.__version__, scipy.__version__, pandas.__version__)
    except Exception as e:
        st.warning(f"Scientific stack check: {e}")
    try:
        import pyriemann
        st.write("pyriemann version:", pyriemann.__version__)
    except Exception as e:
        st.warning(f"pyriemann check: {e}")
    st.caption("Installed (subset):")
    pkgs = sorted([m.name for m in pkgutil.iter_modules()])
    st.code(", ".join(x for x in pkgs if x.lower().startswith((
        "mne", "numpy", "scipy", "pandas", "pyriemann", "neuro", "streamlit"))))
