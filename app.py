import streamlit as st
from pathlib import Path
import tempfile, shutil

# Defer heavy imports so the UI can render even if deps failed on Streamlit Cloud
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

st.caption("Upload a BIDS EEG ZIP or point at a local BIDS folder. Click **Run** to generate outputs.")

tab1, tab2 = st.tabs(["Upload ZIP", "Use Local Folder"])

zip_tmp = None
dataset_dir = None

with tab1:
    up = st.file_uploader("Upload BIDS dataset (.zip)", type=["zip"])
    if up:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        tmp.write(up.read()); tmp.flush(); tmp.close()
        zip_tmp = tmp.name
        st.success("ZIP uploaded.")

with tab2:
    pstr = st.text_input("Local BIDS folder", value="./data/ds000XYZ")
    if pstr:
        p = Path(pstr).expanduser().resolve()
        if p.exists() and p.is_dir():
            dataset_dir = str(p)
            st.success(f"Using local folder: {dataset_dir}")
        else:
            st.warning("Folder not found.")

col1, col2, col3 = st.columns(3)
subject = col1.text_input("Subject (optional)", "")
task    = col2.text_input("Task (optional)", "")
outroot = col3.text_input("Output root", ".")

st.divider()
c1, c2 = st.columns(2)
include_erp = c1.checkbox("Include ERP harmonisation", value=True)
include_csd = c2.checkbox("Include Riemannian CSD harmonisation", value=True)

run_btn = st.button("Run Harmonisation", type="primary")

if run_btn:
    if run_pipeline is None:
        st.error("Core modules not loaded. This usually means a missing package (e.g., 'mne'). "
                 "See 'Diagnostics' below and reboot the app after fixing requirements.")
        st.stop()
    if not (zip_tmp or dataset_dir):
        st.error("Select a ZIP or a valid folder.")
        st.stop()

    ds_input = dataset_dir
    if zip_tmp:
        data_dir = Path(outroot) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        dest_zip = data_dir / Path(zip_tmp).name
        shutil.move(zip_tmp, dest_zip)
        ds_input = str(dest_zip)

    st.info("Running pipeline…")
    try:
        summary = run_pipeline(
            ds_input,
            subject=subject or None,
            task=task or None,
            out_root=outroot,
            include_erp=include_erp,
            include_csd=include_csd,
        )
        st.success("Done.")
        st.subheader("Summary")
        st.json(summary)

        reports_dir = Path(outroot) / "reports"
        figures_dir = Path(outroot) / "figures"

        # Download buttons
        eval_json = reports_dir / "eval_summary.json"
        eval_md   = reports_dir / "eval_summary.md"
        if eval_json.exists():
            st.download_button("Download evaluation JSON", eval_json.read_bytes(), "eval_summary.json")
        if eval_md.exists():
            st.download_button("Download one-pager (Markdown)", eval_md.read_text(), "eval_summary.md")

        # Show figures
        if figures_dir.exists():
            pngs = list(figures_dir.glob("*.png"))
            if pngs:
                st.subheader("Figures")
                for p in pngs:
                    st.image(str(p), caption=p.name, use_column_width=True)
            else:
                st.caption("No figures generated yet.")
    except PipelineError as e:
        st.error(f"Pipeline failed: {e}")
        logf = Path(outroot) / "reports" / "error.log"
        if logf.exists():
            with st.expander("Error log"):
                st.code(logf.read_text())

# --- Diagnostics (always available) -------------------------------------------
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
        st.info(f"pyriemann import: {e}")
    st.caption("Installed packages (subset):")
    pkgs = sorted([m.name for m in pkgutil.iter_modules()])
    st.code(", ".join(x for x in pkgs if x.lower().startswith(
        ("mne","numpy","scipy","pandas","pyriemann","neuro","streamlit"))))
