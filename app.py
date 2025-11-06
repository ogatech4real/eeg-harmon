import streamlit as st
from pathlib import Path
import tempfile, shutil

from main import run_pipeline, PipelineError
from src.io_bids import validate_bids_root, discover_subject_task

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
            v = validate_bids_root(p)
            if v["is_bids"]:
                try:
                    s, t = discover_subject_task(p)
                    st.success(f"Detected BIDS. Example subject/task: {s}/{t}")
                except Exception:
                    st.info("BIDS detected. Subject/task will be auto-discovered at run time.")
            else:
                st.warning("Folder found but not a valid BIDS root (missing dataset_description.json).")
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
