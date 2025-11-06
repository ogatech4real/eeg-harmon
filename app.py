*** a/app.py
--- b/app.py
@@
-import streamlit as st
-from pathlib import Path
-import tempfile, shutil
-
-from main import run_pipeline, PipelineError
+import streamlit as st
+from pathlib import Path
+import tempfile, shutil
+
+# Defer heavy imports so the UI can render even if deps failed on Streamlit Cloud
+run_pipeline = None
+PipelineError = Exception
+_import_error = None
+try:
+    from main import run_pipeline as _run_pipeline, PipelineError as _PipelineError
+    run_pipeline = _run_pipeline
+    PipelineError = _PipelineError
+except Exception as e:
+    _import_error = e
 
 st.set_page_config(page_title="EEG Harmonisation", layout="centered")
 st.title("EEG Harmonisation – One-Click Runner")
 
 st.caption("Upload a BIDS EEG ZIP or point at a local BIDS folder. Click **Run** to generate outputs.")
@@
 run_btn = st.button("Run Harmonisation", type="primary")
 
 if run_btn:
-    if not (zip_tmp or dataset_dir):
+    if run_pipeline is None:
+        st.error("Core modules not loaded. This usually means a missing package (e.g., 'mne'). "
+                 "See 'Diagnostics' below and reboot the app after fixing requirements.")
+        st.stop()
+    if not (zip_tmp or dataset_dir):
         st.error("Select a ZIP or a valid folder.")
         st.stop()
@@
     except PipelineError as e:
         st.error(f"Pipeline failed: {e}")
         logf = Path(outroot) / "reports" / "error.log"
         if logf.exists():
             with st.expander("Error log"):
                 st.code(logf.read_text())
+
+# --- Diagnostics (always available) -------------------------------------------
+st.divider()
+with st.expander("Diagnostics"):
+    import sys, platform, pkgutil
+    st.write("Python:", sys.version)
+    st.write("Platform:", platform.platform())
+    if _import_error:
+        st.warning(f"Deferred import error: {type(_import_error).__name__}: {_import_error}")
+    try:
+        import mne
+        st.write("MNE version:", mne.__version__)
+    except Exception as e:
+        st.error(f"MNE import failed: {e}")
+    try:
+        import numpy, scipy, pandas
+        st.write("NumPy/Scipy/Pandas:", numpy.__version__, scipy.__version__, pandas.__version__)
+    except Exception as e:
+        st.warning(f"Scientific stack check: {e}")
+    st.caption("Installed packages (subset):")
+    pkgs = sorted([m.name for m in pkgutil.iter_modules()])
+    st.code(", ".join(x for x in pkgs if x.lower().startswith(("mne","numpy","scipy","pandas","pyriemann","neuro","streamlit"))))
