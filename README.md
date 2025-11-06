\# EEG Harmonisation – One-Click Runner



\## Quickstart

1\. `python -m venv venv \&\& source venv/bin/activate`

2\. `pip install -r requirements.txt`

3\. `streamlit run app.py`



\## Use

\- Upload a BIDS EEG `.zip` \*\*or\*\* point to a local BIDS folder.

\- Enter subject/task (e.g., `01` / `rest`).

\- Click \*\*Run Harmonisation\*\*.

\- Outputs land in:

&nbsp; - `derivatives/` (features, harmonised artifacts)

&nbsp; - `figures/` (plots)

&nbsp; - `reports/` (`eval\_summary.json`, error logs if any)



\## Notes

\- Keep datasets out of Git.

\- All logic lives in `src/`. `app.py` is UI only.



