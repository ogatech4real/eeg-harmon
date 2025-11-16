# EEG Harmonisation Platform

A production‑grade, automated EEG harmonisation pipeline designed to standardise diverse EEG datasets from any acquisition site, structure, or format. This project integrates **BIDS ingestion**, **preprocessing**, **spectral feature extraction**, **ERP metrics**, **vector and Riemannian ComBat harmonisation**, and **interactive analytics dashboards**.

Developed as part of ongoing research by **Adewale et al.**

---

## 🚀 Key Features

* **Universal EEG ingestion** (ZIP, URL, single files), with automatic BIDSification.
* "Super‑robust" recursive discovery across **multi‑level folders / non‑standard sub‑labels** (e.g., `sub-hc3`, `sub-pd28`).
* Automated **preprocessing & epoching** with configurable sampling rate, reference, and duration limits.
* **Spectral feature extraction** (alpha, beta, custom bands).
* **ERP component detection** (P3b, N2, N170, etc.).
* **Vector ComBat** & **Riemannian ComBat** harmonisation.
* Automated **KPI evaluation**:

  * Site‑variance ratio
  * Site‑leakage AUC
  * ERP integrity gates
* **Rich visual analytics**:

  * Feature histograms by site
  * PCA site‑mixing plots
  * Feature drift tables
  * ERP previews
* **Clean results export** including parquet files, CSD matrices, KPIs, Markdown report, and ZIP bundle.
* Streamlit‑based UI for non‑technical users.

---

## 📦 Project Structure

```
├── app.py                 # Streamlit UI
├── main.py                # Core pipeline orchestrator
├── configs/               # YAML configuration profiles
├── data/                  # User‑provided data (optional local use)
├── bids/                  # BIDSification outputs
├── outputs/               # All pipeline outputs
│   ├── bids/              # BIDS‑organised dataset
│   ├── derivatives/       # Features, harmonised outputs, CSD
│   ├── figures/           # Analysis visuals
│   └── reports/           # JSON + Markdown report bundle
├── src/
│   ├── preproc.py         # Filtering, epoching, referencing
│   ├── features.py        # Bandpower + ERP extraction
│   ├── harmonize.py       # Vector & Riemannian ComBat
│   ├── metrics.py         # Site variance, leakage, ERP gates
│   ├── io_bids.py         # Deep BIDS traversal + loading
│   ├── reporting.py       # Markdown report builder
│   └── viz.py             # Visualisation utilities
```

---

## 🧠 How the Pipeline Works

### 1️⃣ Ingestion & Validation

* Accepts ZIP, EEG file, local folder, or URL.
* If needed, auto‑downloads from Google Drive / OpenNeuro.
* Converts inputs to BIDS using `bidsify()`.
* Recursively detects subjects / sessions / tasks.

### 2️⃣ Preprocessing

* Re‑references to **REST** or **average**.
* Resamples to target frequency.
* Epochs signals (fixed‑length if no events).

### 3️⃣ Feature Extraction

* Calculates alpha, beta, and custom spectral bands.
* Extracts ERP peaks if enabled.
* Generates site labels using intelligent inspection of `participants.tsv`.

### 4️⃣ Harmonisation

* **Vector ComBat** for spectral features.
* **Riemannian ComBat** for covariance/CSD matrices.
* Corrects site bias and scanner variance.

### 5️⃣ Quality Metrics (KPIs)

* **Site Variance Ratio** (pre/post)
* **Site Leakage AUC** (how predictable site is)
* **ERP Integrity Gates** (detect over‑correction)

### 6️⃣ Visual Analytics

All visualisations rendered in‑app + saved to `/outputs/figures`:

* PCA site‑mixing plots
* Histograms before/after harmonisation
* Feature drift tables
* ERP metric preview tables

### 7️⃣ Export Bundle

Automatically generates:

* `spectral.parquet` (pre)
* `features_harmonized_combat.parquet` (post)
* `erp.parquet`
* `csd_pre.npy`, `csd_post_harmonized.npy`
* Markdown report
* Summary JSON
* ZIP archive of all artifacts

---

## 🖥️ Running Locally

### Prerequisites

* Python 3.10+
* MNE, Streamlit, neuroHarmonize, PyRiemann

Install dependencies:

```bash\pip install -r requirements.txt
```

Launch the app:

```bash
streamlit run app.py
```

Upload a dataset (ZIP or folder) and click **Run Harmonisation**.

---

## 🌐 Deployment

Recommended deployment strategies:

* **Google Cloud Run** (serverless + GPUs)
* **AWS ECS / Fargate** for scalable workloads
* **Azure App Service**
* Self‑hosted via **Docker** for enterprise control

Suggested Docker entrypoint: `streamlit run app.py --server.port=8080`.

---

## 📊 Outputs & Interpretation

### Harmonised spectral features

Use these for ML models, biomarker discovery, clustering, or group comparisons.

### PCA & Histograms

Visual proof that site bias has been suppressed.

### ERP Metrics

Assesses signal integrity and confirms that harmonisation did **not** distort cognitive components.

### KPIs

* **Lower site variance ratio** → improved dataset consistency.
* **Lower AUC** → site less predictable → less bias.
* **ERP gates passing** → harmonisation preserved neuroscientific meaning.

---

## 📝 Citation

If using this tool in research, cite:
**Adewale et al., EEG Harmonisation Platform (2025)**

---

## 📨 Contact

For support, research collaboration, or enterprise deployment:
**[hello@adewaleogabi.info](mailto:hello@adewaleogabi.info)**
