from __future__ import annotations
from pathlib import Path
import json
import zipfile

def write_markdown(summary: dict, out_path: Path):
    lines = []
    lines.append("# Harmonisation Evaluation Summary\n\n")
    lines.append(f"- **Dataset**: `{summary['dataset_root']}`\n")
    lines.append(f"- **Subject/Task**: `{summary['subject']}` / `{summary['task']}`\n")
    pre = summary["site_variance_ratio_pre"]; post = summary["site_variance_ratio_post"]
    delta_pp = (pre - post) * 100.0
    lines.append(f"- **Site-variance ratio**: pre = **{pre:.3f}**, post = **{post:.3f}** (Δ = **{delta_pp:.1f} pp**)\n")
    lines.append("\n## What this means\n")
    lines.append("- Lower post-harmonisation ratio ⇒ reduced non-biological site noise.\n")
    lines.append("- Biological preservation checks (age/condition) are run when covariates exist.\n")
    lines.append("\n## Outputs\n")
    for k, v in summary.get("outputs", {}).items():
        lines.append(f"- **{k}** → `{v}`\n")
    out_path.write_text("".join(lines))

def create_results_bundle(P) -> str:
    """
    Package a thin bundle of results for easy download.
    """
    bundle = Path(P.root) / "results_bundle.zip"
    if bundle.exists():
        bundle.unlink()

    to_include = []
    # reports
    for p in (Path(P.reports) / "eval_summary.json",
              Path(P.reports) / "eval_summary.md",
              Path(P.reports) / "run_meta.json",
              Path(P.reports) / "params.json"):
        if p.exists():
            to_include.append(p)
    # features
    feat_dir = Path(P.derivatives) / "features"
    if feat_dir.exists():
        to_include += list(feat_dir.glob("*.parquet"))
    # harmonized
    harm_dir = Path(P.derivatives) / "harmonized"
    if harm_dir.exists():
        to_include += list(harm_dir.rglob("*.*"))

    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in to_include:
            zf.write(p, arcname=p.relative_to(P.root))
    return str(bundle)
