from __future__ import annotations
import numpy as np
import pandas as pd
import mne

def bandpowers(epochs: mne.Epochs, bands: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Return long-table bandpowers: epoch, band, value (mean across channels)."""
    psds, freqs = mne.time_frequency.psd_welch(epochs, fmin=min(b[0] for b in bands.values()),
                                               fmax=max(b[1] for b in bands.values()), verbose=False)
    psds = 10 * np.log10(psds)  # (n_epochs, n_channels, n_freqs)
    rows = []
    for e in range(psds.shape[0]):
        for name, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            val = psds[e, :, mask].mean()
            rows.append({"epoch": e, "band": name, "value": float(val)})
    return pd.DataFrame(rows)

def cross_spectra(epochs: mne.Epochs, fmin: float, fmax: float, method: str = "multitaper"):
    """Return list of (ch x ch) real SPD-like CSD matrices per epoch."""
    if method == "multitaper":
        csd = mne.time_frequency.csd_multitaper(epochs, fmin=fmin, fmax=fmax, verbose=False)
    else:
        csd = mne.time_frequency.csd_welch(epochs, fmin=fmin, fmax=fmax, verbose=False)
    Cs = []
    for i in range(len(epochs)):
        Ci = csd.get_data(index=i).real
        Ci = (Ci + Ci.T) / 2.0
        Ci += np.eye(Ci.shape[0]) * 1e-9
        Cs.append(Ci)
    return Cs

def erp_peaks(epochs: mne.Epochs, tmin: float, tmax: float) -> pd.DataFrame:
    """Simple GFP-based ERP descriptor around a time window."""
    evk = epochs.average()
    gfp = evk.data.std(axis=0)
    times_ms = evk.times * 1000.0
    mask = (times_ms >= tmin * 1000) & (times_ms <= tmax * 1000)
    peak_idx = int(np.argmax(gfp[mask]))
    peak_ms = float(times_ms[mask][peak_idx])
    peak_amp = float(gfp[mask][peak_idx])
    return pd.DataFrame([{"erp_peak_ms": peak_ms, "erp_peak_amp_grand": peak_amp}])
