from __future__ import annotations
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import psd_array_multitaper

def bandpowers(epochs: mne.Epochs, bands: dict[str, tuple[float, float]]):
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]
    # Compute PSD using multitaper per epoch/channel
    bp_rows = []
    for i, e in enumerate(data):
        psd, freqs = psd_array_multitaper(e, sfreq=sfreq, fmin=1.0, fmax=40.0, adaptive=True, normalization="full", verbose=False)
        # psd shape: (n_channels, n_freqs)
        for band, (fmin, fmax) in bands.items():
            idx = (freqs >= fmin) & (freqs <= fmax)
            val = np.trapz(psd[:, idx], freqs[idx], axis=1).mean()  # mean across channels
            bp_rows.append({"epoch": i, "band": band, "value": float(val)})
    return pd.DataFrame(bp_rows)

def cross_spectra(epochs: mne.Epochs, fmin=8.0, fmax=30.0, method="multitaper"):
    cs_list = []
    csd = mne.time_frequency.csd_multitaper(epochs, fmin=fmin, fmax=fmax, adaptive=True, verbose=False) \
        if method == "multitaper" else mne.time_frequency.csd_fourier(epochs, fmin=fmin, fmax=fmax, verbose=False)
    # Extract per-epoch cross-spectral density matrices at the average frequency
    # MNE returns an average across frequencies by default
    for cs in csd.get_data():
        cs_list.append(cs)
    return cs_list

def erp_peaks(epochs: mne.Epochs, tmin=0.25, tmax=0.45):
    """
    Simple ERP proxy: Global Field Power (GFP) peak amplitude in [tmin, tmax].
    Works even if events are absent by using epoch averages.
    """
    evk = epochs.average()
    gfp = evk.data.std(axis=0)
    times = evk.times
    mask = (times >= tmin) & (times <= tmax)
    if mask.sum() == 0:
        return pd.DataFrame([{"component": "P300_like", "amplitude": float("nan"), "latency": float("nan")}])
    amp = float(np.max(gfp[mask]))
    lat = float(times[mask][np.argmax(gfp[mask])])
    return pd.DataFrame([{"component": "P300_like", "amplitude": amp, "latency": lat}])
