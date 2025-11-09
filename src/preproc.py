from __future__ import annotations
import mne

def basic_preproc(raw: mne.io.BaseRaw, target_sfreq: int = 128, max_seconds: int = 300):
    """
    Memory-aware default:
    - crop to max_seconds
    - resample to target_sfreq
    - average reference
    - 1–40 Hz band-pass
    - epoch into 2s fixed windows
    """
    # Crop
    if max_seconds:
        raw.crop(tmax=min(raw.times[-1], max_seconds))

    # Resample
    if raw.info["sfreq"] > target_sfreq:
        raw.resample(target_sfreq, npad="auto")

    # Reference and filter
    raw.set_eeg_reference("average", projection=False)
    raw.filter(1.0, 40.0, fir_design="firwin")

    # Fixed-length epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
    info = {"n_epochs": len(epochs), "sfreq": epochs.info["sfreq"], "n_channels": len(epochs.ch_names)}
    return epochs, info
