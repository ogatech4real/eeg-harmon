import mne
from typing import Tuple, Optional

def basic_preproc(raw: mne.io.BaseRaw, epoch_event_id: Optional[dict] = None) -> Tuple[mne.Epochs | mne.io.BaseRaw, dict]:
    """Bandpass, notch, average-ref; return epochs if events present else Raw."""
    raw = raw.copy().load_data()
    raw.filter(l_freq=0.5, h_freq=40.0, fir_design="firwin")
    try:
        raw.notch_filter(freqs=[50.0])
    except Exception:
        pass
    raw.set_eeg_reference("average", projection=True)

    if epoch_event_id:
        events = mne.find_events(raw, verbose=False)
        epochs = mne.Epochs(raw, events, event_id=epoch_event_id, tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True)
        return epochs, {"events": len(events)}
    else:
        return raw, {}
