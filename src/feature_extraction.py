import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d

from src.data_loader import BVP_FS

# ── Constants ─────────────────────────────────────────────────────────────────
# Peak detection
MIN_PEAK_DIST = int(0.3 * BVP_FS)   # 0.3s → max ~200 BPM

# HRV frequency bands (Hz)
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)

# IBI resampling rate for frequency-domain analysis
IBI_RESAMPLE_FS = 4.0   # Hz — standard in HRV literature

FEATURE_NAMES = [
    # Time-domain HRV
    "mean_ibi",          # mean inter-beat interval (s)
    "mean_hr",           # mean heart rate (BPM)
    "sdnn",              # std dev of IBI (s)
    "rmssd",             # root mean square of successive IBI differences (s)
    "pnn50",             # fraction of successive IBI diffs > 50 ms
    # Frequency-domain HRV
    "lf_power",          # power in low-frequency band (0.04–0.15 Hz)
    "hf_power",          # power in high-frequency band (0.15–0.40 Hz)
    "lf_hf_ratio",       # LF/HF ratio
    # PPG morphology (statistical features of the raw window)
    "ppg_mean",
    "ppg_std",
    "ppg_skew",
    "ppg_kurtosis",
    # PPG spectral power bands (frequency content of raw PPG)
    "ppg_pow_low",       # power in 0.5–1.0 Hz (resting HR band)
    "ppg_pow_mid",       # power in 1.0–2.0 Hz (normal HR band)
    "ppg_pow_high",      # power in 2.0–4.0 Hz (high HR / harmonics)
    # Peak amplitude features
    "peak_amp_mean",     # mean systolic peak amplitude
    "peak_amp_std",      # std of systolic peak amplitudes
    "pulse_amplitude",   # mean peak-to-trough amplitude (pulse strength)
]

N_FEATURES = len(FEATURE_NAMES)


# ── Peak detection ────────────────────────────────────────────────────────────
def _detect_peaks(window: np.ndarray):
    """
    Detect systolic peaks in a filtered PPG window.

    Uses scipy find_peaks with:
    - minimum distance of 0.3s (≈200 BPM upper limit)
    - prominence = 10% of signal range (lenient enough for wrist PPG,
      which has lower amplitude and more variable morphology than chest PPG)

    The wrist E4 BVP signal is noisier than chest signals, so a lower
    prominence threshold is needed to avoid missing real cardiac peaks.

    Args:
        window: 1-D filtered PPG array (WINDOW_SAMPLES,).

    Returns:
        peak_indices: 1-D int array of sample indices where peaks occur.
    """
    prominence = 0.1 * (window.max() - window.min())
    peaks, _ = find_peaks(window, distance=MIN_PEAK_DIST, prominence=prominence)

    # Sanity check: keep only peaks that produce physiologically valid IBIs
    # Valid HR range: 30–200 BPM → IBI range: 0.3–2.0s
    if len(peaks) >= 2:
        ibis = np.diff(peaks) / BVP_FS
        valid = (ibis >= 0.3) & (ibis <= 2.0)
        # Keep a peak if at least one of its adjacent IBIs is valid
        keep = np.concatenate([[valid[0]], valid[:-1] | valid[1:], [valid[-1]]])
        peaks = peaks[keep]

    return peaks


# ── IBI computation ───────────────────────────────────────────────────────────
def _compute_ibi(peaks: np.ndarray, fs: float = BVP_FS):
    """
    Compute inter-beat intervals from peak indices.

    Args:
        peaks: sorted array of peak sample indices.
        fs:    sampling frequency (Hz).

    Returns:
        ibi:        1-D float array of intervals in seconds.
        ibi_times:  time (s) of each IBI, placed at the second peak.
    """
    ibi = np.diff(peaks) / fs               # seconds
    ibi_times = peaks[1:] / fs              # time of second peak in each pair
    return ibi, ibi_times


# ── Time-domain HRV ──────────────────────────────────────────────────────────
def _time_domain_hrv(ibi: np.ndarray):
    """
    Compute standard time-domain HRV features from an IBI sequence.

    Args:
        ibi: 1-D array of inter-beat intervals in seconds.

    Returns:
        mean_ibi, mean_hr, sdnn, rmssd, pnn50
    """
    mean_ibi = np.mean(ibi)
    mean_hr  = 60.0 / mean_ibi
    sdnn     = np.std(ibi, ddof=1) if len(ibi) >= 2 else np.nan

    if len(ibi) >= 2:
        successive_diffs = np.diff(ibi)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        pnn50 = np.mean(np.abs(successive_diffs) > 0.05)
    else:
        rmssd = np.nan
        pnn50 = np.nan

    return mean_ibi, mean_hr, sdnn, rmssd, pnn50


# ── Frequency-domain HRV ─────────────────────────────────────────────────────
def _freq_domain_hrv(ibi: np.ndarray, ibi_times: np.ndarray):
    """
    Compute LF power, HF power, and LF/HF ratio via Welch's PSD.

    The IBI sequence is unevenly spaced in time, so we first interpolate
    it onto a uniform 4 Hz grid (standard practice in HRV analysis) before
    applying Welch's method.

    Args:
        ibi:       1-D array of IBI values in seconds.
        ibi_times: 1-D array of IBI timestamps in seconds.

    Returns:
        lf_power, hf_power, lf_hf_ratio  (all floats)
        Returns (nan, nan, nan) if there is not enough data.
    """
    nan3 = (np.nan, np.nan, np.nan)

    duration = ibi_times[-1] - ibi_times[0]
    if duration < 1.0 or len(ibi) < 4:
        return nan3

    # Interpolate IBI onto uniform grid
    t_uniform = np.arange(ibi_times[0], ibi_times[-1], 1.0 / IBI_RESAMPLE_FS)
    if len(t_uniform) < 8:
        return nan3

    interpolator = interp1d(ibi_times, ibi, kind="cubic", bounds_error=False,
                            fill_value="extrapolate")
    ibi_uniform = interpolator(t_uniform)

    # Welch PSD
    nperseg = min(len(ibi_uniform), 256)
    freqs, psd = welch(ibi_uniform, fs=IBI_RESAMPLE_FS, nperseg=nperseg)

    freq_res = freqs[1] - freqs[0]

    lf_mask = (freqs >= LF_BAND[0]) & (freqs < LF_BAND[1])
    hf_mask = (freqs >= HF_BAND[0]) & (freqs < HF_BAND[1])

    lf_power = np.sum(psd[lf_mask]) * freq_res
    hf_power = np.sum(psd[hf_mask]) * freq_res

    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan

    return lf_power, hf_power, lf_hf_ratio


# ── PPG morphology features ───────────────────────────────────────────────────
def _ppg_stats(window: np.ndarray):
    """
    Statistical features of the raw PPG window amplitude.

    These capture signal morphology (shape, spread, asymmetry) without
    requiring peak detection — useful when peaks are hard to detect.

    Args:
        window: 1-D PPG array.

    Returns:
        mean, std, skewness, kurtosis (all floats)
    """
    return (
        float(np.mean(window)),
        float(np.std(window)),
        float(skew(window)),
        float(kurtosis(window)),
    )


def _ppg_spectral_power(window: np.ndarray):
    """
    Power of the raw PPG signal in three cardiac-relevant frequency bands.

    These capture how much energy the PPG waveform carries at different
    heart rate ranges — a different signal property from the HRV LF/HF
    ratio, which is computed on the IBI sequence.

    Bands:
        low  : 0.5–1.0 Hz  (30–60 BPM, resting / meditation)
        mid  : 1.0–2.0 Hz  (60–120 BPM, normal activity / stress)
        high : 2.0–4.0 Hz  (120–240 BPM, high HR / harmonics)

    Args:
        window: 1-D filtered PPG array.

    Returns:
        pow_low, pow_mid, pow_high (all floats)
    """
    nperseg = min(len(window), 256)
    freqs, psd = welch(window, fs=BVP_FS, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]

    pow_low  = np.sum(psd[(freqs >= 0.5) & (freqs < 1.0)]) * freq_res
    pow_mid  = np.sum(psd[(freqs >= 1.0) & (freqs < 2.0)]) * freq_res
    pow_high = np.sum(psd[(freqs >= 2.0) & (freqs < 4.0)]) * freq_res

    return float(pow_low), float(pow_mid), float(pow_high)


def _peak_amplitude_features(window: np.ndarray, peaks: np.ndarray):
    """
    Amplitude-based features derived from detected systolic peaks.

    Uses troughs (valleys between consecutive peaks) to measure pulse
    amplitude — a proxy for stroke volume and vascular tone that differs
    between emotional states.

    Args:
        window: 1-D PPG array.
        peaks:  indices of detected systolic peaks.

    Returns:
        peak_amp_mean, peak_amp_std, pulse_amplitude (all floats)
        Returns (nan, nan, nan) if fewer than 2 peaks.
    """
    if len(peaks) < 2:
        return np.nan, np.nan, np.nan

    peak_heights = window[peaks]
    peak_amp_mean = float(np.mean(peak_heights))
    peak_amp_std  = float(np.std(peak_heights))

    # Trough between each pair of consecutive peaks
    troughs = [window[peaks[i]:peaks[i+1]].min() for i in range(len(peaks) - 1)]
    pulse_amplitude = float(np.mean(peak_heights[:-1] - troughs))

    return peak_amp_mean, peak_amp_std, pulse_amplitude


# ── Per-window feature extraction ─────────────────────────────────────────────
def extract_features(window: np.ndarray):
    """
    Extract all features from a single PPG window.

    Pipeline:
        1. Detect systolic peaks
        2. Compute IBI sequence
        3. Time-domain HRV (mean IBI, HR, SDNN, RMSSD, pNN50)
        4. Frequency-domain HRV (LF, HF, LF/HF) via Welch PSD on resampled IBI
        5. PPG morphology statistics (mean, std, skew, kurtosis)

    Args:
        window: 1-D float32 array of shape (WINDOW_SAMPLES,), already filtered.

    Returns:
        features: 1-D float64 array of length N_FEATURES.
                  Contains np.nan for any feature that could not be computed
                  (e.g. too few peaks detected).
    """
    features = np.full(N_FEATURES, np.nan)

    peaks = _detect_peaks(window)

    if len(peaks) >= 2:
        ibi, ibi_times = _compute_ibi(peaks)

        mean_ibi, mean_hr, sdnn, rmssd, pnn50 = _time_domain_hrv(ibi)
        features[0] = mean_ibi
        features[1] = mean_hr
        features[2] = sdnn
        features[3] = rmssd
        features[4] = pnn50

        if len(ibi) >= 4:
            lf, hf, lf_hf = _freq_domain_hrv(ibi, ibi_times)
            features[5] = lf
            features[6] = hf
            features[7] = lf_hf

        features[15], features[16], features[17] = _peak_amplitude_features(window, peaks)

    # PPG stats — always computable
    features[8], features[9], features[10], features[11] = _ppg_stats(window)

    # PPG spectral power bands — always computable
    features[12], features[13], features[14] = _ppg_spectral_power(window)

    return features


# ── Full dataset feature extraction ───────────────────────────────────────────
def extract_all_features(X_list: list, y_list: list):
    """
    Extract features for every window across all subjects.

    Windows where peak detection fails (< 2 peaks found) produce NaN rows,
    which are dropped before returning.

    Args:
        X_list: list of per-subject window arrays  (from load_all_subjects).
        y_list: list of per-subject label arrays.

    Returns:
        X_feat: (n_valid_windows, N_FEATURES) float64 array.
        y_feat: (n_valid_windows,) int array.
    """
    all_features, all_labels = [], []

    for X_subj, y_subj in zip(X_list, y_list):
        for window, label in zip(X_subj, y_subj):
            feat = extract_features(window)
            all_features.append(feat)
            all_labels.append(label)

    X_feat = np.array(all_features, dtype=np.float64)
    y_feat = np.array(all_labels, dtype=int)

    # Drop windows where any HRV feature is NaN
    valid_mask = ~np.isnan(X_feat).any(axis=1)
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} windows with failed peak detection")

    return X_feat[valid_mask], y_feat[valid_mask]
