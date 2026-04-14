import os
import pickle
import numpy as np
from scipy.signal import butter, sosfiltfilt

# ── Constants ────────────────────────────────────────────────────────────────
BVP_FS = 64          # Empatica E4 BVP sampling rate (Hz)
LABEL_FS = 700       # WESAD label sampling rate (Hz)

WINDOW_SEC = 60      # window duration in seconds
STEP_SEC = 30        # step between windows in seconds
WINDOW_SAMPLES = WINDOW_SEC * BVP_FS   # 3840
STEP_SAMPLES = STEP_SEC * BVP_FS       # 1920

LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}  # baseline, stress, amusement, meditation
LABEL_NAMES = {0: "baseline", 1: "stress", 2: "amusement", 3: "meditation"}

# Minimum fraction of a window that must share the majority label.
# Windows below this threshold span a condition boundary and are discarded.
PURITY_THRESHOLD = 0.8

# ── Filter ───────────────────────────────────────────────────────────────────
def _build_bandpass_sos(lowcut: float, highcut: float, fs: float, order: int = 4):
    """
    Design a Butterworth bandpass filter and return second-order sections (SOS).
    SOS representation is numerically more stable than transfer-function (b, a).
    """
    nyq = fs / 2.0
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return sos


_BPF_SOS = _build_bandpass_sos(lowcut=0.5, highcut=4.0, fs=BVP_FS)


def apply_bandpass(signal: np.ndarray) -> np.ndarray:
    """
    Apply zero-phase 4th-order Butterworth bandpass filter (0.5–4 Hz).

    sosfiltfilt runs the filter forwards then backwards, giving zero phase
    shift and doubling the effective filter order without extra design cost.

    Args:
        signal: 1-D float array of raw BVP samples.

    Returns:
        Filtered signal of the same shape.
    """
    return sosfiltfilt(_BPF_SOS, signal)


# ── Label alignment ───────────────────────────────────────────────────────────
def _align_labels(labels: np.ndarray, n_bvp: int) -> np.ndarray:
    """
    Downsample labels from LABEL_FS (700 Hz) to BVP_FS (64 Hz) by nearest-
    neighbour mapping so every BVP sample has an associated label.

    For BVP sample i, the corresponding label index is:
        label_idx = round(i * LABEL_FS / BVP_FS)

    This is safe because the .pkl files are already synchronised.

    Args:
        labels:  1-D int array of length N_label (at 700 Hz).
        n_bvp:   Number of BVP samples (at 64 Hz).

    Returns:
        1-D int array of length n_bvp.
    """
    indices = np.round(np.arange(n_bvp) * (LABEL_FS / BVP_FS)).astype(int)
    indices = np.clip(indices, 0, len(labels) - 1)
    return labels[indices]


# ── Windowing ─────────────────────────────────────────────────────────────────
def _segment(bvp: np.ndarray, labels: np.ndarray):
    """
    Slide a fixed window over a subject's BVP signal and assign one label per
    window using majority vote.  Windows that span a condition boundary
    (purity < PURITY_THRESHOLD) or contain only invalid labels (0, 5, 6, 7)
    are discarded.

    Args:
        bvp:    1-D float array, filtered BVP at BVP_FS.
        labels: 1-D int array, aligned labels (same length as bvp).

    Returns:
        windows: (n_windows, WINDOW_SAMPLES) float array.
        y:       (n_windows,) int array of remapped class indices [0–3].
    """
    windows, y = [], []
    n = len(bvp)

    for start in range(0, n - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        end = start + WINDOW_SAMPLES
        window_labels = labels[start:end]

        # Count only valid labels (1–4)
        valid_mask = np.isin(window_labels, list(LABEL_MAP.keys()))
        if valid_mask.sum() == 0:
            continue

        # Majority vote over valid labels only
        valid_labels = window_labels[valid_mask]
        unique, counts = np.unique(valid_labels, return_counts=True)
        majority_label = unique[np.argmax(counts)]
        purity = counts.max() / len(window_labels)  # fraction of full window

        if purity < PURITY_THRESHOLD:
            continue  # window spans a boundary — discard

        windows.append(bvp[start:end])
        y.append(LABEL_MAP[majority_label])

    if len(windows) == 0:
        return np.empty((0, WINDOW_SAMPLES)), np.empty(0, dtype=int)

    return np.stack(windows), np.array(y, dtype=int)


# ── Per-subject loader ────────────────────────────────────────────────────────
def load_subject(pkl_path: str):
    """
    Load one subject's .pkl file and return filtered, windowed PPG data.

    Steps
    -----
    1. Unpickle the file (Python 2 was used for WESAD — latin-1 encoding needed).
    2. Extract raw BVP and labels.
    3. Align labels to BVP sample rate.
    4. Apply bandpass filter.
    5. Segment into fixed-size windows.

    Args:
        pkl_path: Absolute or relative path to SX.pkl.

    Returns:
        subject_id: String, e.g. "S2".
        windows:    (n_windows, WINDOW_SAMPLES) float32 array.
        labels:     (n_windows,) int array, values in {0, 1, 2, 3}.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    subject_id = data["subject"]

    # BVP is stored as shape (N, 1) — squeeze to 1-D
    bvp_raw = data["signal"]["wrist"]["BVP"].squeeze().astype(np.float64)
    labels_raw = data["label"].squeeze().astype(int)

    # Align labels → BVP sample rate
    labels_aligned = _align_labels(labels_raw, len(bvp_raw))

    # Filter
    bvp_filtered = apply_bandpass(bvp_raw)

    # Segment
    windows, y = _segment(bvp_filtered, labels_aligned)

    return subject_id, windows.astype(np.float32), y


# ── Full dataset loader ───────────────────────────────────────────────────────
def load_all_subjects(wesad_dir: str):
    """
    Load every subject in the WESAD directory.

    Walks wesad_dir looking for SX/SX.pkl files and calls load_subject on each.

    Args:
        wesad_dir: Path to the WESAD/ folder (contains S2/, S3/, … S17/).

    Returns:
        subjects: List of subject ID strings (e.g. ["S2", "S3", ...]).
        X_list:   List of (n_windows, WINDOW_SAMPLES) float32 arrays, one per subject.
        y_list:   List of (n_windows,) int arrays, one per subject.
    """
    subjects, X_list, y_list = [], [], []

    subject_dirs = sorted(
        d for d in os.listdir(wesad_dir)
        if os.path.isdir(os.path.join(wesad_dir, d)) and d.startswith("S")
    )

    for subject_dir in subject_dirs:
        pkl_path = os.path.join(wesad_dir, subject_dir, f"{subject_dir}.pkl")
        if not os.path.exists(pkl_path):
            print(f"  [skip] {pkl_path} not found")
            continue

        subject_id, windows, y = load_subject(pkl_path)
        subjects.append(subject_id)
        X_list.append(windows)
        y_list.append(y)

        label_counts = {LABEL_NAMES[k]: int((y == k).sum()) for k in range(4)}
        print(f"  Loaded {subject_id}: {len(windows)} windows | {label_counts}")

    return subjects, X_list, y_list


# ── Train / test split by subject ─────────────────────────────────────────────
def split_by_subject(subjects, X_list, y_list, test_subjects: list):
    """
    Split data into train and test sets by subject ID.

    The split is done at the subject level — entire subjects go to either
    train or test.  This prevents data leakage: a model cannot memorise
    one subject's windows during training and see the same subject at test time.

    Args:
        subjects:      List of subject ID strings from load_all_subjects().
        X_list:        List of per-subject window arrays.
        y_list:        List of per-subject label arrays.
        test_subjects: List of subject IDs to hold out (e.g. ["S2", "S3"]).

    Returns:
        X_train: (n_train_windows, WINDOW_SAMPLES) float32 array.
        y_train: (n_train_windows,) int array.
        X_test:  (n_test_windows,  WINDOW_SAMPLES) float32 array.
        y_test:  (n_test_windows,)  int array.
    """
    test_set = set(test_subjects)

    train_X, train_y = [], []
    test_X,  test_y  = [], []

    for sid, X, y in zip(subjects, X_list, y_list):
        if sid in test_set:
            test_X.append(X)
            test_y.append(y)
        else:
            train_X.append(X)
            train_y.append(y)

    X_train = np.concatenate(train_X)
    y_train = np.concatenate(train_y)
    X_test  = np.concatenate(test_X)
    y_test  = np.concatenate(test_y)

    print(f"Train: {X_train.shape[0]} windows from {len(train_X)} subjects")
    print(f"Test:  {X_test.shape[0]} windows from {len(test_X)} subjects {test_subjects}")

    return X_train, y_train, X_test, y_test
