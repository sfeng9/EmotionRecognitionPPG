import numpy as np
import pickle
import matplotlib.pyplot as plt

from src.data_loader import (
    load_all_subjects, load_subject,
    apply_bandpass, BVP_FS, LABEL_NAMES, WINDOW_SAMPLES
)

WESAD_DIR = "src/WESAD"


def test_single_subject():
    """
    Load one subject, plot raw vs filtered PPG to verify the filter works.
    This also satisfies the report requirement of visualizing a processed segment.
    """
    pkl_path = f"{WESAD_DIR}/S2/S2.pkl"
    print("── Single subject test (S2) ──────────────────────────────────────")

    # Load raw BVP directly so we can compare before/after filtering
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    bvp_raw = data["signal"]["wrist"]["BVP"].squeeze().astype(float)
    bvp_filtered = apply_bandpass(bvp_raw)

    # Plot 10 seconds of signal starting at 60s (avoids initial transient)
    start = 60 * BVP_FS
    end   = start + 10 * BVP_FS
    t = np.arange(end - start) / BVP_FS

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    axes[0].plot(t, bvp_raw[start:end], color="steelblue", linewidth=0.8)
    axes[0].set_title("Raw PPG (BVP) — S2")
    axes[0].set_ylabel("Amplitude (a.u.)")

    axes[1].plot(t, bvp_filtered[start:end], color="tomato", linewidth=0.8)
    axes[1].set_title("Filtered PPG — Butterworth bandpass 0.5–4 Hz, order 4")
    axes[1].set_ylabel("Amplitude (a.u.)")
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig("ppg_filter_comparison.png", dpi=150)
    plt.close()
    print("  Plot saved to ppg_filter_comparison.png")


def test_all_subjects():
    """
    Load every subject, print per-subject window counts and class distribution.
    """
    print("\n── All subjects ──────────────────────────────────────────────────")
    subjects, X_list, y_list = load_all_subjects(WESAD_DIR)

    total_windows = sum(len(y) for y in y_list)
    print(f"\nTotal subjects loaded : {len(subjects)}")
    print(f"Total windows         : {total_windows}")
    print(f"Window shape          : ({WINDOW_SAMPLES},)  [{WINDOW_SAMPLES/BVP_FS:.0f}s @ {BVP_FS}Hz]")

    # Overall class distribution
    all_y = np.concatenate(y_list)
    print("\nOverall class distribution:")
    for idx, name in LABEL_NAMES.items():
        count = (all_y == idx).sum()
        pct = 100 * count / len(all_y)
        print(f"  {idx} {name:<12}: {count:5d} windows ({pct:.1f}%)")

    return subjects, X_list, y_list


if __name__ == "__main__":
    test_single_subject()
    test_all_subjects()
