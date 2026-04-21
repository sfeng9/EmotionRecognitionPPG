import numpy as np
import pickle
import matplotlib.pyplot as plt

from src.data_loader import (
    load_all_subjects, load_subject, split_by_subject,
    apply_bandpass, BVP_FS, LABEL_NAMES, WINDOW_SAMPLES
)
from src.feature_extraction import extract_all_features, FEATURE_NAMES, N_FEATURES
from src.models import train_svm, train_random_forest, train_cnn, loso_cv, print_comparison

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


def test_feature_extraction():
    """
    Run feature extraction on all subjects and print a summary.
    """
    print("\n── Feature extraction ────────────────────────────────────────────")
    subjects, X_list, y_list = load_all_subjects(WESAD_DIR)

    X_feat, y_feat = extract_all_features(X_list, y_list)

    print(f"\nFeature matrix shape : {X_feat.shape}  "
          f"({X_feat.shape[0]} windows × {N_FEATURES} features)")
    print(f"Labels shape         : {y_feat.shape}")

    print("\nFeature summary (mean ± std across all windows):")
    for i, name in enumerate(FEATURE_NAMES):
        col = X_feat[:, i]
        print(f"  {name:<15}: {col.mean():.4f} ± {col.std():.4f}")

    return X_feat, y_feat


def run_feature_based_models():
    """
    Full feature-based pipeline:
      1. Load all subjects
      2. Split by subject (3 test subjects, 12 train)
      3. Extract features from train and test windows separately
      4. Train SVM and Random Forest
      5. Evaluate and compare
    """
    print("\n── Feature-based models ──────────────────────────────────────────")

    subjects, X_list, y_list = load_all_subjects(WESAD_DIR)

    # Hold out 3 subjects for testing — ~20% of subjects
    TEST_SUBJECTS = ["S2", "S3", "S4"]
    X_train, y_train, X_test, y_test = split_by_subject(
        subjects, X_list, y_list, test_subjects=TEST_SUBJECTS
    )

    # Extract features — fit only on train windows
    print("\nExtracting features...")
    X_train_feat, y_train_feat = extract_all_features(
        [X_train], [y_train]
    )
    X_test_feat, y_test_feat = extract_all_features(
        [X_test], [y_test]
    )

    print(f"Train: {X_train_feat.shape}, Test: {X_test_feat.shape}")

    # Train and evaluate
    svm_metrics, _, _ = train_svm(X_train_feat, y_train_feat,
                                   X_test_feat,  y_test_feat)
    rf_metrics,  _    = train_random_forest(X_train_feat, y_train_feat,
                                             X_test_feat,  y_test_feat)
    return [svm_metrics, rf_metrics]


def run_cnn():
    """
    End-to-end 1D CNN pipeline:
      1. Load all subjects
      2. Split by subject (same 3 test subjects as feature-based for fair comparison)
      3. Feed raw PPG windows directly into the CNN — no hand-crafted features
    """
    print("\n── 1D CNN (end-to-end) ───────────────────────────────────────────")

    subjects, X_list, y_list = load_all_subjects(WESAD_DIR)

    TEST_SUBJECTS = ["S2", "S3", "S4"]
    X_train, y_train, X_test, y_test = split_by_subject(
        subjects, X_list, y_list, test_subjects=TEST_SUBJECTS
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    cnn_metrics, _ = train_cnn(X_train, y_train, X_test, y_test,
                                epochs=50, batch_size=32, lr=1e-3)
    return cnn_metrics


def run_loso():
    """
    LOSO cross-validation for SVM and Random Forest.
    Trains on 14 subjects, tests on 1, rotates through all 15.
    Uses the full 18-feature set.
    """
    print("\n── LOSO Cross-Validation ─────────────────────────────────────────")
    subjects, X_list, y_list = load_all_subjects(WESAD_DIR)
    svm_loso = loso_cv(subjects, X_list, y_list, model_type="svm")
    rf_loso  = loso_cv(subjects, X_list, y_list, model_type="rf")
    return [svm_loso, rf_loso]


if __name__ == "__main__":
    test_single_subject()
    test_all_subjects()
    test_feature_extraction()

    # Fixed split: SVM + RF + CNN
    feature_results = run_feature_based_models()
    cnn_metrics     = run_cnn()

    # LOSO cross-validation: SVM + RF
    loso_results = run_loso()

    # Final comparison across all models
    print_comparison(feature_results + [cnn_metrics] + loso_results)
