import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

from src.data_loader import LABEL_NAMES, N_CLASSES

# ── Helpers ───────────────────────────────────────────────────────────────────
def _scale(X_train: np.ndarray, X_test: np.ndarray):
    """
    Fit a StandardScaler on training data only and apply to both splits.

    Fitting on train only prevents test statistics leaking into the scaler —
    the same rule as the subject-level split.

    Returns:
        X_train_scaled, X_test_scaled, fitted scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """
    Print accuracy, macro F1, and full classification report.
    Return a dict of metrics for later comparison.
    """
    acc    = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Macro F1  : {macro_f1:.4f}")
    print()
    print(classification_report(
        y_true, y_pred,
        target_names=[LABEL_NAMES[i] for i in range(N_CLASSES)],
        digits=3
    ))

    return {"model": model_name, "accuracy": acc, "macro_f1": macro_f1}


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str, save_path: str = None):
    """
    Plot and optionally save a normalised confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    labels = [LABEL_NAMES[i] for i in range(N_CLASSES)]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Confusion matrix saved to {save_path}")
    plt.close()


# ── SVM ───────────────────────────────────────────────────────────────────────
def train_svm(X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray,  y_test: np.ndarray):
    """
    Train an RBF-kernel SVM on scaled features and evaluate on the test set.

    Key hyperparameters:
        kernel : RBF — good default for non-linear physiological feature spaces
        C      : 10  — moderate regularisation; controls margin vs misclassification
        gamma  : 'scale' — 1 / (n_features * X.var()), adapts to feature scale
        class_weight : 'balanced' — compensates for class imbalance
                       (amusement has ~3× fewer windows than baseline)

    Args:
        X_train, y_train : training features and labels.
        X_test,  y_test  : held-out test features and labels.

    Returns:
        metrics dict, fitted model, fitted scaler
    """
    X_train_s, X_test_s, scaler = _scale(X_train, X_test)

    # Apply SMOTE to train data
    smote = SMOTE(random_state=42)
    X_train_reshaped, y_train_reshaped = smote.fit_resample(X_train_s, y_train)

    print(f"  Before SMOTE: {np.bincount(y_train)}")
    print(f"  After SMOTE:  {np.bincount(y_train_reshaped)}")
    
    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
    )
    model.fit(X_train_reshaped, y_train_reshaped)
    y_pred = model.predict(X_test_s)

    metrics = evaluate(y_test, y_pred, "SVM (RBF kernel)")
    plot_confusion_matrix(y_test, y_pred, "SVM",
                          save_path="confusion_matrix_svm.png")

    return metrics, model, scaler


# ── Random Forest ─────────────────────────────────────────────────────────────
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray,  y_test: np.ndarray):
    """
    Train a Random Forest on raw (unscaled) features and evaluate.

    Random Forest does not require feature scaling — tree splits are based
    on rank order, not absolute distances.

    Key hyperparameters:
        n_estimators  : 300   — enough trees for stable predictions
        max_depth     : None  — trees grow fully; forest variance controlled
                                by averaging and feature subsampling
        max_features  : 'sqrt'— each split considers sqrt(n_features) features,
                                standard for classification
        class_weight  : 'balanced' — same class imbalance correction as SVM
        random_state  : 42    — reproducibility

    Args:
        X_train, y_train : training features and labels.
        X_test,  y_test  : held-out test features and labels.

    Returns:
        metrics dict, fitted model
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred, "Random Forest")
    plot_confusion_matrix(y_test, y_pred, "Random Forest",
                          save_path="confusion_matrix_rf.png")

    # Feature importance
    _plot_feature_importance(model)

    return metrics, model


def _plot_feature_importance(model: RandomForestClassifier):
    """
    Plot Random Forest feature importances — useful insight for the report.
    """
    from src.feature_extraction import FEATURE_NAMES

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(importances)),
           importances[indices], color="steelblue")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in indices],
                       rotation=45, ha="right", fontsize=9)
    ax.set_title("Random Forest — Feature Importances")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance_rf.png", dpi=150)
    plt.close()
    print("  Feature importance plot saved to feature_importance_rf.png")


# ── 1D CNN ───────────────────────────────────────────────────────────────────
class PPGDataset(Dataset):
    """
    PyTorch Dataset wrapping numpy arrays of PPG windows and labels.
    Adds a channel dimension so input shape is (1, WINDOW_SAMPLES).

    When augment=True (training only), applies two lightweight transforms:
    - Gaussian noise  (σ = 2% of signal std): simulates sensor noise
    - Amplitude scale (uniform 0.9–1.1):      simulates inter-subject variability
    These help the CNN generalise to unseen subjects on the small WESAD dataset.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            noise = torch.randn_like(x) * 0.02 * x.std()
            scale = 0.9 + 0.2 * torch.rand(1)
            x = x * scale + noise
        return x, self.y[idx]


class CNN1D(nn.Module):
    """
    End-to-end 1D CNN for PPG emotion classification.

    Architecture
    ─────────────────────────────────────────────────────────
    Input  : (batch, 1, 3840)   — raw filtered PPG window

    Block 1: Conv1d(1→64,  k=7) → BN → ReLU → MaxPool(4)  → (batch, 64,  960)
    Block 2: Conv1d(64→128, k=5) → BN → ReLU → MaxPool(4)  → (batch, 128, 240)
    Block 3: Conv1d(128→256,k=3) → BN → ReLU → MaxPool(4)  → (batch, 256,  60)

    GlobalAvgPool → (batch, 256)
    FC(256→128) → ReLU → Dropout(0.5)
    FC(128→4)   → logits (CrossEntropyLoss handles softmax)
    ─────────────────────────────────────────────────────────

    Design choices:
    - Decreasing kernel size (7→5→3): first layer captures broad waveform
      shape; later layers capture finer morphological details.
    - MaxPool(4) aggressively downsamples — keeps model lightweight.
    - GlobalAvgPool instead of Flatten — reduces parameters and overfitting
      risk on the small WESAD dataset (15 subjects).
    - Dropout(0.5) before final layer — key regulariser for small datasets.
    """
    def __init__(self, n_classes: int = 4):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_avg_pool(x).squeeze(-1)  # (batch, 256)
        return self.classifier(x)


def train_cnn(X_train: np.ndarray, y_train: np.ndarray,
              X_test:  np.ndarray, y_test:  np.ndarray,
              epochs: int = 50, batch_size: int = 32, lr: float = 1e-3):
    """
    Train the 1D CNN end-to-end on raw PPG windows.

    Key training decisions:
    - class_weight: computed from training labels to handle imbalance
      (same motivation as class_weight='balanced' in sklearn models)
    - Adam optimiser: adaptive learning rate, standard for deep learning
    - ReduceLROnPlateau: halves LR if val loss stalls for 5 epochs
    - No feature scaling needed — BatchNorm inside the network normalises
      activations per layer.

    Args:
        X_train, y_train : (n, WINDOW_SAMPLES) train arrays.
        X_test,  y_test  : (n, WINDOW_SAMPLES) test arrays.
        epochs           : number of training epochs.
        batch_size       : mini-batch size.
        lr               : initial learning rate.

    Returns:
        metrics dict, trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training on: {device}")

    # Class weights to handle imbalance
    classes, counts = np.unique(y_train, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # Data loaders
    train_loader = DataLoader(PPGDataset(X_train, y_train, augment=True),
                              batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(PPGDataset(X_test,  y_test),
                              batch_size=batch_size, shuffle=False)

    model = CNN1D(n_classes=N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # CosineAnnealingLR decays LR smoothly to eta_min over all epochs,
    # avoiding the abrupt drops of ReduceLROnPlateau and letting the model
    # explore a wider loss landscape early then converge tightly at the end.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    # Training loop
    train_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)

        epoch_loss /= len(y_train)
        train_losses.append(epoch_loss)
        scheduler.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss: {epoch_loss:.4f}")

    # Evaluate
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds = model(X_batch.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    y_pred = np.array(all_preds)
    metrics = evaluate(y_test, y_pred, "1D CNN (end-to-end)")
    plot_confusion_matrix(y_test, y_pred, "1D CNN",
                          save_path="confusion_matrix_cnn.png")
    _plot_training_loss_named(train_losses, "1D CNN", "cnn_training_loss.png")

    return metrics, model


# ── CNN-LSTM hybrid ───────────────────────────────────────────────────────────
class CNNLSTMClassifier(nn.Module):
    """
    CNN-LSTM hybrid emotion classifier for PPG windows.

    Architecture
    ─────────────────────────────────────────────────────────
    Input  : (batch, 3840)   — raw filtered, z-normalised PPG window

    Reshape: (batch, 60, 64) — treat the window as a sequence of 60
             one-second chunks (64 samples each @ 64 Hz).

    Per-chunk CNN encoder (applied identically to each of the 60 chunks):
      Conv1d(1→32, k=5, pad=2) → BN → ReLU → MaxPool(2) → (32, 32)
      Conv1d(32→64, k=3, pad=1) → BN → ReLU → AdaptiveAvgPool → (64,)
    This replaces 64 raw samples with a 64-dim cardiac feature vector.

    LSTM(64→128, 2 layers, dropout=0.3):
      Sees a sequence of 60 meaningful cardiac feature vectors and models
      how the cardiac signal evolves over the 60-second window (e.g., HR
      drift during stress, HRV changes during amusement).

    Last hidden state: (batch, 128)
    FC(128→64) → ReLU → Dropout(0.5)
    FC(64→N_CLASSES) → logits
    ─────────────────────────────────────────────────────────

    Why hybrid beats pure LSTM:
    The pure LSTM received 64 raw PPG samples per timestep — noisy,
    unstructured input that forces the LSTM to simultaneously learn local
    waveform features AND long-range temporal patterns.  The CNN encoder
    handles local waveform feature extraction, leaving the LSTM free to
    focus on temporal dynamics across the full 60-second window.

    This mirrors the architecture described in the course reference:
    "Feature Augmented Hybrid CNN for Stress Recognition Using
     Wrist-based Photoplethysmography Sensor."
    """
    def __init__(self, n_classes: int = 4,
                 hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.seq_len    = 60  # one-second chunks
        self.chunk_size = 64  # samples per chunk (= BVP_FS)

        # Small CNN applied to every 1-second chunk independently
        self.chunk_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),                          # (32, 32)
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                  # (64, 1)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, 3840)
        B = x.size(0)
        # Split into 60 one-second chunks → (batch*60, 1, 64)
        chunks = x.view(B * self.seq_len, 1, self.chunk_size)
        # Encode each chunk → (batch*60, 64)
        enc = self.chunk_encoder(chunks).squeeze(-1)
        # Reshape back to sequence → (batch, 60, 64)
        enc = enc.view(B, self.seq_len, 64)
        # LSTM over the sequence
        _, (h_n, _) = self.lstm(enc)
        out = h_n[-1]                  # last layer hidden state: (batch, 128)
        return self.classifier(out)


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_test:  np.ndarray, y_test:  np.ndarray,
               epochs: int = 60, batch_size: int = 32, lr: float = 5e-4):
    """
    Train the LSTM classifier on raw PPG windows.

    Key training decisions (same rationale as CNN except where noted):
    - lr=5e-4: LSTMs often need a slightly lower LR than CNNs to avoid
      exploding gradients through the recurrent connections.
    - CosineAnnealingLR: smoothly decays LR to near-zero over training,
      helping the model settle into a better minimum than step decay.
    - gradient clipping (max_norm=1.0): standard practice for RNNs to
      prevent gradient explosion.

    Args / Returns: same signature as train_cnn.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training on: {device}")

    classes, counts = np.unique(y_train, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    train_loader = DataLoader(PPGDataset(X_train, y_train, augment=True),
                              batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(PPGDataset(X_test,  y_test),
                              batch_size=batch_size, shuffle=False)

    model     = CNNLSTMClassifier(n_classes=N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    train_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.squeeze(1).to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)

        epoch_loss /= len(y_train)
        train_losses.append(epoch_loss)
        scheduler.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss: {epoch_loss:.4f}")

    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds = model(X_batch.squeeze(1).to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    y_pred = np.array(all_preds)
    metrics = evaluate(y_test, y_pred, "CNN-LSTM (hybrid)")
    plot_confusion_matrix(y_test, y_pred, "CNN-LSTM",
                          save_path="confusion_matrix_cnn_lstm.png")
    _plot_training_loss_named(train_losses, "CNN-LSTM", "cnn_lstm_training_loss.png")

    return metrics, model


def _plot_training_loss_named(losses: list, name: str, path: str):
    _, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(1, len(losses) + 1), losses, color="mediumseagreen", linewidth=1.2)
    ax.set_title(f"{name} — Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Training loss plot saved to {path}")


# ── Comparison summary ────────────────────────────────────────────────────────
def print_comparison(results: list):
    """
    Print a side-by-side summary table of all model results.

    Args:
        results: list of metrics dicts returned by evaluate().
    """
    print(f"\n{'═'*50}")
    print(f"  Model Comparison Summary")
    print(f"{'═'*50}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'─'*45}")
    for r in results:
        print(f"  {r['model']:<25} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f}")
    print(f"{'═'*50}\n")


# ── Leave-One-Subject-Out cross-validation ────────────────────────────────────
def loso_cv(subjects: list, X_list: list, y_list: list, model_type: str = "svm"):
    """
    Leave-One-Subject-Out cross-validation for feature-based models.

    Each fold: train on 14 subjects, test on 1. Average metrics over all folds.
    This is the evaluation protocol used in the WESAD paper and gives a more
    robust estimate than a fixed 3-subject split, since every subject is tested.

    Args:
        subjects:   list of subject ID strings.
        X_list:     list of per-subject feature arrays (n_windows, N_FEATURES).
        y_list:     list of per-subject label arrays.
        model_type: "svm" or "rf".

    Returns:
        mean accuracy, mean macro F1, per-fold results list
    """
    from src.feature_extraction import extract_all_features

    all_accs, all_f1s = [], []
    all_preds, all_true = [], []

    print(f"\n── LOSO-CV  [{model_type.upper()}] ({'18 features'}) ─────────────────────────")

    for i, test_sid in enumerate(subjects):
        # Split
        train_idx = [j for j in range(len(subjects)) if j != i]
        X_tr_raw = np.concatenate([X_list[j] for j in train_idx])
        y_tr_raw = np.concatenate([y_list[j] for j in train_idx])
        X_te_raw = X_list[i]
        y_te_raw = y_list[i]

        # Feature extraction
        X_tr, y_tr = extract_all_features([X_tr_raw], [y_tr_raw])
        X_te, y_te = extract_all_features([X_te_raw], [y_te_raw])

        if len(X_te) == 0:
            continue

        # Train
        if model_type == "svm":
            X_tr_s, X_te_s, _ = _scale(X_tr, X_te)
            clf = SVC(kernel="rbf", C=10, gamma="scale",
                      class_weight="balanced", random_state=42)
            clf.fit(X_tr_s, y_tr)
            y_pred = clf.predict(X_te_s)
        else:
            clf = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                         class_weight="balanced", n_jobs=-1,
                                         random_state=42)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred, average="macro")
        all_accs.append(acc)
        all_f1s.append(f1)
        all_preds.extend(y_pred)
        all_true.extend(y_te)
        print(f"  {test_sid}  acc={acc:.3f}  macro_f1={f1:.3f}")

    mean_acc = float(np.mean(all_accs))
    mean_f1  = float(np.mean(all_f1s))
    print(f"\n  Mean accuracy : {mean_acc:.4f}")
    print(f"  Mean macro F1 : {mean_f1:.4f}")

    # Aggregate confusion matrix over all folds
    model_name = f"LOSO {model_type.upper()}"
    plot_confusion_matrix(np.array(all_true), np.array(all_preds),
                          model_name,
                          save_path=f"confusion_matrix_loso_{model_type}.png")

    return {"model": model_name, "accuracy": mean_acc, "macro_f1": mean_f1}
