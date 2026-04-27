# EmotionRecognitionPPG

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-Educational-blue?style=for-the-badge)

**A multi-modal emotion classification pipeline using raw BVP (PPG) data from the WESAD dataset.**

</div>

---

## 📋 Table of Contents

- [About](#-about)
- [Team Members](#-team-members)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Project Structure](#-project-structure)
---

## 🎯 About

![Status](https://img.shields.io/badge/Status-Complete-green?style=flat-square)
![Version](https://img.shields.io/badge/Version-1.0.0-blue?style=flat-square)

---

## 👥 Team Members

| Member Name                | Unity ID |
| -------------------------- | -------- |
| Edward Feng                | sfeng9   |
| Shazia Muckram             | smuckra  |

---

## ✨ Features

- **Robust Preprocessing:** 4th-order Butterworth Bandpass filtering (0.5–4.0 Hz) with zero-phase distortion.
- **Advanced Feature Extraction:** 18 handcrafted features including Time-Domain HRV (RMSSD, pNN50), Frequency-Domain HRV (LF/HF), and PPG Morphological statistics.
- **Deep Learning Architectures:**
  - **1D CNN:** Automated feature learning directly from filtered waveforms.
  - **CNN-LSTM Hybrid:** Spatial feature extraction combined with temporal sequence modeling.
- **Subject-Independent Evaluation:** Implements both a fixed subject-level split and **Leave-One-Subject-Out (LOSO)** cross-validation to ensure model generalizability.
- **Class Imbalance Handling:** Utilization of SMOTE for feature-based models and class-weighted loss functions for neural networks.

---

## 🛠️ Tech Stack

- **Data Processing:** NumPy, Scipy (signal processing), Pandas
- **Machine Learning:** Scikit-learn, Imbalanced-learn (SMOTE)
- **Deep Learning:** PyTorch (1D CNN, LSTM)
- **Visualization:** Matplotlib, Seaborn

---

## 📦 Installation

### Prerequisites

![Requirements](https://img.shields.io/badge/Requirements-Check-green?style=flat-square)

- Python 3.7 or higher

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd emotionRecognitionPPG
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Data Setup

Ensure the WESAD dataset is placed in the src/WESAD/ directory. The structure should follow:
src/WESAD/S2/S2.pkl, src/WESAD/S3/S3.pkl, etc.

---

## 🚀 Usage

### Running the Application

The project is controlled via main.py, which executes the full pipeline: single subject visualization, feature extraction, and model training/testing.

```bash
python main.py
```

---

## 📊 Results

Our experiments highlight the superior performance of deep learning for physiological signals, particularly for stress detection.

| Model | Accuracy | Macro F1 | Key Highlight |
| :--- | :---: | :---: | :--- |
| **1D CNN (End-to-End)** | **65.45%** | **0.5982** | **97% Recall for Stress** |
| Random Forest | 63.64% | 0.5564 | Best interpretability |
| CNN-LSTM (Hybrid) | 64.00% | 0.5321 | 86% Recall for Meditation |
| LOSO Random Forest | 60.28% | 0.4855 | Most realistic for new users |


---

## 📁 Project Structure

```
emotionRecognitionPPG/
├── 📄 main.py                    # Main entry point
├── 📄 .gitignore                 # Git ignore rules
├── 📄 README.md                  # This file
│
emotionRecognitionPPG/
├── 📄 main.py                    # Main entry point (Runs all experiments)
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # This file
├── 📄 report.pdf                 # Final report
├── src/
│   ├── data_loader.py             # WESAD loading, filtering, & windowing
│   ├── feature_extraction.py      # 18-feature HRV & Morphological extraction
│   ├── models.py                  # PyTorch and Scikit-Learn model definitions
│   └── WESAD/                     # [Folder] Subject .pkl files
└── plots/                         # Generated confusion matrices & loss curves
```

---

## 📝 Notes


---

## 📄 License

![License](https://img.shields.io/badge/License-Educational-blue?style=flat-square)

This project is for **educational purposes** (CSC591 012 - Project).

---

## 🙏 Acknowledgments

- **Course**: CSC591 012 - Ubiquitous Computing and Mobile Health
- **Institution**: North Carolina State University
- **Dataset**: Schmidt, P. et al. (WESAD)
- **Project**: Emotion Recognition from PPG Signals

---
