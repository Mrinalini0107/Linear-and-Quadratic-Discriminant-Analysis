# 🔬 Linear and Quadratic Discriminant Analysis for Breast Cancer Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20WDBC-green?style=flat-square)](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Dataset Description](#dataset-description)
4. [Project Workflow](#project-workflow)
5. [Tasks Covered](#tasks-covered)
   - [Task 1 — Linear Discriminant Analysis (LDA) with 3-Fold Cross-Validation](#task-1--linear-discriminant-analysis-lda-with-3-fold-cross-validation)
   - [Task 2 — Quadratic Discriminant Analysis (QDA) with 4-Fold Cross-Validation](#task-2--quadratic-discriminant-analysis-qda-with-4-fold-cross-validation)
6. [Performance Metrics Explained](#performance-metrics-explained)
7. [Results Summary](#results-summary)
8. [Learning Objectives](#learning-objectives)
9. [Project Structure](#project-structure)
10. [Requirements](#requirements)
11. [How to Run](#how-to-run)
12. [Conclusion](#conclusion)
13. [References](#references)

---

## Introduction

Breast cancer is one of the most prevalent and life-threatening diseases worldwide. Early and accurate detection of whether a tumour is **benign** or **malignant** is critical for timely treatment and improved patient outcomes. Machine learning-based classification models offer a powerful, non-invasive approach to support clinical diagnosis.

This project applies two classical and statistically principled discriminant analysis techniques — **Linear Discriminant Analysis (LDA)** and **Quadratic Discriminant Analysis (QDA)** — to the well-known **Wisconsin Breast Cancer Diagnostic (WDBC)** dataset from the UCI Machine Learning Repository. Both models are evaluated using stratified k-fold cross-validation, and their performance is assessed through a comprehensive suite of metrics: **ROC curves**, **AUC (Area Under the Curve)**, **EER (Equal Error Rate)**, and the **discriminability index d′ (d-prime)**.

The project demonstrates how classical statistical classifiers, even without deep learning, can achieve near-perfect discrimination performance on structured medical imaging data — while also offering interpretable, theoretically grounded model outputs.

---

## Problem Statement

> **Given 30 numeric measurements of tumour cell nuclei derived from digitised fine-needle aspirate (FNA) images, classify each tumour sample as Malignant (+1) or Benign (−1) using discriminant analysis.**

**Task 1:** Apply **Linear Discriminant Analysis (LDA)** with **3-fold stratified cross-validation**. For each fold, produce both the training and validation ROC curves (6 curves total). Report the:
- Average training and validation **AUC**
- Approximate **Equal Error Rate (EER)**
- **Discriminability index d′**

**Task 2:** Repeat the above using **Quadratic Discriminant Analysis (QDA)** with **4-fold stratified cross-validation** (8 ROC curves total). Report the same set of metrics.

---

## Dataset Description

| Property | Details |
|---|---|
| **Source** | [UCI ML Repository — Breast Cancer Wisconsin (Diagnostic)](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) |
| **Total Samples** | 569 |
| **Features** | 30 numeric features (+ ID and Diagnosis columns) |
| **Target Variable** | Diagnosis: `M` → Malignant (+1), `B` → Benign (−1) |
| **Class Distribution** | 357 Benign · 212 Malignant |
| **Missing Values** | None |
| **Feature Type** | Continuous (real-valued measurements) |

### Feature Engineering

The 30 features are computed from digitised images of FNA samples and describe characteristics of the cell nuclei present in each image. They are organised as **10 measurements × 3 statistical summaries** (mean, standard error, worst/largest value):

| # | Nucleus Characteristic |
|---|---|
| 1 | Radius (mean of distances from centre to points on perimeter) |
| 2 | Texture (standard deviation of grey-scale values) |
| 3 | Perimeter |
| 4 | Area |
| 5 | Smoothness (local variation in radius lengths) |
| 6 | Compactness (perimeter² / area − 1.0) |
| 7 | Concavity (severity of concave portions of contour) |
| 8 | Concave points (number of concave portions of contour) |
| 9 | Symmetry |
| 10 | Fractal dimension ("coastline approximation" − 1) |

### Target Encoding

```python
T = data['Diagnosis'].map({'M': +1, 'B': -1}).values
# Malignant → +1  (positive class)
# Benign    → −1  (negative class)
```

---

## Project Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                           │
│                                                             │
│  UCI WDBC Dataset                                           │
│       │                                                     │
│       ▼                                                     │
│  Load & Parse  ──►  Drop ID  ──►  Encode Target             │
│  (569 × 32)         (569 × 31)    M→+1, B→−1                │
│       │                                                     │
│       ▼                                                     │
│  Feature Matrix P (569 × 30)  +  Target Vector T (569,)     │
│       │                           │                         │
│       ├──────── TASK 1 ───────────┤                         │
│       │   LDA + StratifiedKFold   │                         │
│       │   (3 folds → 6 ROC curves)│                         │
│       │   AUC · EER · d′          │                         │
│       │                           │                         │
│       └──────── TASK 2 ───────────┘                         │
│           QDA + StratifiedKFold                             │
│           (4 folds → 8 ROC curves)                          │
│           AUC · EER · d′                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Tasks Covered

### Task 1 — Linear Discriminant Analysis (LDA) with 3-Fold Cross-Validation

**Description:**

Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction and classification technique that finds a linear combination of features that best separates two or more classes. It operates under two key assumptions:

1. **Multivariate normality** — each class follows a Gaussian distribution.
2. **Homoscedasticity** — all classes share the same covariance matrix.

Under these assumptions, LDA derives a single linear decision boundary by maximising the ratio of between-class variance to within-class variance (Fisher's criterion).

**Implementation:**

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

lda = LinearDiscriminantAnalysis()
skf = StratifiedKFold(n_splits=3)

for fold, (train_index, val_index) in enumerate(skf.split(P, T), 1):
    P_train, P_val = P[train_index], P[val_index]
    T_train, T_val = T[train_index], T[val_index]

    lda.fit(P_train, T_train)

    train_probs = lda.predict_proba(P_train)[:, 1]
    val_probs   = lda.predict_proba(P_val)[:, 1]

    train_fpr, train_tpr, _ = roc_curve(T_train, train_probs)
    val_fpr,   val_tpr,   _ = roc_curve(T_val,   val_probs)
```

**Cross-Validation Design:**

| Setting | Value |
|---|---|
| Strategy | Stratified K-Fold (preserves class ratio per fold) |
| Number of folds | 3 |
| ROC curves produced | 6 (3 training + 3 validation) |

**Key Results (LDA):**

| Metric | Value |
|---|---|
| Average Training AUC | ≈ **0.997** |
| Average Validation AUC | ≈ **0.991** |
| Equal Error Rate (EER) | ≈ **0.045** |
| Discriminability Index d′ | ≈ **5.353** |

---

### Task 2 — Quadratic Discriminant Analysis (QDA) with 4-Fold Cross-Validation

**Description:**

Quadratic Discriminant Analysis (QDA) relaxes the homoscedasticity assumption of LDA and allows each class to have its own covariance matrix. This results in a **quadratic (curved) decision boundary**, offering greater flexibility when the classes have different spreads or orientations in feature space.

The trade-off is increased model complexity and a higher risk of overfitting with small datasets, since QDA must estimate a separate covariance matrix for each class.

**Implementation:**

```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

qda = QuadraticDiscriminantAnalysis()
skf = StratifiedKFold(n_splits=4)

for fold, (train_index, val_index) in enumerate(skf.split(P, T), 1):
    P_train, P_val = P[train_index], P[val_index]
    T_train, T_val = T[train_index], T[val_index]

    qda.fit(P_train, T_train)

    train_probs = qda.predict_proba(P_train)[:, 1]
    val_probs   = qda.predict_proba(P_val)[:, 1]

    train_fpr, train_tpr, _ = roc_curve(T_train, train_probs)
    val_fpr,   val_tpr,   _ = roc_curve(T_val,   val_probs)
```

**Cross-Validation Design:**

| Setting | Value |
|---|---|
| Strategy | Stratified K-Fold (preserves class ratio per fold) |
| Number of folds | 4 |
| ROC curves produced | 8 (4 training + 4 validation) |

**Key Results (QDA):**

| Metric | Value |
|---|---|
| Average Training AUC | ≈ **0.996** |
| Average Validation AUC | ≈ **0.988** |
| Equal Error Rate (EER) | ≈ **0.042** |
| Discriminability Index d′ | ≈ **5.835** |

---

## Performance Metrics Explained

### 1. ROC Curve (Receiver Operating Characteristic)
A ROC curve plots the **True Positive Rate (Sensitivity / Recall)** against the **False Positive Rate (1 − Specificity)** across all possible classification thresholds. A perfect classifier produces a curve that passes through the top-left corner (TPR = 1, FPR = 0). A random classifier follows the diagonal.

$$\text{TPR} = \frac{TP}{TP + FN}, \qquad \text{FPR} = \frac{FP}{FP + TN}$$

### 2. AUC (Area Under the ROC Curve)
AUC summarises the entire ROC curve as a single scalar. An AUC of 1.0 represents perfect discrimination; 0.5 represents chance-level performance.

| AUC Range | Interpretation |
|---|---|
| 0.90 – 1.00 | Excellent |
| 0.80 – 0.90 | Good |
| 0.70 – 0.80 | Fair |
| 0.60 – 0.70 | Poor |
| 0.50 – 0.60 | Fail |

### 3. EER (Equal Error Rate)
The EER is the threshold at which the **False Acceptance Rate (FAR)** equals the **False Rejection Rate (FRR)**. A lower EER indicates better classifier performance. It is frequently used in biometric and medical diagnostic systems.

$$\text{EER} : \text{FAR} = \text{FRR}$$

### 4. d′ (d-prime / Discriminability Index)
Originating in signal detection theory, d′ measures the separation between the distributions of positive and negative class scores, normalised by their pooled standard deviation:

$$d' = \frac{|\mu_1 - \mu_0|}{\sqrt{\frac{\sigma_1^2 + \sigma_0^2}{2}}}$$

where $\mu_1, \mu_0$ are the means and $\sigma_1^2, \sigma_0^2$ are the variances of the positive (Malignant) and negative (Benign) class score distributions respectively. A higher d′ indicates greater discriminability between the two classes.

---

## Results Summary

| Model | CV Folds | Avg Train AUC | Avg Val AUC | EER | d′ |
|---|:---:|:---:|:---:|:---:|:---:|
| **LDA** | 3 | 0.997 | 0.991 | 0.045 | 5.353 |
| **QDA** | 4 | 0.996 | 0.988 | 0.042 | 5.835 |

**Key Observations:**

- Both LDA and QDA achieve **near-perfect AUC scores** (>0.98), confirming that the 30 morphological features of tumour cell nuclei are highly discriminative for malignancy classification.
- LDA achieves a marginally higher validation AUC (0.991 vs. 0.988), suggesting that a linear boundary may be adequate — and slightly more generalisable — for this dataset.
- QDA achieves a lower EER (0.042 vs. 0.045) and a higher d′ (5.835 vs. 5.353), indicating that allowing class-specific covariance matrices captures slightly better distributional separation.
- The negligible gap between training and validation AUC in both models demonstrates **no significant overfitting**, indicating robust generalisation to unseen data.

---

## Learning Objectives

By working through this project, learners will be able to:

- **Understand the mathematical foundations of LDA and QDA**, including the role of Fisher's criterion, Gaussian class-conditional densities, shared vs. class-specific covariance matrices, and the derivation of linear vs. quadratic decision boundaries.
- **Implement LDA and QDA in Python** using `sklearn.discriminant_analysis`, including fitting, probability prediction, and decision score extraction.
- **Apply stratified k-fold cross-validation** using `StratifiedKFold` to ensure class balance is preserved across all training and validation splits, and understand why stratification matters for imbalanced medical datasets.
- **Generate and interpret ROC curves** for both training and validation sets across multiple folds, understanding what the shape of the curve reveals about classifier calibration and threshold behaviour.
- **Calculate and interpret AUC** as a threshold-independent summary measure of classifier discrimination performance.
- **Compute the Equal Error Rate (EER)** by identifying the operating point where false acceptance and false rejection rates are equal, and understand its relevance in clinical decision-making contexts.
- **Calculate and interpret the discriminability index d′** from signal detection theory, understanding its relationship to the Gaussian score distributions of each class.
- **Compare LDA and QDA** empirically and theoretically, evaluating the trade-off between the simplicity of a shared covariance assumption (LDA) and the flexibility of class-specific covariances (QDA).
- **Load and preprocess real-world medical datasets** from the UCI ML Repository, including feature-target separation, label encoding, and verification of class distributions.
- **Interpret the generalisation gap** between training and validation metrics to assess the degree of model overfitting and the reliability of reported performance figures.

---

## Project Structure

```
LDA-QDA-Breast-Cancer-Detection/
│
├── Linear_and_Quadratic__Discriminant_Analysis_for_Breast_Cancer_Detection.ipynb
│       ├── Step 1 — Load & Prepare Data (UCI WDBC)
│       ├── Task 1 — LDA with 3-Fold Cross-Validation
│       │     ├── Fit LDA per fold
│       │     ├── ROC curves (3 train + 3 validation)
│       │     └── AUC · EER · d′ reporting
│       └── Task 2 — QDA with 4-Fold Cross-Validation
│             ├── Fit QDA per fold
│             ├── ROC curves (4 train + 4 validation)
│             └── AUC · EER · d′ reporting
│
└── README.md
```

---

## Requirements

### Python Version
Python 3.8 or higher is recommended.

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.21 | Numerical arrays and matrix operations |
| `pandas` | ≥ 1.3 | Dataset loading, parsing, and manipulation |
| `scikit-learn` | ≥ 1.0 | LDA, QDA, StratifiedKFold, roc_curve, auc |
| `scipy` | ≥ 1.7 | `scipy.stats.norm` for d′ computation |
| `matplotlib` | ≥ 3.4 | ROC curve visualisation |

### Installation

```bash
pip install numpy pandas scikit-learn scipy matplotlib
```

Or using a `requirements.txt`:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
scipy>=1.7
matplotlib>=3.4
```

### Jupyter Notebook Setup

```bash
pip install notebook
jupyter notebook
```

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/LDA-QDA-Breast-Cancer-Detection.git
cd LDA-QDA-Breast-Cancer-Detection
```

### 2. Install Dependencies

```bash
pip install numpy pandas scikit-learn scipy matplotlib
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open and Execute the Notebook

Open `Linear_and_Quadratic__Discriminant_Analysis_for_Breast_Cancer_Detection.ipynb` in your browser and run all cells sequentially from top to bottom.

> ⚠️ **Note on Data Loading:** The notebook fetches the WDBC dataset directly from the UCI ML Repository via URL. An active internet connection is required for the first run. The dataset loads automatically — no manual download is needed.

```python
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
data = pd.read_csv(url, header=None, names=columns)
```

> 💡 **Tip:** If the UCI URL is temporarily unavailable, the dataset can also be downloaded directly from [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) and loaded from a local path.

---

## Conclusion

This project demonstrates that **classical discriminant analysis methods remain highly competitive** for structured biomedical classification tasks, achieving near-perfect breast cancer detection performance without recourse to complex deep learning architectures.

The application of **Linear Discriminant Analysis** to the WDBC dataset yielded an average validation AUC of **0.991**, an EER of **0.045**, and a d′ of **5.353** under 3-fold stratified cross-validation. These figures confirm that the 30 morphological tumour features carry sufficient information to support a highly effective **linear decision boundary** between benign and malignant samples.

**Quadratic Discriminant Analysis** under 4-fold cross-validation achieved a validation AUC of **0.988**, a marginally lower EER of **0.042**, and a higher d′ of **5.835**. The relaxation of the shared covariance assumption allowed QDA to model the distinct distributional shapes of each class more precisely — reflected in the improved d′ — at the cost of a marginal reduction in validation AUC due to increased model complexity.

Critically, the **negligible gap between training and validation AUC** in both models (< 0.01) confirms that neither LDA nor QDA is overfitting the training data. Both models generalise robustly to unseen patient samples, which is a paramount requirement in clinical diagnostic applications where over-optimistic in-sample performance can have serious real-world consequences.

From a methodological standpoint, the use of **stratified cross-validation** ensures that the class imbalance between benign (357) and malignant (212) samples is faithfully preserved across all folds, preventing the evaluation from being biased towards the majority class. The adoption of **multiple complementary metrics** — AUC for overall discrimination, EER for threshold-agnostic error balance, and d′ for signal-theoretic separability — provides a richer and more clinically meaningful characterisation of model performance than accuracy alone.

In summary, this project illustrates that with carefully engineered domain-specific features, statistically principled classifiers like LDA and QDA can deliver both high performance and interpretable decision-making — qualities that are essential for building trustworthy AI-assisted diagnostic tools in healthcare.

---

## References

1. **Wolberg, W. H., Street, W. N., & Mangasarian, O. L.** (1995). *Breast Cancer Wisconsin (Diagnostic) Data Set.* UCI Machine Learning Repository. [http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

2. **Fisher, R. A.** (1936). *The Use of Multiple Measurements in Taxonomic Problems.* Annals of Eugenics, 7(2), 179–188.

3. **Green, D. M., & Swets, J. A.** (1966). *Signal Detection Theory and Psychophysics.* Wiley.

4. **Scikit-learn Documentation** — [LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) · [QuadraticDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)

---

*This project is intended for educational purposes in applied machine learning, medical image analysis, and statistical pattern recognition.*
