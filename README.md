# Cardio-SVM — Cardiovascular Risk Screener

**Author:** Abdallah Mohamed Mohamed
**Student ID:** 231002006
**Course:** Machine Learning, final project

record:https://canva.link/tovbsw0jbp1nda8

A supervised binary-classification pipeline that estimates the probability of cardiovascular disease (CVD) from routine vitals and lifestyle answers. The deployed screener is an **RBF Support Vector Machine**, benchmarked against K-Nearest Neighbours and XGBoost. The repository ships the cleaned modelling dataset, an end-to-end notebook, eleven figures, a **single** serialized bundle (pipeline + threshold + UI metadata in one `.pkl`), and a Flask web app for interactive predictions.

---

## Contents
1. [Why this problem](#why-this-problem)
2. [Dataset at a glance](#dataset-at-a-glance)
3. [Repository tour](#repository-tour)
4. [Environment](#environment)
5. [Modelling pipeline](#modelling-pipeline)
6. [Results](#results)
7. [Web application](#web-application)
8. [Running locally](#running-locally)
9. [Reproducibility notes](#reproducibility-notes)
10. [Caveats and next steps](#caveats-and-next-steps)

---

## Why this problem

Cardiovascular disease is the leading cause of death worldwide and is tightly coupled to blood pressure, body composition, cholesterol, and lifestyle. A cheap screening model that runs on measurements a clinic already records — age, height, weight, two blood-pressure readings, and a few lifestyle answers — can flag who might benefit from a fuller cardiology work-up.

This project answers that with an **RBF Support Vector Machine**. The two CVD classes are nearly balanced (~49.5 % positive), so models are compared on **ROC-AUC** and the deployed SVM's decision threshold is then tuned on **F1**.

---

## Dataset at a glance

| Property | Value |
|---|---|
| Source | Cardiovascular Disease dataset (`cardio_train.csv`, public mirror) |
| Raw rows | 70,000 |
| Modelling rows after cleaning | 68,359 |
| Features used | 17 (8 numeric, 9 categorical) |
| Target column | `Cardio` — presence of cardiovascular disease (0/1) |
| Positive / negative | 33,832 / 34,527 (≈ 49.5 % positive) |

Cleaning and feature engineering baked into `data/cardio_clean.csv`:
- Age converted from days to years; coded integers mapped to readable labels (gender, cholesterol/glucose bands, smoking/alcohol/activity flags).
- Physiologically implausible rows dropped (the raw file contains blood pressures like `-150` or `16020`): kept `ap_hi ∈ [90, 200]`, `ap_lo ∈ [60, 140]`, `ap_hi > ap_lo`, `height ∈ [120, 220]`, `weight ∈ [30, 200]`, `BMI ∈ [12, 60]`.
- Engineered clinical features: **BMI**, **BMI category**, **pulse pressure** (`ap_hi − ap_lo`), **mean arterial pressure** (`ap_lo + (ap_hi − ap_lo)/3`), **ACC/AHA blood-pressure stage**, and a five-year **age band**.

---

## Repository tour

```
.
├── README.md                     <- this file
├── requirements.txt              <- pinned dependencies
├── Procfile                      <- gunicorn config for cloud deploy
├── run.bat                       <- Windows launcher
├── .gitignore
│
├── train_cardio_svm.py           <- end-to-end builder: download -> clean -> figures -> train -> save -> notebook
├── notebook.ipynb                <- end-to-end notebook (41 cells)
├── app.py                        <- Flask backend (loads the single bundle)
├── templates/
│   └── index.html                <- web form (presets + derived-feature panel)
│
├── data/
│   ├── cardio_train_raw.csv      <- cached raw download (git-ignored)
│   └── cardio_clean.csv          <- cleaned modelling dataset
│
├── figures/                      <- 11 PNGs, indigo / vermillion palette
│   ├── 01_target_distribution.png
│   ├── 02_age_distribution.png
│   ├── 03_bmi_boxplot.png
│   ├── 04_systolic_violin.png
│   ├── 05_cholesterol_vs_cardio.png
│   ├── 06_bp_stage_vs_cardio.png
│   ├── 07_correlation_heatmap.png
│   ├── 08_model_comparison.png
│   ├── 09_confusion_matrix.png
│   ├── 10_roc_curve.png
│   └── 11_threshold_tuning.png
│
└── cardio_svm_bundle.pkl         <- ONE artifact: pipeline + features + threshold + metadata
```

The single-bundle layout is the deliberate structural difference from a four-file (`model` / `features` / `threshold` / `metadata`) layout: `app.py` and the notebook load exactly one file.

---

## Environment

- Python 3.12
- scikit-learn 1.8 (`Pipeline`, `ColumnTransformer`, `OneHotEncoder`, `SVC`, `KNeighborsClassifier`)
- xgboost 3.0 (`XGBClassifier`, benchmark model)
- pandas 2.2, numpy 2.4
- matplotlib 3.10, seaborn 0.13
- joblib 1.3 for persistence
- Flask 3.0 + Jinja2 for the web app, gunicorn for production
- nbformat for the notebook builder

Pinned versions in `requirements.txt`.

---

## Modelling pipeline

### 1. Features

Eight numeric features (`Age`, `Height`, `Weight`, `BMI`, `SystolicBP`, `DiastolicBP`, `PulsePressure`, `MeanArterialPressure`) and nine categorical features (`Gender`, `Cholesterol`, `Glucose`, `Smoker`, `Alcohol`, `PhysicallyActive`, `BMICategory`, `BPCategory`, `AgeGroup`).

### 2. Preprocessing (leak-safe)

```
ColumnTransformer
├── numeric    -> SimpleImputer(median) -> StandardScaler
└── categorical-> SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
```

Scaling matters here: both the SVM and KNN are distance/margin based. `handle_unknown='ignore'` keeps the deployed model from crashing on an unseen category.

### 3. Sample and split

An RBF SVM does not scale to ~68k rows, so a **stratified sample of 15,000 rows** is drawn (the classes are near-balanced) and split 80/20 stratified on the target.

### 4. Model bake-off

Three deliberately different families, all in the same pipeline, compared on the test set:

| Model | Role | Key hyperparameters |
|---|---|---|
| SVM (RBF) | **deployed** | `C=2.0, gamma='scale', class_weight='balanced', probability=True` |
| K-Nearest Neighbors | benchmark | `n_neighbors=35, weights='distance'` |
| XGBoost | benchmark | `n_estimators=300, max_depth=4, learning_rate=0.08` |

### 5. Threshold tuning

The SVM's probability cutoff is swept from 0.10 → 0.90 in 0.01 steps; the F1-maximising value (**0.32**) is kept as the operating point.

---

## Results

Deployed model: **SVM (RBF)**.

### Model comparison at the default 0.5 threshold

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.738 | 0.764 | 0.680 | 0.720 | **0.799** |
| **SVM (RBF)** | **0.741** | **0.767** | 0.684 | **0.723** | 0.790 |
| K-Nearest Neighbors | 0.722 | 0.754 | 0.650 | 0.698 | 0.780 |

XGBoost edges ROC-AUC by ~0.009, but the SVM is the best model on accuracy, precision, and F1 at the default cutoff, gives well-separated margin-based probabilities, and serialises to a compact bundle — so it is the deployed screener, with the gradient-boosting and instance-based models kept as honest benchmarks.

### Final operating point (tuned threshold = 0.32)

| Metric | Score |
|---|---:|
| Accuracy | 0.7227 |
| Precision (CVD) | 0.7012 |
| Recall (CVD) | 0.7663 |
| F1-score (CVD) | 0.7323 |
| ROC-AUC | 0.7898 |

Lowering the threshold to 0.32 trades a little precision for recall — appropriate for a screening tool that should miss as few cases as possible.

### Figures

All eleven PNGs in `figures/` share one coordinated palette: indigo/periwinkle for the negative class, vermillion for the positive class, a maroon ROC accent, the `icefire` colormap for the correlation heatmap, and `rocket_r` for the confusion matrix.

---

## Web application

`app.py` loads the single `cardio_svm_bundle.pkl` at import time and serves one page from `templates/index.html`.

What the page does:
- Asks only the **primary** measurements (age, gender, height, weight, two BP readings, cholesterol, glucose, smoking, alcohol, activity). BMI, pulse pressure, mean arterial pressure, BMI category, blood-pressure stage, and age band are **derived server-side** with the exact logic from `train_cardio_svm.py`, so a request equals a row of the training frame.
- Three preset patients (low / moderate / high risk) populate the form with one click.
- The result card flips green (low) or maroon/red (high), shows the probability, the tuned threshold, and the computed features.

The predict route builds a one-row DataFrame in `FEATURES` order, calls `PIPELINE.predict_proba(...)[0, 1]`, and compares to `THRESHOLD`.

---

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000>. On Windows, double-click `run.bat`.

Re-execute the notebook:
```bash
jupyter notebook notebook.ipynb
```

Rebuild everything from scratch (re-downloads the raw CSV, regenerates figures, the bundle, and the notebook):
```bash
python train_cardio_svm.py
```

---

## Reproducibility notes

- One `RANDOM_STATE = 2025` controls the stratified sample, the train/test split, and every model.
- The raw dataset is fetched from a public mirror and cached under `data/`; the cleaned CSV is committed.
- Categorical preprocessing uses `handle_unknown='ignore'`, so the saved pipeline tolerates unseen labels.
- Dependencies are pinned in `requirements.txt`.

---

## Caveats and next steps

What the model isn't:
- The dataset is a single curated survey; it is not a longitudinal clinical cohort.
- "Cardiovascular disease" is a single binary label with no subtype or severity.
- The SVM was trained on a 15k stratified sample for tractability, not the full 68k.

What I would change next:
- Widen the SVM hyper-parameter search (`C`, `gamma`) with cross-validated grid search.
- Add a calibration curve and report Brier score alongside ROC-AUC.
- Surface per-patient feature contributions (e.g., permutation or SHAP) in the UI.
- Cross-validate the threshold sweep instead of tuning on a single split.

---

> Coursework deliverable. Not a clinical tool and not validated for medical decision-making.
