"""Flask backend for the Cardio-SVM Cardiovascular Risk Screener.

Loads ONE artifact - `cardio_svm_bundle.pkl` - which holds the fitted
scikit-learn pipeline, the ordered feature list, the tuned probability
threshold, and the UI metadata (dropdown options, defaults, metrics).

The page only asks for the primary measurements a clinic already records
(age, height, weight, blood pressure, cholesterol/glucose bands, lifestyle).
The engineered features the model expects - BMI, pulse pressure, mean arterial
pressure, BMI category, ACC/AHA blood-pressure stage, age band - are derived
server-side with the exact same logic the training script used, so a request
is byte-equivalent to a row from `data/cardio_clean.csv`.

Author: Abdallah Mohamed Mohamed (231002006)
"""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


HERE = Path(__file__).resolve().parent
BUNDLE = joblib.load(HERE / "cardio_svm_bundle.pkl")

PIPELINE = BUNDLE["pipeline"]
FEATURES = BUNDLE["feature_columns"]
THRESHOLD = float(BUNDLE["threshold"])
META = BUNDLE["metadata"]
OPTIONS = META["options"]

# Primary inputs the user actually fills in (everything else is derived).
PRIMARY_NUMERIC = {
    "Age": 50,
    "Height": 168,
    "Weight": 74,
    "SystolicBP": 120,
    "DiastolicBP": 80,
}
PRIMARY_CATEGORICAL = [
    "Gender",
    "Cholesterol",
    "Glucose",
    "Smoker",
    "Alcohol",
    "PhysicallyActive",
]

SENSIBLE_CATEGORICAL = {
    "Gender": "Female",
    "Cholesterol": "Normal",
    "Glucose": "Normal",
    "Smoker": "No",
    "Alcohol": "No",
    "PhysicallyActive": "Yes",
}


# ---------------------------------------------------------------------------
# Feature engineering - identical to train_cardio_svm.py
# ---------------------------------------------------------------------------
def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal weight"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def bp_category(sys_bp: float, dia_bp: float) -> str:
    if sys_bp >= 180 or dia_bp >= 120:
        return "Hypertensive crisis"
    if sys_bp >= 140 or dia_bp >= 90:
        return "Hypertension stage 2"
    if sys_bp >= 130 or dia_bp >= 80:
        return "Hypertension stage 1"
    if sys_bp >= 120:
        return "Elevated"
    return "Normal"


def age_group(years: float) -> str:
    if years < 40:
        return "30-39"
    if years < 45:
        return "40-44"
    if years < 50:
        return "45-49"
    if years < 55:
        return "50-54"
    if years < 60:
        return "55-59"
    return "60+"


def _number(form, name, fallback):
    raw = form.get(name)
    if raw in (None, ""):
        return float(fallback)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(fallback)


def _category(form, name):
    allowed = OPTIONS[name]
    value = form.get(name)
    if value in allowed:
        return value
    preferred = SENSIBLE_CATEGORICAL.get(name)
    return preferred if preferred in allowed else allowed[0]


def assemble_row(form) -> dict:
    """Build the full 17-feature row from the primary inputs."""
    age = _number(form, "Age", PRIMARY_NUMERIC["Age"])
    height = _number(form, "Height", PRIMARY_NUMERIC["Height"])
    weight = _number(form, "Weight", PRIMARY_NUMERIC["Weight"])
    sys_bp = _number(form, "SystolicBP", PRIMARY_NUMERIC["SystolicBP"])
    dia_bp = _number(form, "DiastolicBP", PRIMARY_NUMERIC["DiastolicBP"])

    height = max(height, 1.0)
    bmi = round(weight / (height / 100) ** 2, 1)

    row = {
        "Age": age,
        "Height": height,
        "Weight": weight,
        "BMI": bmi,
        "SystolicBP": sys_bp,
        "DiastolicBP": dia_bp,
        "PulsePressure": sys_bp - dia_bp,
        "MeanArterialPressure": round(dia_bp + (sys_bp - dia_bp) / 3, 1),
        "BMICategory": bmi_category(bmi),
        "BPCategory": bp_category(sys_bp, dia_bp),
        "AgeGroup": age_group(age),
    }
    for col in PRIMARY_CATEGORICAL:
        row[col] = _category(form, col)
    return row


def render_page(**extra):
    context = {
        "options": OPTIONS,
        "defaults": PRIMARY_NUMERIC,
        "categorical_defaults": SENSIBLE_CATEGORICAL,
        "primary_categorical": PRIMARY_CATEGORICAL,
        "metrics": META["metrics"],
        "rows": META["rows"],
        "deployed_model": META["metrics"]["deployed_model"],
        "threshold": round(THRESHOLD * 100, 1),
    }
    context.update(extra)
    return render_template("index.html", **context)


app = Flask(__name__)


@app.route("/")
def home():
    return render_page()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        row = assemble_row(request.form)
        frame = pd.DataFrame([row])[FEATURES]
        probability = float(PIPELINE.predict_proba(frame)[0, 1])
        is_high = probability >= THRESHOLD
        return render_page(
            prediction="Higher CVD Risk" if is_high else "Lower CVD Risk",
            probability=round(probability * 100, 1),
            risk="high" if is_high else "low",
            derived=row,
            form_data=request.form,
        )
    except Exception as exc:  # noqa: BLE001 - surface any error in the UI
        return render_page(error=str(exc), form_data=request.form)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
