"""Cardiac DSES 28-class ML layer.

This module is deliberately Streamlit-free so it can be imported from
Google Colab, scripts, tests, or the Streamlit app without requiring the
Streamlit package.
"""
from pathlib import Path
import json
import importlib.util
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent

# Robustly locate the Digital Twin module regardless of whether the bundle
# calls it digital_twin.py or digital_twin_final.py.
_DT_CANDIDATES = [ROOT / "digital_twin.py", ROOT / "digital_twin_final.py"]
_DT_PATH = next((p for p in _DT_CANDIDATES if p.exists()), None)
if _DT_PATH is None:
    raise FileNotFoundError(
        "Digital Twin module not found. Expected digital_twin.py or digital_twin_final.py "
        f"inside {ROOT}."
    )

_spec = importlib.util.spec_from_file_location("digital_twin_runtime", _DT_PATH)
dt = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(dt)

# Model artifacts.
RF_PATH = ROOT / "disease_dses_rf.joblib"
RF_COLS = ROOT / "disease_dses_model_columns.joblib"
RF_MED = ROOT / "disease_dses_model_medians.joblib"
CLF_PATH = ROOT / "final_28class_dses_classifier.joblib"
FEATURE_COLS_PATH = ROOT / "final_28class_feature_columns.joblib"
META_PATH = ROOT / "final_28class_dses_meta.json"

DISEASES = list(dt.DISEASE_REACTION_MAP.keys())

# Keep the raw clinical layer intentionally compact.
RAW_FIELDS = [
    "Age_years",
    "Biological_Sex",
    "Weight_kg",
    "Heart_Rate_bpm",
    "Systolic_BP_mmHg",
    "Diastolic_BP_mmHg",
    "EDV_mL",
    "ESV_mL",
    "PAWP_mmHg",
    "Myocardial_Mass_g",
    "Fasting_Blood_Sugar",
    "Serum_Cholesterol",
]


def _patient_dict(r: pd.Series) -> dict:
    return {
        "age": float(r["Age_years"]),
        "sex": str(r["Biological_Sex"]),
        "height_cm": float(r["Height_cm"]),
        "weight_kg": float(r["Weight_kg"]),
        "hr": float(r["Heart_Rate_bpm"]),
        "sbp": float(r["Systolic_BP_mmHg"]),
        "dbp": float(r["Diastolic_BP_mmHg"]),
        "edv": float(r["EDV_mL"]),
        "esv": float(r["ESV_mL"]),
        "pawp": float(r["PAWP_mmHg"]),
        "myocardial_mass_g": float(r["Myocardial_Mass_g"]),
        "temp_c": float(r["Core_Temp_C"]),
        "spo2": float(r["SpO2_pct"]),
        "ph": float(r["pH"]),
        "fbs": float(r["Fasting_Blood_Sugar"]),
        "chol": float(r["Serum_Cholesterol"]),
        "chest_pain": str(r.get("Chest_Pain_Type", "asymptomatic")),
        "rest_ecg": str(r.get("Resting_ECG", "Normal")),
    }


def _expected_dses(df: pd.DataFrame) -> np.ndarray:
    """Predict expected DSES for every patient x 28 disease pair."""
    for path in (RF_PATH, RF_COLS, RF_MED):
        if not path.exists():
            raise FileNotFoundError(f"Required expected-DSES artifact not found: {path}")

    rf = joblib.load(RF_PATH)
    columns = joblib.load(RF_COLS)
    medians = joblib.load(RF_MED)

    rows = []
    for _, r in df.iterrows():
        bsa = float(np.sqrt(float(r["Height_cm"]) * float(r["Weight_kg"]) / 3600.0))
        for disease in DISEASES:
            rows.append({
                "Disease": disease,
                "Age (mean)": float(r["Age_years"]),
                "Biological Sex": str(r["Biological_Sex"]),
                "Height (cm, mean)": float(r["Height_cm"]),
                "Weight (kg, mean)": float(r["Weight_kg"]),
                "Body Surface Area (BSA)": bsa,
                "Heart Rate (bpm, mean/resting)": float(r["Heart_Rate_bpm"]),
                "Systolic Blood Pressure (mmHg, mean)": float(r["Systolic_BP_mmHg"]),
                "Diastolic Blood Pressure (mmHg, mean)": float(r["Diastolic_BP_mmHg"]),
                "End-Diastolic Volume (EDV, mL, representative/mean)": float(r["EDV_mL"]),
                "End-Systolic Volume (ESV, mL, representative/mean)": float(r["ESV_mL"]),
                "Pulmonary Artery Wedge Pressure (PAWP, mmHg, representative/mean)": float(r["PAWP_mmHg"]),
                "Myocardial Mass (g)": float(r["Myocardial_Mass_g"]),
                "Core Body Temperature (°C, mean)": float(r["Core_Temp_C"]),
                "SpO₂ (% , mean)": float(r["SpO2_pct"]),
                "Arterial pH (pHₐ, mean)": float(r["pH"]),
                "Serum Cholesterol (mg/dL, mean)": float(r["Serum_Cholesterol"]),
                "Fasting Blood Sugar (mg/dL, mean)": float(r["Fasting_Blood_Sugar"]),
                "Chest Pain Type": str(r.get("Chest_Pain_Type", "asymptomatic")),
                "Resting ECG Result": str(r.get("Resting_ECG", "Normal")),
            })

    X = pd.DataFrame(rows)
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    X = X.reindex(columns=columns, fill_value=0.0)

    for col, value in medians.items():
        if col in X.columns:
            X[col] = X[col].fillna(value)

    pred = rf.predict(X)
    return pred.reshape(len(df), len(DISEASES))


def build_features(df: pd.DataFrame, refs=None) -> pd.DataFrame:
    """Build the final DSES-centered feature matrix."""
    required = [
        "Age_years", "Biological_Sex", "Height_cm", "Weight_kg",
        "Heart_Rate_bpm", "Systolic_BP_mmHg", "Diastolic_BP_mmHg",
        "EDV_mL", "ESV_mL", "SpO2_pct", "pH", "Fasting_Blood_Sugar",
        "Serum_Cholesterol", "PAWP_mmHg", "Core_Temp_C",
        "Myocardial_Mass_g", "Chest_Pain_Type", "Resting_ECG",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    if refs is None:
        ref_candidates = [
            ROOT / "Cardiac_Disease_Synchronized_Dataset_Patient_1_Literature_Conditioned_CLEAN (4).xlsx",
            ROOT / "Cardiac_Disease_Synchronized_Dataset_Patient_1_Literature_Conditioned_CLEAN (4)_2.xlsx",
        ]
        ref_path = next((p for p in ref_candidates if p.exists()), None)
        if ref_path is None:
            raise FileNotFoundError(
                "Reference workbook not found in model directory. "
                "Pass refs=dt.build_training_references(reference_path)."
            )
        refs = dt.build_training_references(str(ref_path))

    expected = _expected_dses(df)
    rows = []

    for i, (_, r) in enumerate(df.iterrows()):
        patient = _patient_dict(r)
        dt_result = dt.compute_patient_dses_scores(patient, refs)
        table = dt_result["disease_table"].set_index("Disease")
        row = {}

        # Patient DSES, expected DSES, residual: 28 values each.
        for j, disease in enumerate(DISEASES):
            patient_dses = float(table.loc[disease, "Patient DSES (1-100)"])
            expected_dses = float(expected[i, j])
            row[f"P_{disease}"] = patient_dses
            row[f"E_{disease}"] = expected_dses
            row[f"R_{disease}"] = patient_dses - expected_dses

        # Global reaction-stress summaries across mapped disease/reaction pairs.
        details = []
        for disease in DISEASES:
            for reaction in dt.DISEASE_REACTION_MAP[disease]:
                details.append(dt_result["reaction_details"][disease][reaction])

        for key in [
            "Metabolic Stress",
            "Mechanical Stress",
            "Thermodynamic Stress",
            "ATP Stress",
            "Entropy Stress",
        ]:
            row[f"G_{key}"] = float(np.mean([d[key] for d in details]))

        common = dt_result["common"]
        for key in ["EF", "MAP", "CO", "RPP", "SV"]:
            row[f"G_{key}"] = float(common[key])

        rows.append(row)

    X_dses = pd.DataFrame(rows)
    X_dses = X_dses.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Small raw clinical layer; these are the same patient inputs, not target columns.
    X_raw = df[RAW_FIELDS].copy()
    X_raw = pd.get_dummies(
        X_raw,
        columns=["Biological_Sex"],
        drop_first=False,
        dtype=float,
    )
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    X_raw = X_raw.fillna(X_raw.median(numeric_only=True)).fillna(0.0)
    X_raw = X_raw.reset_index(drop=True)

    return pd.concat(
        [X_dses.reset_index(drop=True), X_raw],
        axis=1,
    )


def make_classifier() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        max_features=0.5,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )


def train_and_save(dev_csv: str, refs) -> dict:
    dev = pd.read_csv(dev_csv)
    X = build_features(dev, refs)
    y = dev["Diagnosed_Disease"].astype(str)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    clf = make_classifier()
    scores = cross_val_score(
        clf,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    clf.fit(X, y)
    joblib.dump(clf, CLF_PATH)
    joblib.dump(list(X.columns), FEATURE_COLS_PATH)

    meta = {
        "classes": list(clf.classes_),
        "development_rows": int(len(dev)),
        "feature_count": int(X.shape[1]),
        "cv_mean_accuracy": float(scores.mean()),
        "cv_fold_accuracy": [float(x) for x in scores],
        "holdout_used_for_training": False,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def predict_patient(patient: dict, refs=None) -> pd.DataFrame:
    """Runtime inference for the Streamlit app or a standalone script."""
    if not CLF_PATH.exists() or not FEATURE_COLS_PATH.exists():
        raise FileNotFoundError(
            "Final classifier artifacts are missing. Train the final model first."
        )

    if refs is None:
        ref_candidates = [
            ROOT / "Cardiac_Disease_Synchronized_Dataset_Patient_1_Literature_Conditioned_CLEAN (4).xlsx",
            ROOT / "Cardiac_Disease_Synchronized_Dataset_Patient_1_Literature_Conditioned_CLEAN (4)_2.xlsx",
        ]
        ref_path = next((p for p in ref_candidates if p.exists()), None)
        if ref_path is None:
            raise FileNotFoundError("Reference workbook not found.")
        refs = dt.build_training_references(str(ref_path))

    clf = joblib.load(CLF_PATH)
    columns = joblib.load(FEATURE_COLS_PATH)

    row = pd.DataFrame([{
        "Age_years": patient["age"],
        "Biological_Sex": patient["sex"],
        "Height_cm": patient["height_cm"],
        "Weight_kg": patient["weight_kg"],
        "Heart_Rate_bpm": patient["hr"],
        "Systolic_BP_mmHg": patient["sbp"],
        "Diastolic_BP_mmHg": patient["dbp"],
        "EDV_mL": patient["edv"],
        "ESV_mL": patient["esv"],
        "SpO2_pct": patient["spo2"],
        "pH": patient["ph"],
        "Fasting_Blood_Sugar": patient["fbs"],
        "Serum_Cholesterol": patient["chol"],
        "Chest_Pain_Type": patient.get("chest_pain", "asymptomatic"),
        "PAWP_mmHg": patient["pawp"],
        "Core_Temp_C": patient["temp_c"],
        "Myocardial_Mass_g": patient["myocardial_mass_g"],
        "Resting_ECG": patient.get("rest_ecg", "Normal"),
    }])

    X = build_features(row, refs)
    X = X.reindex(columns=columns, fill_value=0.0)

    probabilities = clf.predict_proba(X)[0]
    order = np.argsort(probabilities)[::-1]

    return pd.DataFrame({
        "Disease": clf.classes_[order],
        "Probability": probabilities[order],
    })


# Compatibility alias for app code that may use a different function name.
predict_disease = predict_patient
