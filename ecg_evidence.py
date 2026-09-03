"""
Independent PTB-XL+/12SL ECG evidence layer for the Cardio-Thermodynamic DSES app.

The ECG model predicts broad PTB-XL diagnostic superclasses (NORM, MI, STTC, CD, HYP).
It does NOT pretend those five classes are equivalent to the app's individual diseases.
Disease-level use is limited to an explicitly defined compatibility layer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional
import json
import numpy as np
import pandas as pd
from lightgbm import Booster

ECG_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

ECG_DISEASE_COMPATIBILITY = {
    "Healthy": {"NORM": 1.0},
    "Myocardial infarction": {"MI": 1.0},
    "Ischemia": {"MI": 0.65, "STTC": 0.35},
    "Ischemic heart disease": {"MI": 0.65, "STTC": 0.35},
    "Coronary artery disease": {"MI": 0.50, "STTC": 0.50},
    "Hypertrophic cardiomyopathy": {"HYP": 1.0},
    "Hypertension": {"HYP": 1.0},
    "Cardiac arrhythmias": {"CD": 1.0},
    "Arrhythmias": {"CD": 1.0},
    "Atrial fibrillation": {"CD": 1.0},
    "Brugada syndrome": {"CD": 1.0},
    "Long QT syndrome": {"CD": 1.0},
    "Short QT syndrome": {"CD": 1.0},
    "Heart block": {"CD": 1.0},
    "Sudden cardiac death": {"CD": 1.0},
    "CPVT": {"CD": 1.0},
}


def _softmax(log_scores: Mapping[str, float]) -> Dict[str, float]:
    names = list(log_scores)
    values = np.asarray([float(log_scores[n]) for n in names], dtype=float)
    values = values - np.max(values)
    ex = np.exp(values)
    denom = float(np.sum(ex))
    if denom <= 0 or not np.isfinite(denom):
        p = 1.0 / max(len(names), 1)
        return {n: p for n in names}
    return {n: float(v / denom) for n, v in zip(names, ex)}


def load_model(model_path: str | Path) -> dict:
    path = Path(model_path)
    with path.open("r", encoding="utf-8") as f:
        artifact = json.load(f)
    if artifact.get("classes") != ECG_CLASSES:
        raise ValueError("ECG artifact classes do not match the expected PTB-XL classes.")
    return artifact


def _prepare_row(feature_df: pd.DataFrame, artifact: dict, age: float, biological_sex: str) -> pd.DataFrame:
    required = list(artifact["feature_columns"])
    missing = [c for c in required if c not in feature_df.columns]
    if missing:
        raise ValueError(
            f"Uploaded ECG row is missing {len(missing)} model features. "
            f"First missing columns: {', '.join(missing[:10])}"
        )

    row = feature_df.iloc[[0]][required].copy()
    row["age"] = float(age)
    row["sex"] = 1.0 if str(biological_sex).strip().lower() == "male" else 0.0

    medians = artifact.get("medians", {})
    for col in required + ["age", "sex"]:
        row[col] = pd.to_numeric(row[col], errors="coerce")
        row[col] = row[col].fillna(float(medians.get(col, 0.0)))
    return row


def predict_superclass_probabilities(feature_df: pd.DataFrame, artifact: dict, age: float, biological_sex: str) -> Dict[str, float]:
    row = _prepare_row(feature_df, artifact, age, biological_sex)
    out = {}
    for cls in ECG_CLASSES:
        booster = Booster(model_str=artifact["models"][cls])
        raw = float(booster.predict(row, num_iteration=booster.current_iteration())[0])
        out[cls] = float(np.clip(raw, 1e-6, 1.0 - 1e-6))
    return out


def disease_ecg_support(disease: str, superclass_probabilities: Mapping[str, float]) -> Optional[float]:
    weights = ECG_DISEASE_COMPATIBILITY.get(disease)
    if not weights:
        return None
    support = sum(float(w) * float(superclass_probabilities.get(c, 0.0)) for c, w in weights.items())
    return float(np.clip(support, 1e-4, 1.0))


def fuse_dses_and_ecg(dses_probabilities: Mapping[str, float], superclass_probabilities: Mapping[str, float], strength: float = 0.75) -> pd.DataFrame:
    scores = {}
    supports = {}
    for disease, p in dses_probabilities.items():
        p = float(np.clip(p, 1e-12, 1.0))
        support = disease_ecg_support(disease, superclass_probabilities)
        supports[disease] = np.nan if support is None else support
        scores[disease] = np.log(p) if support is None else np.log(p) + float(strength) * np.log(max(support, 1e-4))

    combined = _softmax(scores)
    rows = []
    for disease in dses_probabilities:
        rows.append({
            "Disease": disease,
            "DSES Probability": float(dses_probabilities[disease]),
            "ECG Evidence Support": supports[disease],
            "Combined Probability": combined[disease],
            "ECG Evidence Applied": bool(np.isfinite(supports[disease])),
        })
    return pd.DataFrame(rows).sort_values("Combined Probability", ascending=False).reset_index(drop=True)
