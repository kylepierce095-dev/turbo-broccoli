from pathlib import Path
import os
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Cardio-Thermodynamic Digital Twin",
    layout="wide",
)

st.title("Cardio-Thermodynamic Digital Twin")

# ============================================================
# CONSTANTS / SAFE NUMERIC HELPERS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
SOFT_FLOOR = 1e-6
DG_FLOOR = 1.0


def soft_linear_floor(value, floor=SOFT_FLOOR):
    """Replace invalid/exact-zero numeric values with a small positive floor."""
    if np.isscalar(value):
        try:
            x = float(value)
        except (TypeError, ValueError):
            return float(floor)
        if not np.isfinite(x) or x == 0:
            return float(floor)
        return x

    s = pd.to_numeric(value, errors="coerce").astype(float).copy()
    invalid = s.isna() | ~np.isfinite(s)
    s.loc[invalid] = floor
    s.loc[s == 0] = floor
    return s


def soft_scalar(value, floor=SOFT_FLOOR):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(floor)
    if not np.isfinite(x) or x == 0:
        return float(floor)
    return x


def soft_denominator(value, floor=SOFT_FLOOR):
    """Protect denominator before division."""
    if isinstance(value, pd.Series):
        s = pd.to_numeric(value, errors="coerce")
        return s.where(
            s.notna() & np.isfinite(s) & (s != 0),
            floor,
        )
    return soft_scalar(value, floor=floor)


def soft_positive_input(value, floor=SOFT_FLOOR):
    """Protect values used in log/power operations."""
    if isinstance(value, pd.Series):
        s = pd.to_numeric(value, errors="coerce")
        return s.where(
            s.notna() & np.isfinite(s) & (s > 0),
            floor,
        )

    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(floor)

    return x if np.isfinite(x) and x > 0 else float(floor)


def load_joblib_required(filename):
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required model artifact is missing: {filename}\n"
            f"Expected location: {path}"
        )
    return joblib.load(path)


# ============================================================
# LOAD DSES / DIGITAL-TWIN ARTIFACTS
# ============================================================

try:
    _signature_ref = load_joblib_required(
        "disease_entropy_signature_reference.pkl"
    )
    refs = load_joblib_required("dt_reference_values.pkl")

    DSES_RF_MODEL = load_joblib_required("dses_rf_model.joblib")
    DSES_MODEL_COLUMNS = load_joblib_required("dses_model_columns.joblib")
    DSES_MODEL_MEDIANS = load_joblib_required("dses_model_medians.joblib")
    DSES_CATEGORICAL_COLS = load_joblib_required(
        "dses_categorical_columns.joblib"
    )

except Exception as exc:
    st.error("The Digital Twin model artifacts could not be loaded.")
    st.code(str(exc))
    st.info(
        "Place the required .pkl/.joblib files in the same GitHub folder as app.py."
    )
    st.stop()


T = float(_signature_ref.get("temperature_K", 310.15))
DSES_MAPPING = _signature_ref.get("mapping", {})
REACTION_REFS = _signature_ref.get("reaction_references", {})
DISEASE_REACTION_FACTORS = _signature_ref.get(
    "disease_reaction_factors", {}
)

if not DSES_MAPPING:
    st.error("The disease entropy signature mapping is empty.")
    st.stop()

# ============================================================
# LOAD DISEASE-SPECIFIC ONE-VS-REST MODELS
# ============================================================

_DSES_MODEL_DIR = BASE_DIR / "disease_rf_models"
_DSES_MODELS = {}
_missing_disease_models = []

for disease in sorted(DSES_MAPPING):
    safe_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(disease),
    ).strip("_")

    model_path = _DSES_MODEL_DIR / f"{safe_name}.joblib"

    if model_path.exists():
        try:
            _DSES_MODELS[disease] = joblib.load(model_path)
        except Exception as exc:
            st.error(
                f"Could not load disease-specific model for "
                f"'{disease}': {exc}"
            )
            st.stop()
    else:
        _missing_disease_models.append(str(model_path))

if _missing_disease_models:
    st.error(
        "One or more disease-specific DSES models are missing."
    )
    st.code("\n".join(_missing_disease_models))
    st.info(
        "Upload the complete disease_rf_models/ directory "
        "to the GitHub repository before running disease prediction."
    )
    st.stop()


# ============================================================
# PATIENT INPUTS
# ============================================================

st.header("Patient Input Parameters")

c1, c2, c3 = st.columns(3)

with c1:
    hr = st.number_input(
        "Heart Rate (bpm)",
        min_value=20.0,
        max_value=250.0,
        value=72.0,
        step=1.0,
    )

    sbp = st.number_input(
        "Systolic BP (mmHg)",
        min_value=50.0,
        max_value=250.0,
        value=120.0,
        step=1.0,
    )

    dbp = st.number_input(
        "Diastolic BP (mmHg)",
        min_value=30.0,
        max_value=150.0,
        value=80.0,
        step=1.0,
    )

with c2:
    edv = st.number_input(
        "End-Diastolic Volume (mL)",
        min_value=20.0,
        max_value=500.0,
        value=140.0,
        step=1.0,
    )

    esv = st.number_input(
        "End-Systolic Volume (mL)",
        min_value=5.0,
        max_value=400.0,
        value=60.0,
        step=1.0,
    )

    ph = st.number_input(
        "pH",
        min_value=6.5,
        max_value=8.0,
        value=7.40,
        step=0.01,
    )

with c3:
    spo2 = st.number_input(
        "SpO₂ (%)",
        min_value=50.0,
        max_value=100.0,
        value=98.0,
        step=1.0,
    )

    fbs = st.number_input(
        "Fasting Blood Sugar",
        min_value=20.0,
        max_value=600.0,
        value=90.0,
        step=1.0,
    )

    chol = st.number_input(
        "Serum Cholesterol",
        min_value=50.0,
        max_value=600.0,
        value=180.0,
        step=1.0,
    )

# ============================================================
# ADDITIONAL CLINICAL PARAMETERS REQUIRED BY ML DSES MODEL
# ============================================================

st.header("Additional Clinical Parameters")

c4, c5, c6 = st.columns(3)

with c4:
    age = st.number_input(
        "Age (years)",
        min_value=18.0,
        max_value=81.0,
        value=45.0,
        step=1.0,
    )

    sex = st.selectbox(
        "Biological Sex",
        ["Female", "Male"],
    )

    height = st.number_input(
        "Height (cm)",
        min_value=150.0,
        max_value=183.0,
        value=165.0,
        step=1.0,
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=45.0,
        max_value=97.0,
        value=70.0,
        step=1.0,
    )

with c5:
    cp = st.selectbox(
        "Chest Pain Type",
        [
            "typical angina",
            "atypical angina",
            "non-anginal pain",
            "asymptomatic",
        ],
    )

    exang = st.selectbox(
        "Exercise-Induced Angina",
        ["No", "Yes"],
    )

    oldpeak = st.number_input(
        "ST Depression (oldpeak)",
        min_value=0.05,
        max_value=4.34,
        value=1.0,
        step=0.05,
    )

    slope = st.selectbox(
        "Slope of ST Segment",
        ["Down", "Flat", "Up"],
    )

with c6:
    ca = st.selectbox(
        "Number of Major Vessels (ca)",
        [0, 1, 2, 3],
    )

    thal = st.selectbox(
        "Thallium Stress Test Result (thal)",
        ["Normal"],
    )

    restecg = st.selectbox(
        "Resting ECG Result",
        [
            "Normal",
            "ST-T wave abnormality",
            "Left ventricular hypertrophy",
        ],
    )

    pawp = st.number_input(
        "PAWP (mmHg)",
        min_value=5.06,
        max_value=24.99,
        value=12.0,
        step=0.5,
    )

    core_temp = st.number_input(
        "Core Body Temperature (°C)",
        min_value=36.3,
        max_value=37.83,
        value=36.9,
        step=0.05,
    )

    # Fixed: default must be inside the allowed range.
    myo_mass_input = st.number_input(
        "Myocardial Mass (g)",
        min_value=80.26,
        max_value=239.89,
        value=150.0,
        step=1.0,
    )

disease_choice = st.selectbox(
    "Disease to inspect in the detailed biochemical pathway view (optional)",
    sorted(DSES_MAPPING.keys()),
)


# ============================================================
# ML INPUT CONSTRUCTION
# ============================================================

def build_ml_input_row(reaction_name):
    row = {
        "Age (mean)": age,
        "Biological Sex": sex,
        "Height (cm, mean)": height,
        "Weight (kg, mean)": weight,
        "Body Surface Area (BSA)": np.sqrt(
            (height * weight) / 3600.0
        ),
        "Heart Rate (bpm, mean/resting)": hr,
        "Systolic Blood Pressure (mmHg, mean)": sbp,
        "Diastolic Blood Pressure (mmHg, mean)": dbp,
        "End-Diastolic Volume (EDV, mL, representative/mean)": edv,
        "End-Systolic Volume (ESV, mL, representative/mean)": esv,
        "EDV/ESV ratio": edv / soft_denominator(esv),
        "Pulmonary Artery Wedge Pressure (PAWP, mmHg, representative/mean)": pawp,
        "Myocardial Mass (g)": myo_mass_input,
        "Core Body Temperature (°C, mean)": core_temp,
        "SpO₂ (% , mean)": spo2,
        "Arterial pH (pHₐ, mean)": ph,
        "Chest Pain Type": cp,
        "Exercise-Induced Angina": exang,
        "ST Depression (oldpeak)": oldpeak,
        "Slope of ST Segment": slope,
        "Number of Major Vessels (ca)": ca,
        "Thallium Stress Test Result (thal)": thal,
        "Serum Cholesterol (mg/dL, mean)": chol,
        "Fasting Blood Sugar (mg/dL, mean)": fbs,
        "Resting ECG Result": restecg,
        "Disease": disease_choice,
        "Biochemical Reaction": reaction_name,
    }
    return pd.DataFrame([row])


def predict_dses(raw_row_df):
    df_enc = pd.get_dummies(
        raw_row_df,
        columns=[
            c for c in DSES_CATEGORICAL_COLS
            if c in raw_row_df.columns
        ],
        drop_first=True,
        dtype=float,
    )

    df_aligned = df_enc.reindex(
        columns=DSES_MODEL_COLUMNS,
        fill_value=0.0,
    )

    for col in DSES_MODEL_COLUMNS:
        if df_aligned[col].isna().any():
            df_aligned[col] = df_aligned[col].fillna(
                DSES_MODEL_MEDIANS.get(col, 0.0)
            )

    prediction = DSES_RF_MODEL.predict(df_aligned)[0]
    return float(prediction)


run_dt = st.button(
    "RUN DIGITAL TWIN",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PATIENT CALCULATION
# ============================================================

if run_dt:

    if edv <= esv:
        st.error("EDV must be greater than ESV.")
        st.stop()

    if sbp <= dbp:
        st.error("SBP must be greater than DBP.")
        st.stop()

    # --------------------------------------------------------
    # Hemodynamics
    # --------------------------------------------------------

    sv = edv - esv

    edv_safe = soft_denominator(edv)
    ef = sv / edv_safe

    map_pressure = dbp + (sbp - dbp) / 3.0

    co = sv * hr
    rpp = hr * sbp
    lvsw = sv * map_pressure * 0.000133322

    # --------------------------------------------------------
    # Metabolic layer
    # --------------------------------------------------------

    RPP_ref = 10000
    LVSW_ref = 0.9
    MVO2_rest = 8.0
    O2_ENERGY = 20.2
    MYOCARDIAL_MASS = myo_mass_input

    rpp_component = rpp / soft_denominator(RPP_ref)
    lvsw_component = lvsw / soft_denominator(LVSW_ref)

    metabolic_demand = (
        0.7 * rpp_component
        + 0.3 * lvsw_component
    )

    mvo2 = MVO2_rest * metabolic_demand

    o2_consumption = (
        mvo2 * MYOCARDIAL_MASS / 100.0
    )

    chemical_power = (
        o2_consumption * O2_ENERGY / 60.0
    )

    mechanical_power = (
        lvsw * hr / 60.0
    )

    heat_production = max(
        chemical_power - mechanical_power,
        0.0,
    )

    # --------------------------------------------------------
    # ATP
    # --------------------------------------------------------

    ATP_ENERGY = 30500
    ATP_COUPLING_EFFICIENCY = 0.60

    atp_production = (
        chemical_power
        * 60.0
        * ATP_COUPLING_EFFICIENCY
        / soft_denominator(ATP_ENERGY)
    )

    mechanical_energy = mechanical_power * 60.0

    atp_utilization = (
        mechanical_energy
        / soft_denominator(ATP_ENERGY)
    )

    atp_fraction = (
        atp_utilization
        / soft_denominator(atp_production)
    )

    atp_balance = (
        atp_production - atp_utilization
    )

    # --------------------------------------------------------
    # Thermodynamic proxy layer
    # --------------------------------------------------------

    R = 8.314

    safe_atp_balance = soft_positive_input(atp_balance)

    Q_metabolism = (
        ((fbs + 1.0) * chol)
        / soft_denominator(spo2 * safe_atp_balance)
    )
    Q_metabolism = soft_positive_input(Q_metabolism)

    dg_metabolism = (
        -2870000
        + R * T * np.log(Q_metabolism)
    )

    Q_atp = soft_positive_input(
        atp_utilization
        / soft_denominator(atp_production)
    )

    dg_atp = (
        -30500
        + R * T * np.log(Q_atp)
    )

    pH_factor = 10 ** (7.4 - ph)

    Q_ion = soft_positive_input(
        atp_fraction * pH_factor
    )

    dg_ion = (
        -50000
        + R * T * np.log(Q_ion)
    )

    calcium_factor = (
        ph * hr / soft_denominator(hr)
    )

    Q_calcium = soft_positive_input(
        atp_fraction * calcium_factor
    )

    dg_calcium = (
        -50000
        + R * T * np.log(Q_calcium)
    )

    redox_factor = spo2 / 100.0

    Q_redox = soft_positive_input(
        atp_fraction * redox_factor
    )

    dg_redox = (
        -220000
        + R * T * np.log(Q_redox)
    )

    no_factor = (
        ph * spo2 / 100.0
    )

    Q_no = soft_positive_input(
        atp_fraction * no_factor
    )

    dg_no = (
        -100000
        + R * T * np.log(Q_no)
    )

    total_dg = (
        dg_metabolism
        + dg_atp
        + dg_ion
        + dg_calcium
        + dg_redox
        + dg_no
    )

    # --------------------------------------------------------
    # ENTROPY FROM ENERGY / GIBBS BALANCE
    # --------------------------------------------------------

    DG_FLOOR = 1.0

    # Energy terms in J/min
    chem_input_j = chemical_power * 60.0
    mech_work_j = mechanical_power * 60.0
    heat_j = heat_production * 60.0

    # Existing pathway energy allocations
    ion_energy = mech_work_j * atp_fraction
    calcium_energy = mech_work_j * atp_fraction
    redox_energy = chem_input_j * atp_fraction
    no_energy = chem_input_j * atp_fraction

    dG_dn_sum = 0.0

    dn_metab = (
        chem_input_j
        / max(abs(dg_metabolism), DG_FLOOR)
    )
    dG_dn_sum += dg_metabolism * dn_metab

    dn_atp = atp_utilization
    dG_dn_sum += dg_atp * dn_atp

    dn_ion = (
        ion_energy
        / max(abs(dg_ion), DG_FLOOR)
    )
    dG_dn_sum += dg_ion * dn_ion

    dn_calcium = (
        calcium_energy
        / max(abs(dg_calcium), DG_FLOOR)
    )
    dG_dn_sum += dg_calcium * dn_calcium

    dn_redox = (
        redox_energy
        / max(abs(dg_redox), DG_FLOOR)
    )
    dG_dn_sum += dg_redox * dn_redox

    dn_no = (
        no_energy
        / max(abs(dg_no), DG_FLOOR)
    )
    dG_dn_sum += dg_no * dn_no

    entropy_flow = (
        chem_input_j
        + mech_work_j
        - dG_dn_sum
    ) / T

    entropy = entropy_flow * 1.0

    # --------------------------------------------------------
    # Other ML stress features
    # --------------------------------------------------------

    entropy_stress = (
        entropy_flow
        / soft_denominator(refs["Entropy_Flow_median"])
    )

    metabolic_stress = (
        mvo2
        / soft_denominator(refs["MVO2_median"])
    )

    mechanical_stress = (
        mechanical_power
        / soft_denominator(refs["Mechanical_Power_median"])
    )

    thermodynamic_stress = (
        abs(total_dg)
        / soft_denominator(refs["Total_DG_abs_median"])
    )

    atp_stress = (
        atp_fraction
        / soft_denominator(
            refs["ATP_Utilization_Fraction_median"]
        )
    )

    # --------------------------------------------------------
    # DISEASE-SPECIFIC ENTROPY SIGNATURE
    # Same scalar definition as the training pipeline.
    # --------------------------------------------------------

    def _patient_reaction_dses_components():
        reaction_values = {
            "Metabolism": {
                "dg": dg_metabolism,
                "energy": chem_input_j,
            },
            "ATP Utilization": {
                "dg": dg_atp,
                "energy": mechanical_energy,
            },
            "Ion Transport": {
                "dg": dg_ion,
                "energy": ion_energy,
            },
            "Calcium Handling": {
                "dg": dg_calcium,
                "energy": calcium_energy,
            },
            "Redox Metabolism": {
                "dg": dg_redox,
                "energy": redox_energy,
            },
            "Nitric Oxide Metabolism": {
                "dg": dg_no,
                "energy": no_energy,
            },
        }

        result = {}

        for reaction, vals in reaction_values.items():
            if reaction not in REACTION_REFS:
                continue

            ref = REACTION_REFS[reaction]

            dg_val = float(vals["dg"])
            energy_val = float(vals["energy"])

            dn_val = (
                energy_val
                / max(abs(dg_val), DG_FLOOR)
            )

            reaction_entropy_flow = max(
                abs(dg_val)
                * dn_val
                / T,
                SOFT_FLOOR,
            )

            r_met = (
                mvo2
                / max(ref["mvo2_median"], SOFT_FLOOR)
            )

            r_mech = (
                mechanical_power
                / max(
                    ref["mechanical_power_median"],
                    SOFT_FLOOR,
                )
            )

            r_thermo = (
                abs(dg_val)
                / max(
                    ref["dg_abs_median"],
                    SOFT_FLOOR,
                )
            )

            r_atp = (
                atp_fraction
                / max(
                    ref["atp_fraction_median"],
                    SOFT_FLOOR,
                )
            )

            r_entropy = (
                reaction_entropy_flow
                / max(
                    ref["entropy_flow_median"],
                    SOFT_FLOOR,
                )
            )

            result[reaction] = {
                "entropy_ratio": max(
                    r_entropy,
                    SOFT_FLOOR,
                ),
                "stress_mean": max(
                    float(
                        np.mean(
                            [
                                r_met,
                                r_mech,
                                r_thermo,
                                r_atp,
                                r_entropy,
                            ]
                        )
                    ),
                    SOFT_FLOOR,
                ),
            }

        return result

    _patient_components = (
        _patient_reaction_dses_components()
    )

    # ========================================================
    # DISEASE-SPECIFIC DSES + PATIENT-CONDITIONED EXPECTED DSES
    # ========================================================
    # IMPORTANT: the original DSES calculation above is preserved.
    # We use its raw value for distance calculations and the original
    # 1-100 representation for the already-trained one-dimensional RFs.
    patient_dses = {}
    patient_raw_dses = {}
    expected_dses = {}
    expected_dses_uncertainty = {}

    # Reference range used by the original DSES representation.
    dses_min_ref = float(_signature_ref.get("raw_dses_min", 0.0))
    dses_max_ref = float(_signature_ref.get("raw_dses_max", 1.0))
    dses_low = float(_signature_ref.get("dses_range_low", 1.0))
    dses_high = float(_signature_ref.get("dses_range_high", 100.0))

    if dses_max_ref - dses_min_ref <= SOFT_FLOOR:
        dses_max_ref = dses_min_ref + 1.0

    def scale_original_dses(raw_dses):
        """Preserve the original 1-100 DSES representation for RF input."""
        value = dses_low + (
            (raw_dses - dses_min_ref)
            * (dses_high - dses_low)
            / (dses_max_ref - dses_min_ref)
        )
        return float(np.clip(value, dses_low, dses_high))

    def predict_expected_dses_for_disease(disease):
        """Predict expected DSES separately for this patient and disease.

        The regression model is asked about each biochemical reaction mapped
        to the disease, then the reaction-level predictions are averaged.
        Random-forest tree spread is retained as an uncertainty proxy.
        """
        reactions = DSES_MAPPING.get(disease, [])
        predictions = []
        spreads = []

        for reaction_name in reactions:
            raw_row = build_ml_input_row_for_disease(
                disease, reaction_name
            )
            pred = predict_dses(raw_row)
            predictions.append(float(pred))

            # Ensemble spread: not a clinical error estimate; it is an
            # internal uncertainty proxy available from the RF regressor.
            try:
                tree_preds = np.array([
                    estimator.predict(
                        _prepare_ml_dataframe(raw_row)
                    )[0]
                    for estimator in DSES_RF_MODEL.estimators_
                ],
                dtype=float)
                tree_preds = tree_preds[np.isfinite(tree_preds)]
                if tree_preds.size > 1:
                    spreads.append(float(np.std(tree_preds, ddof=1)))
            except Exception:
                pass

        if not predictions:
            return SOFT_FLOOR, 1.0

        expected = float(np.mean(predictions))
        uncertainty = float(np.mean(spreads)) if spreads else 1.0
        uncertainty = max(uncertainty, 1e-3)
        return expected, uncertainty

    def _prepare_ml_dataframe(raw_row_df):
        df_enc = pd.get_dummies(
            raw_row_df,
            columns=[
                c for c in DSES_CATEGORICAL_COLS
                if c in raw_row_df.columns
            ],
            drop_first=True,
            dtype=float,
        )
        df_aligned = df_enc.reindex(
            columns=DSES_MODEL_COLUMNS,
            fill_value=0.0,
        )
        for col in DSES_MODEL_COLUMNS:
            if df_aligned[col].isna().any():
                df_aligned[col] = df_aligned[col].fillna(
                    DSES_MODEL_MEDIANS.get(col, 0.0)
                )
        return df_aligned

    def predict_dses_for_disease(disease, reaction_name):
        return predict_dses(
            build_ml_input_row_for_disease(
                disease, reaction_name
            )
        )

    def build_ml_input_row_for_disease(disease, reaction_name):
        row = build_ml_input_row(reaction_name).copy()
        row.loc[:, "Disease"] = disease
        return row

    for disease, reactions in DSES_MAPPING.items():
        components = []

        for reaction in reactions:
            if reaction not in _patient_components:
                continue

            factor = float(
                DISEASE_REACTION_FACTORS
                .get(disease, {})
                .get(reaction, 1.0)
            )

            components.append(
                _patient_components[reaction]["entropy_ratio"]
                * _patient_components[reaction]["stress_mean"]
                * max(factor, SOFT_FLOOR)
            )

        raw_dses = max(
            float(np.mean(components))
            if components
            else SOFT_FLOOR,
            SOFT_FLOOR,
        )

        patient_raw_dses[disease] = raw_dses
        patient_dses[disease] = scale_original_dses(raw_dses)

        # Patient-conditioned expected DSES: every disease gets its own
        # expected value using the current patient's complete profile.
        expected, uncertainty = predict_expected_dses_for_disease(disease)
        expected_dses[disease] = expected
        expected_dses_uncertainty[disease] = uncertainty

    # --------------------------------------------------------
    # ONE-DIMENSIONAL RANDOM-FOREST PROBABILITIES
    # --------------------------------------------------------
    # The RF still receives exactly ONE scalar: the original DSES.
    # Keeping it in its trained 1-100 domain avoids feeding the classifier
    # values outside the distribution it was trained on.
    raw_rf_probs = {}

    for disease, dses_value in patient_dses.items():
        model = _DSES_MODELS[disease]
        x_one = np.array([[dses_value]], dtype=float)
        proba_matrix = model.predict_proba(x_one)
        class_to_col = {int(cls): i for i, cls in enumerate(model.classes_)}
        raw_rf_probs[disease] = float(
            proba_matrix[0, class_to_col[1]]
            if 1 in class_to_col
            else SOFT_FLOOR
        )

    rf_total = sum(raw_rf_probs.values())
    if rf_total <= 0 or not np.isfinite(rf_total):
        rf_total = 1.0
    rf_probabilities = {
        disease: value / rf_total
        for disease, value in raw_rf_probs.items()
    }

    # --------------------------------------------------------
    # PATIENT-vs-EXPECTED DSES DISTANCE
    # --------------------------------------------------------
    # Distance is calculated in the same raw DSES scale, so clipping of the
    # RF input cannot hide how far the patient's raw DSES is from expectation.
    distances = {}
    standardized_distances = {}
    distance_similarity = {}

    for disease in DSES_MAPPING:
        distance = abs(
            patient_raw_dses[disease] - expected_dses[disease]
        )
        scale = max(expected_dses_uncertainty[disease], 1e-3)
        z = distance / scale

        distances[disease] = float(distance)
        standardized_distances[disease] = float(z)
        distance_similarity[disease] = float(
            np.exp(-0.5 * min(z * z, 700.0))
        )

    sim_total = sum(distance_similarity.values())
    if sim_total <= 0 or not np.isfinite(sim_total):
        sim_total = 1.0
    distance_probabilities = {
        disease: value / sim_total
        for disease, value in distance_similarity.items()
    }

    # --------------------------------------------------------
    # ROBUST TWO-SIGNAL ENSEMBLE
    # --------------------------------------------------------
    # Geometric mean prevents a disease from winning solely because one
    # signal is large while the other signal strongly disagrees.
    combined_raw = {}
    for disease in DSES_MAPPING:
        combined_raw[disease] = np.sqrt(
            max(rf_probabilities[disease], SOFT_FLOOR)
            * max(distance_probabilities[disease], SOFT_FLOOR)
        )

    combined_total = sum(combined_raw.values())
    if combined_total <= 0 or not np.isfinite(combined_total):
        combined_total = 1.0
    combined_probabilities = {
        disease: value / combined_total
        for disease, value in combined_raw.items()
    }

    disease_results = pd.DataFrame({
        "Disease": list(DSES_MAPPING.keys()),
        "Patient DSES": [patient_dses[d] for d in DSES_MAPPING],
        "Raw Patient DSES": [patient_raw_dses[d] for d in DSES_MAPPING],
        "Expected DSES": [expected_dses[d] for d in DSES_MAPPING],
        "DSES Distance": [distances[d] for d in DSES_MAPPING],
        "RF Uncertainty Scale": [expected_dses_uncertainty[d] for d in DSES_MAPPING],
        "Standardized Distance": [standardized_distances[d] for d in DSES_MAPPING],
        "RF Probability (%)": [rf_probabilities[d] * 100 for d in DSES_MAPPING],
        "Distance Probability (%)": [distance_probabilities[d] * 100 for d in DSES_MAPPING],
        "Probability": [combined_probabilities[d] for d in DSES_MAPPING],
    })

    disease_results["Probability (%)"] = disease_results["Probability"] * 100.0
    disease_results = disease_results.sort_values(
        "Probability", ascending=False
    ).reset_index(drop=True)
    disease_results.insert(
        0, "Rank", np.arange(1, len(disease_results) + 1)
    )

    # ========================================================
    # DISPLAY DT OUTPUTS
    # ========================================================

    st.header("Digital Twin Results")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Stroke Volume",
        f"{sv:.2f} mL",
    )

    m2.metric(
        "Ejection Fraction",
        f"{ef * 100:.2f}%",
    )

    m3.metric(
        "Entropy",
        f"{entropy:.4f} J/K",
    )

    m4.metric(
        "Entropy Stress",
        f"{entropy_stress:.3f}",
    )

    # ========================================================
    # DISEASE RANKING
    # ========================================================

    st.header("Disease Probability Ranking")

    st.dataframe(
        disease_results[
            [
                "Rank",
                "Disease",
                "Patient DSES",
                "Expected DSES",
                "DSES Distance",
                "Standardized Distance",
                "RF Probability (%)",
                "Distance Probability (%)",
                "Probability (%)",
            ]
        ].style.format(
            {
                "Patient DSES": "{:.3f}",
                "Expected DSES": "{:.3f}",
                "DSES Distance": "{:.3f}",
                "Standardized Distance": "{:.3f}",
                "RF Probability (%)": "{:.2f}%",
                "Distance Probability (%)": "{:.2f}%",
                "Probability (%)": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # DISEASE BAR CHART
    # ========================================================

    fig = go.Figure(
        go.Bar(
            x=disease_results["Probability (%)"],
            y=disease_results["Disease"],
            orientation="h",
            marker={
                "color": disease_results["Probability (%)"],
                "colorscale": "Viridis",
            },
        )
    )

    fig.update_layout(
        title="All Predicted Disease Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="Disease",
        yaxis={
            "categoryorder": "array",
            "categoryarray": disease_results[
                "Disease"
            ].tolist(),
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.info(
        "Final probability uses a geometric ensemble of the existing one-scalar RF probability "
        "and the patient-vs-expected DSES distance signal. The RF still receives exactly one DSES value. "
        "The uncertainty scale is an RF ensemble-spread proxy, not a clinical measurement error."
    )

    st.success(
        "Highest model-predicted disease probability: "
        f"{disease_results.iloc[0]['Disease']} "
        f"({disease_results.iloc[0]['Probability (%)']:.2f}%)"
    )

    # ========================================================
    # PATIENT CALCULATED VALUES
    # ========================================================

    with st.expander(
        "Show Digital Twin Calculations"
    ):
        st.dataframe(
            pd.DataFrame(
                {
                    "Variable": [
                        "Heart Rate",
                        "SBP",
                        "DBP",
                        "EDV",
                        "ESV",
                        "pH",
                        "SpO₂",
                        "Fasting Blood Sugar",
                        "Cholesterol",
                        "Stroke Volume",
                        "Ejection Fraction",
                        "Cardiac Output",
                        "MVO₂",
                        "Chemical Power",
                        "Mechanical Power",
                        "Heat Production",
                        "Entropy",
                        "Entropy Stress",
                        "Total ΔG",
                    ],
                    "Value": [
                        hr,
                        sbp,
                        dbp,
                        edv,
                        esv,
                        ph,
                        spo2,
                        fbs,
                        chol,
                        sv,
                        ef,
                        co,
                        mvo2,
                        chemical_power,
                        mechanical_power,
                        heat_production,
                        entropy,
                        entropy_stress,
                        total_dg,
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "ML-Predicted DSES "
            "(per mapped biochemical reaction)"
        )

        mapped_reactions = DSES_MAPPING.get(
            disease_choice,
            [],
        )

        ml_results = []

        for reaction_name in mapped_reactions:
            predicted_dses = predict_dses(
                build_ml_input_row(
                    reaction_name
                )
            )

            ml_results.append(
                {
                    "Biochemical Reaction": reaction_name,
                    "ML-Predicted DSES": predicted_dses,
                }
            )

        st.dataframe(
            pd.DataFrame(ml_results),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "Disease-Specific DSES Values"
        )

        dses_display = disease_results[
            [
                "Rank",
                "Disease",
                "Patient DSES",
                "Expected DSES",
                "DSES Distance",
                "Standardized Distance",
                "RF Probability (%)",
                "Distance Probability (%)",
                "Probability (%)",
            ]
        ].copy()

        st.dataframe(
            dses_display.style.format(
                {
                    "Patient DSES": "{:.3f}",
                    "Expected DSES": "{:.3f}",
                    "DSES Distance": "{:.3f}",
                    "Standardized Distance": "{:.3f}",
                    "RF Probability (%)": "{:.2f}%",
                    "Distance Probability (%)": "{:.2f}%",
                    "Probability (%)": "{:.2f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
