from pathlib import Path
import re
import numpy as np
import pandas as pd
import joblib
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
# GLOBAL CONSTANTS / SAFE NUMERIC HELPERS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
SOFT_FLOOR = 1e-6
DG_FLOOR = 1.0
R = 8.314


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


def load_with_diagnostic(filename):
    path = BASE_DIR / filename
    if not path.exists():
        st.error(f"❌ FILE NOT FOUND: {filename}")
        st.code(f"Expected location:\n{path}")
        st.stop()
    try:
        obj = joblib.load(path)
        st.success(f"✅ Loaded: {filename}")
        return obj
    except Exception as exc:
        st.error(f"❌ FAILED TO LOAD: {filename}")
        st.code(
            f"Path: {path}\n"
            f"Exception type: {type(exc).__name__}\n"
            f"Error: {exc}"
        )
        st.stop()


# ============================================================
# LOAD EXISTING ARTIFACTS
# ============================================================

_signature_ref = load_with_diagnostic(
    "disease_entropy_signature_reference.pkl"
)

refs = load_with_diagnostic("dt_reference_values.pkl")

DSES_RF_MODEL = load_with_diagnostic("dses_rf_model.joblib")
DSES_MODEL_COLUMNS = load_with_diagnostic("dses_model_columns.joblib")
DSES_MODEL_MEDIANS = load_with_diagnostic("dses_model_medians.joblib")
DSES_CATEGORICAL_COLS = load_with_diagnostic(
    "dses_categorical_columns.joblib"
)


# ============================================================
# GLOBAL DSES SETTINGS
# ============================================================

T = float(
    _signature_ref.get(
        "temperature_K",
        310.15,
    )
)

DSES_MAPPING_RAW = _signature_ref.get("mapping", {})
REACTION_REFS = _signature_ref.get("reaction_references", {})
DISEASE_REACTION_FACTORS = _signature_ref.get(
    "disease_reaction_factors",
    {},
)

KNOWN_REACTIONS = {
    "Metabolism",
    "ATP Utilization",
    "Ion Transport",
    "Calcium Handling",
    "Redox Metabolism",
    "Nitric Oxide Metabolism",
}


# ============================================================
# NORMALIZE DISEASE MAPPING
# ============================================================
# The reference file can store either:
#   Reaction -> [Diseases]
# or:
#   Disease -> [Reactions]
# The prediction app needs Disease -> Reactions because the
# repository contains one disease-specific RF model per disease.
# ============================================================

if set(DSES_MAPPING_RAW.keys()).intersection(KNOWN_REACTIONS):
    DSES_MAPPING = {}
    for reaction_name, disease_list in DSES_MAPPING_RAW.items():
        if reaction_name not in KNOWN_REACTIONS or disease_list is None:
            continue
        for disease_name in disease_list:
            DSES_MAPPING.setdefault(
                str(disease_name), []
            ).append(reaction_name)
else:
    DSES_MAPPING = {
        str(disease): list(reactions or [])
        for disease, reactions in DSES_MAPPING_RAW.items()
    }

DSES_MAPPING = {
    disease: sorted(set(reactions))
    for disease, reactions in DSES_MAPPING.items()
}

if not DSES_MAPPING:
    st.error("The disease entropy signature mapping is empty.")
    st.stop()


# ============================================================
# LOAD DISEASE-SPECIFIC RANDOM FOREST MODELS
# ============================================================

_DSES_MODEL_DIR = BASE_DIR / "disease_rf_models"
_DSES_MODELS = {}
_missing_disease_models = []

for disease in sorted(DSES_MAPPING):
    safe_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        disease,
    ).strip("_")

    model_path = _DSES_MODEL_DIR / f"{safe_name}.joblib"

    if not model_path.exists():
        _missing_disease_models.append(
            str(model_path)
        )
        continue

    try:
        _DSES_MODELS[disease] = joblib.load(model_path)
    except Exception as exc:
        st.error(
            f"Could not load disease-specific model for '{disease}': {exc}"
        )
        st.stop()


if _missing_disease_models:
    st.error(
        "One or more disease-specific DSES models are missing."
    )
    st.code("\n".join(_missing_disease_models))
    st.info(
        "Make sure the complete disease_rf_models/ folder is beside app.py."
    )
    st.stop()

st.success(
    f"✅ Loaded {len(_DSES_MODELS)} disease-specific Random Forest models"
)


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
# ADDITIONAL CLINICAL PARAMETERS
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

    myo_mass_input = st.number_input(
        "Myocardial Mass (g)",
        min_value=80.26,
        max_value=239.89,
        value=150.0,
        step=1.0,
    )


disease_choice = st.selectbox(
    "Disease for per-reaction DSES inspection",
    sorted(DSES_MAPPING.keys()),
)


# ============================================================
# RAW ML ROW FOR EXPECTED DSES MODEL
# ============================================================

def build_expected_dses_row(disease_name, reaction_name):
    """Build the same raw feature structure used to train dses_rf_model."""

    return pd.DataFrame([
        {
            "Age (mean)": age,
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
            "Number of Major Vessels (ca)": ca,
            "Thallium Stress Test Result (thal)": thal,
            "Serum Cholesterol (mg/dL, mean)": chol,
            "Fasting Blood Sugar (mg/dL, mean)": fbs,
            "Resting ECG Result": restecg,
            "Disease": disease_name,
            "Biochemical Reaction": reaction_name,
        }
    ])


def predict_expected_dses(disease_name, reaction_name):
    """
    Predict patient-conditioned expected DSES for one disease/reaction pair.

    Returns both:
        - mean expected DSES
        - Random-Forest ensemble spread as an uncertainty proxy

    The ensemble spread is only a fallback uncertainty estimate. When a
    validation-derived disease-specific error artifact is available, that
    artifact is preferred later in the standardized-distance calculation.
    """

    raw_row = build_expected_dses_row(
        disease_name,
        reaction_name,
    )

    categorical_present = [
        c for c in DSES_CATEGORICAL_COLS
        if c in raw_row.columns
    ]

    df_enc = pd.get_dummies(
        raw_row,
        columns=categorical_present,
        drop_first=True,
        dtype=float,
    )

    df_aligned = df_enc.reindex(
        columns=list(DSES_MODEL_COLUMNS),
        fill_value=0.0,
    )

    for col in df_aligned.columns:
        if df_aligned[col].isna().any():
            df_aligned[col] = df_aligned[col].fillna(
                DSES_MODEL_MEDIANS.get(
                    col,
                    0.0,
                )
            )

    # Main Random Forest regressor prediction.
    mean_prediction = float(
        DSES_RF_MODEL.predict(
            df_aligned
        )[0]
    )

    # Ensemble spread across individual trees. This is an uncertainty proxy,
    # not a clinical error bar. It is used only if a validation-derived
    # error-scale artifact is not available.
    tree_predictions = []

    if hasattr(DSES_RF_MODEL, "estimators_"):
        for tree in DSES_RF_MODEL.estimators_:
            try:
                tree_pred = float(
                    tree.predict(df_aligned)[0]
                )
                if np.isfinite(tree_pred):
                    tree_predictions.append(tree_pred)
            except Exception:
                continue

    if len(tree_predictions) >= 2:
        uncertainty = float(
            np.std(
                tree_predictions,
                ddof=1,
            )
        )
    else:
        uncertainty = 0.0

    return mean_prediction, uncertainty


# ============================================================
# RUN BUTTON
# ============================================================

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

    ef = sv / soft_denominator(edv)

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

    rpp_component = (
        rpp / soft_denominator(RPP_ref)
    )

    lvsw_component = (
        lvsw / soft_denominator(LVSW_ref)
    )

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

    safe_atp_balance = soft_positive_input(
        atp_balance
    )

    Q_metabolism = (
        ((fbs + 1.0) * chol)
        / soft_denominator(
            spo2 * safe_atp_balance
        )
    )

    Q_metabolism = soft_positive_input(
        Q_metabolism
    )

    dg_metabolism = (
        -2870000
        + R * T * np.log(Q_metabolism)
    )

    Q_atp = soft_positive_input(
        atp_utilization
        / soft_denominator(
            atp_production
        )
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

    no_factor = ph * spo2 / 100.0

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
    # Entropy from energy / Gibbs balance
    # --------------------------------------------------------

    chem_input_j = chemical_power * 60.0
    mech_work_j = mechanical_power * 60.0
    heat_j = heat_production * 60.0

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

    entropy = entropy_flow


    # --------------------------------------------------------
    # Stress features
    # --------------------------------------------------------

    entropy_stress = (
        entropy_flow
        / soft_denominator(
            refs["Entropy_Flow_median"]
        )
    )

    metabolic_stress = (
        mvo2
        / soft_denominator(
            refs["MVO2_median"]
        )
    )

    mechanical_stress = (
        mechanical_power
        / soft_denominator(
            refs["Mechanical_Power_median"]
        )
    )

    thermodynamic_stress = (
        abs(total_dg)
        / soft_denominator(
            refs["Total_DG_abs_median"]
        )
    )

    atp_stress = (
        atp_fraction
        / soft_denominator(
            refs["ATP_Utilization_Fraction_median"]
        )
    )


    # ========================================================
    # PATIENT REACTION-LEVEL DSES COMPONENTS
    # ========================================================

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

    patient_components = {}

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
            abs(dg_val) * dn_val / T,
            SOFT_FLOOR,
        )

        r_met = (
            mvo2
            / max(
                ref["mvo2_median"],
                SOFT_FLOOR,
            )
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

        patient_components[reaction] = {
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
            "metabolic_stress": r_met,
            "mechanical_stress": r_mech,
            "thermodynamic_stress": r_thermo,
            "atp_stress": r_atp,
            "entropy_stress": r_entropy,
        }


    # ========================================================
    # ACTUAL PATIENT DSES FOR EVERY DISEASE
    # ========================================================

    patient_dses = {}
    patient_dses_details = {}

    for disease, reactions in DSES_MAPPING.items():

        components = []
        used_reactions = []

        for reaction in reactions:

            if reaction not in patient_components:
                continue

            factor = float(
                DISEASE_REACTION_FACTORS
                .get(disease, {})
                .get(reaction, 1.0)
            )

            component_value = (
                patient_components[reaction][
                    "entropy_ratio"
                ]
                * patient_components[reaction][
                    "stress_mean"
                ]
                * max(
                    factor,
                    SOFT_FLOOR,
                )
            )

            components.append(
                component_value
            )
            used_reactions.append(reaction)

        raw_dses = max(
            float(np.mean(components))
            if components
            else SOFT_FLOOR,
            SOFT_FLOOR,
        )

        dses_min = float(
            _signature_ref.get(
                "raw_dses_min",
                raw_dses,
            )
        )

        dses_max = float(
            _signature_ref.get(
                "raw_dses_max",
                raw_dses,
            )
        )

        dses_low = float(
            _signature_ref.get(
                "dses_range_low",
                1.0,
            )
        )

        dses_high = float(
            _signature_ref.get(
                "dses_range_high",
                100.0,
            )
        )

        if dses_max - dses_min <= SOFT_FLOOR:
            range_dses = (
                dses_low + dses_high
            ) / 2.0
        else:
            range_dses = (
                dses_low
                + (
                    (raw_dses - dses_min)
                    * (dses_high - dses_low)
                    / (dses_max - dses_min)
                )
            )

        patient_dses[disease] = float(
            np.clip(
                range_dses,
                dses_low,
                dses_high,
            )
        )

        patient_dses_details[disease] = {
            "raw_dses": raw_dses,
            "mapped_reactions": reactions,
            "used_reactions": used_reactions,
            "missing_reactions": [
                r for r in reactions
                if r not in used_reactions
            ],
        }


    # ========================================================
    # PATIENT-CONDITIONED EXPECTED DSES
    # ========================================================

    expected_dses = {}
    expected_dses_details = {}
    expected_dses_uncertainty = {}

    progress = st.empty()

    for idx, disease in enumerate(
        sorted(DSES_MAPPING),
        start=1,
    ):

        reactions = DSES_MAPPING.get(
            disease,
            [],
        )

        predictions = []
        uncertainty_values = []
        reaction_rows = []

        for reaction in reactions:

            try:
                expected_value, ensemble_uncertainty = (
                    predict_expected_dses(
                        disease,
                        reaction,
                    )
                )

                if np.isfinite(expected_value):
                    predictions.append(
                        expected_value
                    )

                    if np.isfinite(ensemble_uncertainty):
                        uncertainty_values.append(
                            ensemble_uncertainty
                        )

                    reaction_rows.append(
                        {
                            "Biochemical Reaction": reaction,
                            "Patient-Conditioned Expected DSES": expected_value,
                            "Expected DSES Ensemble Uncertainty": (
                                ensemble_uncertainty
                            ),
                        }
                    )

            except Exception as exc:
                reaction_rows.append(
                    {
                        "Biochemical Reaction": reaction,
                        "Patient-Conditioned Expected DSES": np.nan,
                        "Expected DSES Ensemble Uncertainty": np.nan,
                        "Prediction Error": str(exc),
                    }
                )

        if predictions:
            expected_dses[disease] = float(
                np.mean(predictions)
            )
        else:
            expected_dses[disease] = np.nan

        if uncertainty_values:
            expected_dses_uncertainty[disease] = float(
                np.mean(uncertainty_values)
            )
        else:
            expected_dses_uncertainty[disease] = np.nan

        expected_dses_details[disease] = reaction_rows

    progress.empty()


    # ========================================================
    # VALIDATION-DERIVED ERROR SCALE (PREFERRED)
    # ========================================================
    # Optional artifact supported by the training pipeline:
    #   expected_dses_error_by_disease.joblib
    # If it is absent, the app falls back to the RF tree-ensemble spread.
    # ========================================================

    validation_error_scales = _signature_ref.get(
        "expected_dses_error_by_disease",
        {}
    )

    if not isinstance(validation_error_scales, dict):
        validation_error_scales = {}

    fallback_error_values = [
        float(v)
        for v in expected_dses_uncertainty.values()
        if np.isfinite(v) and v > SOFT_FLOOR
    ]

    global_error_floor = (
        float(np.median(fallback_error_values))
        if fallback_error_values
        else 1.0
    )

    # Never allow zero/negative/invalid error scales.
    global_error_floor = max(
        global_error_floor,
        SOFT_FLOOR,
    )


    # ========================================================
    # DSES STANDARDIZED DISTANCE PROBABILITY
    # ========================================================

    valid_expected = {
        d: v
        for d, v in expected_dses.items()
        if np.isfinite(v)
        and d in patient_dses
    }

    if not valid_expected:
        st.error(
            "No patient-conditioned expected DSES values could be calculated."
        )
        st.stop()

    distances = {
        disease: abs(
            patient_dses[disease]
            - expected_value
        )
        for disease, expected_value in valid_expected.items()
    }

    error_scales = {}
    standardized_distances = {}

    for disease, distance in distances.items():

        configured_error = validation_error_scales.get(
            disease,
            np.nan,
        )

        try:
            configured_error = float(
                configured_error
            )
        except (TypeError, ValueError):
            configured_error = np.nan

        ensemble_error = expected_dses_uncertainty.get(
            disease,
            np.nan,
        )

        if not np.isfinite(ensemble_error):
            ensemble_error = np.nan

        if (
            np.isfinite(configured_error)
            and configured_error > SOFT_FLOOR
        ):
            error_scale = configured_error
            error_source = "Validation residual SD"

        elif (
            np.isfinite(ensemble_error)
            and ensemble_error > SOFT_FLOOR
        ):
            error_scale = ensemble_error
            error_source = "RF ensemble spread"

        else:
            error_scale = global_error_floor
            error_source = "Global fallback error scale"

        error_scales[disease] = {
            "value": float(
                max(error_scale, SOFT_FLOOR)
            ),
            "source": error_source,
        }

        standardized_distances[disease] = (
            distance
            / error_scales[disease]["value"]
        )

    # Gaussian-style similarity:
    # z = distance / error scale
    # similarity = exp(-0.5 * z^2)
    # This strongly rewards a close match relative to expected model error.
    distance_similarity_raw = {
        disease: float(
            np.exp(
                -0.5
                * standardized_distances[disease] ** 2
            )
        )
        for disease in distances
    }

    similarity_total = sum(
        distance_similarity_raw.values()
    )

    if (
        similarity_total <= 0
        or not np.isfinite(similarity_total)
    ):
        similarity_total = float(
            len(distance_similarity_raw)
        )

    distance_probability = {
        disease: value / similarity_total
        for disease, value in (
            distance_similarity_raw.items()
        )
    }


    # ========================================================
    # DSES DISTANCE PROBABILITY
    # ========================================================

    distance_rows = []

    valid_expected = {
        d: v
        for d, v in expected_dses.items()
        if np.isfinite(v)
        and d in patient_dses
    }

    if not valid_expected:
        st.error(
            "No patient-conditioned expected DSES values could be calculated."
        )
        st.stop()

    distances = {
        disease: abs(
            patient_dses[disease]
            - expected_value
        )
        for disease, expected_value in valid_expected.items()
    }

    distance_values = np.array(
        list(distances.values()),
        dtype=float,
    )

    # A patient-independent numerical scale based on the spread of
    # expected disease signatures. This prevents the exponential
    # similarity from becoming artificially sharp for small spreads.
    tau = max(
        float(np.std(distance_values)),
        1.0,
    )

    distance_similarity_raw = {
        disease: float(
            np.exp(
                -distance / tau
            )
        )
        for disease, distance in distances.items()
    }

    similarity_total = sum(
        distance_similarity_raw.values()
    )

    distance_probability = {
        disease: value / similarity_total
        for disease, value in (
            distance_similarity_raw.items()
        )
    }


    # ========================================================
    # EXISTING RANDOM FOREST PROBABILITY
    # ========================================================

    raw_rf_probability = {}

    for disease, dses_value in patient_dses.items():

        if disease not in _DSES_MODELS:
            continue

        model = _DSES_MODELS[disease]

        x_one = np.array(
            [[dses_value]],
            dtype=float,
        )

        proba_matrix = model.predict_proba(
            x_one
        )

        class_to_col = {
            int(cls): i
            for i, cls in enumerate(
                model.classes_
            )
        }

        raw_rf_probability[disease] = float(
            proba_matrix[
                0,
                class_to_col[1],
            ]
            if 1 in class_to_col
            else 0.0
        )

    rf_total = sum(
        raw_rf_probability.values()
    )

    if (
        rf_total <= 0
        or not np.isfinite(rf_total)
    ):
        rf_total = 1.0

    rf_probability = {
        disease: value / rf_total
        for disease, value in (
            raw_rf_probability.items()
        )
    }


    # ========================================================
    # COMBINED MODEL SCORE
    # ========================================================
    # Distance probability is the primary explainable signal.
    # RF probability is retained as a second model signal.
    # Default blend: 70% DSES-distance + 30% RF.
    # This is a model-estimated ranking probability, not a clinically
    # calibrated probability unless separately calibrated on held-out data.
    # ========================================================

    DISTANCE_WEIGHT = 0.70
    RF_WEIGHT = 0.30

    combined_rows = []

    common_diseases = sorted(
        set(distance_probability.keys())
        & set(rf_probability.keys())
    )

    for disease in common_diseases:

        combined_probability = (
            DISTANCE_WEIGHT
            * distance_probability[disease]
            + RF_WEIGHT
            * rf_probability[disease]
        )

        combined_rows.append(
            {
                "Disease": disease,
                "Patient DSES": patient_dses[disease],
                "Expected DSES": expected_dses[disease],
                "DSES Distance": distances[disease],
                "DSES Error Scale": error_scales[disease]["value"],
                "Standardized DSES Distance": standardized_distances[disease],
                "DSES Distance Probability": (
                    distance_probability[disease] * 100.0
                ),
                "RF Probability": (
                    rf_probability[disease] * 100.0
                ),
                "Combined Model Probability": (
                    combined_probability * 100.0
                ),
            }
        )

    disease_results = pd.DataFrame(
        combined_rows
    ).sort_values(
        "Combined Model Probability",
        ascending=False,
    ).reset_index(drop=True)

    disease_results.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(disease_results) + 1,
        ),
    )

    top_disease = disease_results.iloc[0][
        "Disease"
    ]

    top_probability = disease_results.iloc[0][
        "Combined Model Probability"
    ]

    top_patient_dses = disease_results.iloc[0][
        "Patient DSES"
    ]

    top_expected_dses = disease_results.iloc[0][
        "Expected DSES"
    ]

    top_distance = disease_results.iloc[0][
        "DSES Distance"
    ]

    top_error_scale = disease_results.iloc[0][
        "DSES Error Scale"
    ]

    top_standardized_distance = disease_results.iloc[0][
        "Standardized DSES Distance"
    ]


    # ========================================================
    # DIGITAL TWIN RESULTS
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
    # CLEAN DSES-DISTANCE DISEASE RANKING
    # ========================================================

    st.header(
        "Disease Probability Ranking"
    )

    st.caption(
        "Primary ranking uses an uncertainty-adjusted DSES distance. "
        "The patient's DSES is compared with a patient-conditioned expected "
        "DSES, then divided by an error/uncertainty scale. Smaller standardized "
        "distance means a closer disease match. The RF value is shown as a "
        "second model signal."
    )

    display_cols = [
        "Rank",
        "Disease",
        "Patient DSES",
        "Expected DSES",
        "DSES Distance",
        "DSES Error Scale",
        "Standardized DSES Distance",
        "DSES Distance Probability",
        "RF Probability",
        "Combined Model Probability",
    ]

    st.dataframe(
        disease_results[display_cols].style.format(
            {
                "Patient DSES": "{:.3f}",
                "Expected DSES": "{:.3f}",
                "DSES Distance": "{:.3f}",
                "DSES Error Scale": "{:.3f}",
                "Standardized DSES Distance": "{:.3f}",
                "DSES Distance Probability": "{:.2f}%",
                "RF Probability": "{:.2f}%",
                "Combined Model Probability": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # FINAL PREDICTION
    # ========================================================

    st.success(
        f"Top model-estimated disease: {top_disease} "
        f"({top_probability:.2f}%)"
    )

    st.info(
        f"Patient DSES = {top_patient_dses:.3f} | "
        f"Expected DSES = {top_expected_dses:.3f} | "
        f"Distance = {top_distance:.3f} | "
        f"Error scale = {top_error_scale:.3f} | "
        f"Standardized distance = {top_standardized_distance:.3f}"
    )

    st.caption(
        "The displayed percentages are model-estimated ranking scores. "
        "The DSES-distance component uses standardized distance = absolute "
        "DSES difference divided by an error/uncertainty scale, then applies "
        "a Gaussian-style similarity. This is not a clinically calibrated "
        "probability unless calibrated and validated on independent clinical data."
    )

    st.caption(
        "Error-scale source: validation-derived residual SD when available; "
        "otherwise the Random Forest ensemble spread is used as an uncertainty "
        "proxy, with a global fallback if necessary."
    )


    # ========================================================
    # PROBABILITY COMPARISON CHART
    # ========================================================

    fig_prob = go.Figure()

    fig_prob.add_bar(
        x=disease_results["Disease"],
        y=disease_results[
            "DSES Distance Probability"
        ],
        name="DSES Distance",
    )

    fig_prob.add_bar(
        x=disease_results["Disease"],
        y=disease_results[
            "RF Probability"
        ],
        name="Random Forest",
    )

    fig_prob.add_bar(
        x=disease_results["Disease"],
        y=disease_results[
            "Combined Model Probability"
        ],
        name="Combined",
    )

    fig_prob.update_layout(
        title="Distance vs Random Forest vs Combined Disease Probability",
        xaxis_title="Disease",
        yaxis_title="Model-estimated probability (%)",
        barmode="group",
        xaxis={"tickangle": -60},
    )

    st.plotly_chart(
        fig_prob,
        use_container_width=True,
    )


    # ========================================================
    # DSES DISTANCE GRAPH
    # ========================================================

    fig_distance = go.Figure(
        go.Bar(
            x=disease_results["Disease"],
            y=disease_results["DSES Distance"],
        )
    )

    fig_distance.update_layout(
        title="Distance Between Patient DSES and Patient-Conditioned Expected DSES",
        xaxis_title="Disease",
        yaxis_title="Absolute DSES Distance",
        xaxis={"tickangle": -60},
    )

    st.plotly_chart(
        fig_distance,
        use_container_width=True,
    )


    # ========================================================
    # STANDARDIZED DISTANCE GRAPH
    # ========================================================

    fig_std_distance = go.Figure(
        go.Bar(
            x=disease_results["Disease"],
            y=disease_results["Standardized DSES Distance"],
        )
    )

    fig_std_distance.update_layout(
        title="Standardized DSES Distance (Distance ÷ Error/Uncertainty Scale)",
        xaxis_title="Disease",
        yaxis_title="Standardized DSES Distance",
        xaxis={"tickangle": -60},
    )

    fig_std_distance.add_hline(
        y=1.0,
        line_dash="dash",
        annotation_text="1 × error scale",
    )

    st.plotly_chart(
        fig_std_distance,
        use_container_width=True,
    )


    # ========================================================
    # WHY WAS THIS DISEASE PREDICTED?
    # ========================================================

    st.header("Why Was This Disease Predicted?")

    top_info = disease_results.iloc[0]

    st.write(
        f"The top-ranked disease is **{top_disease}**. "
        f"Its patient-conditioned expected DSES is "
        f"**{top_expected_dses:.3f}**, while the patient's calculated "
        f"DSES is **{top_patient_dses:.3f}**. The absolute distance is "
        f"**{top_distance:.3f}**. The distance is then scaled by an "
        f"error/uncertainty estimate of **{top_error_scale:.3f}**, giving a "
        f"standardized distance of **{top_standardized_distance:.3f}**. "
        f"Smaller standardized distance means a closer match relative to "
        f"the expected model uncertainty."
    )


    # ========================================================
    # STRESS PROFILE
    # ========================================================

    st.subheader("Patient Stress Profile")

    stress_df = pd.DataFrame(
        {
            "Stress Component": [
                "Metabolic Stress",
                "Mechanical Stress",
                "Thermodynamic Stress",
                "ATP Stress",
                "Entropy Stress",
            ],
            "Value": [
                metabolic_stress,
                mechanical_stress,
                thermodynamic_stress,
                atp_stress,
                entropy_stress,
            ],
        }
    )

    st.dataframe(
        stress_df.style.format(
            {"Value": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig_stress = go.Figure()

    fig_stress.add_bar(
        x=stress_df["Stress Component"],
        y=stress_df["Value"],
    )

    fig_stress.add_hline(
        y=1.0,
        line_dash="dash",
        annotation_text="Reference = 1.0",
    )

    fig_stress.update_layout(
        title=f"Patient Stress Profile Supporting {top_disease}",
        xaxis_title="Stress Component",
        yaxis_title="Normalized Stress",
    )

    st.plotly_chart(
        fig_stress,
        use_container_width=True,
    )


    # ========================================================
    # REACTION-LEVEL EXPLANATION FOR TOP DISEASE
    # ========================================================

    st.subheader(
        f"{top_disease}: Reaction-Level DSES Explanation"
    )

    top_reactions = DSES_MAPPING.get(
        top_disease,
        [],
    )

    explanation_rows = []

    for reaction in top_reactions:

        if reaction not in patient_components:
            continue

        component = patient_components[reaction]

        factor = float(
            DISEASE_REACTION_FACTORS
            .get(top_disease, {})
            .get(reaction, 1.0)
        )

        component_dses = (
            component["entropy_ratio"]
            * component["stress_mean"]
            * max(
                factor,
                SOFT_FLOOR,
            )
        )

        expected_rows = expected_dses_details.get(
            top_disease,
            [],
        )

        expected_value = np.nan

        for erow in expected_rows:
            if erow.get(
                "Biochemical Reaction"
            ) == reaction:
                expected_value = erow.get(
                    "Patient-Conditioned Expected DSES",
                    np.nan,
                )
                break

        explanation_rows.append(
            {
                "Biochemical Reaction": reaction,
                "Metabolic Stress": component[
                    "metabolic_stress"
                ],
                "Mechanical Stress": component[
                    "mechanical_stress"
                ],
                "Thermodynamic Stress": component[
                    "thermodynamic_stress"
                ],
                "ATP Stress": component[
                    "atp_stress"
                ],
                "Entropy Stress": component[
                    "entropy_stress"
                ],
                "Entropy Ratio": component[
                    "entropy_ratio"
                ],
                "Literature Factor": factor,
                "Patient DSES Component": component_dses,
                "Expected DSES Component": expected_value,
            }
        )

    explanation_df = pd.DataFrame(
        explanation_rows
    )

    if not explanation_df.empty:

        st.dataframe(
            explanation_df.style.format(
                {
                    "Metabolic Stress": "{:.3f}",
                    "Mechanical Stress": "{:.3f}",
                    "Thermodynamic Stress": "{:.3f}",
                    "ATP Stress": "{:.3f}",
                    "Entropy Stress": "{:.3f}",
                    "Entropy Ratio": "{:.3f}",
                    "Literature Factor": "{:.3f}",
                    "Patient DSES Component": "{:.3f}",
                    "Expected DSES Component": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        fig_reaction = go.Figure()

        fig_reaction.add_bar(
            x=explanation_df[
                "Biochemical Reaction"
            ],
            y=explanation_df[
                "Patient DSES Component"
            ],
            name="Patient DSES Component",
        )

        if explanation_df[
            "Expected DSES Component"
        ].notna().any():

            fig_reaction.add_bar(
                x=explanation_df[
                    "Biochemical Reaction"
                ],
                y=explanation_df[
                    "Expected DSES Component"
                ],
                name="Expected DSES Component",
            )

        fig_reaction.update_layout(
            title=f"Patient vs Expected DSES Components — {top_disease}",
            xaxis_title="Biochemical Reaction",
            yaxis_title="DSES Component",
            barmode="group",
        )

        st.plotly_chart(
            fig_reaction,
            use_container_width=True,
        )


    # ========================================================
    # ML EXPECTED DSES INSPECTION
    # ========================================================

    with st.expander(
        "Show Patient-Conditioned Expected DSES Predictions"
    ):

        selected_reactions = DSES_MAPPING.get(
            disease_choice,
            [],
        )

        ml_results = []

        for reaction_name in selected_reactions:

            try:
                predicted_dses = (
                    predict_expected_dses(
                        disease_choice,
                        reaction_name,
                    )
                )

                ml_results.append(
                    {
                        "Biochemical Reaction": reaction_name,
                        "Patient-Conditioned Expected DSES": predicted_dses,
                    }
                )

            except Exception as exc:

                ml_results.append(
                    {
                        "Biochemical Reaction": reaction_name,
                        "Patient-Conditioned Expected DSES": np.nan,
                        "Error": str(exc),
                    }
                )

        st.dataframe(
            pd.DataFrame(ml_results),
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # DIGITAL TWIN CALCULATIONS
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


    # ========================================================
    # FULL DISEASE DSES TABLE
    # ========================================================

    st.subheader(
        "Disease DSES / Distance Details"
    )

    st.dataframe(
        disease_results[
            [
                "Rank",
                "Disease",
                "Patient DSES",
                "Expected DSES",
                "DSES Distance",
                "DSES Distance Probability",
                "RF Probability",
                "Combined Model Probability",
            ]
        ].style.format(
            {
                "Patient DSES": "{:.3f}",
                "Expected DSES": "{:.3f}",
                "DSES Distance": "{:.3f}",
                "DSES Error Scale": "{:.3f}",
                "Standardized DSES Distance": "{:.3f}",
                "DSES Distance Probability": "{:.2f}%",
                "RF Probability": "{:.2f}%",
                "Combined Model Probability": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
