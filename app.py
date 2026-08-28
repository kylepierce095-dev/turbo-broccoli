from pathlib import Path
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
DEFAULT_TEMPERATURE_K = 310.15
GAS_CONSTANT_R = 8.314

# Healthy-state center on the same scale as disease_signature_scalars.
# Disease scalars in the artifact range ~1–100; healthy is deliberately low.
HEALTHY_LABEL = "Healthy"
HEALTHY_SIGNATURE_SCALAR = 5.0


def soft_scalar(value, floor=SOFT_FLOOR):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(floor)
    if not np.isfinite(x) or x == 0:
        return float(floor)
    return x


def soft_denominator(value, floor=SOFT_FLOOR):
    if isinstance(value, pd.Series):
        s = pd.to_numeric(value, errors="coerce")
        return s.where(
            s.notna() & np.isfinite(s) & (s != 0),
            floor,
        )
    return soft_scalar(value, floor)


def soft_positive_input(value, floor=SOFT_FLOOR):
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


def safe_float(value, default=0.0):
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def load_artifact(filename, required=True):
    path = BASE_DIR / filename
    if not path.exists():
        if required:
            st.error(f"Required file not found: {filename}")
            st.code(str(path))
            st.stop()
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        st.error(f"Could not load {filename}: {exc}")
        st.stop()


# ============================================================
# LOAD EXISTING DSES ARTIFACTS
# ============================================================

signature = load_artifact("disease_entropy_signature_reference.pkl")
refs = load_artifact("dt_reference_values.pkl")
DSES_RF_MODEL = load_artifact("dses_rf_model.joblib")
DSES_MODEL_COLUMNS = load_artifact("dses_model_columns.joblib")
DSES_MODEL_MEDIANS = load_artifact("dses_model_medians.joblib")
DSES_CATEGORICAL_COLS = load_artifact("dses_categorical_columns.joblib")

if not isinstance(DSES_MODEL_COLUMNS, (list, tuple, np.ndarray, pd.Index)):
    DSES_MODEL_COLUMNS = list(DSES_MODEL_COLUMNS)

if not isinstance(DSES_MODEL_MEDIANS, dict):
    try:
        DSES_MODEL_MEDIANS = dict(DSES_MODEL_MEDIANS)
    except Exception:
        DSES_MODEL_MEDIANS = {}

DSES_MODEL_COLUMNS = list(DSES_MODEL_COLUMNS)
DSES_CATEGORICAL_COLS = list(DSES_CATEGORICAL_COLS or [])

T = safe_float(
    signature.get("temperature_K", DEFAULT_TEMPERATURE_K),
    DEFAULT_TEMPERATURE_K,
)
if T <= 0:
    T = DEFAULT_TEMPERATURE_K

DSES_MAPPING_RAW = signature.get("mapping", {})
REACTION_REFS = signature.get("reaction_references", {})
DISEASE_REACTION_FACTORS = signature.get(
    "disease_reaction_factors",
    {},
)
DISEASE_SIGNATURE_SCALARS = dict(
    signature.get("disease_signature_scalars", {}) or {}
)
# Inject Healthy so Expected DSES has a unique center for the healthy state.
DISEASE_SIGNATURE_SCALARS[HEALTHY_LABEL] = HEALTHY_SIGNATURE_SCALAR

KNOWN_REACTIONS = {
    "Metabolism",
    "ATP Utilization",
    "Ion Transport",
    "Calcium Handling",
    "Redox Metabolism",
    "Nitric Oxide Metabolism",
}

# ============================================================
# NORMALIZE DISEASE -> REACTION MAPPING
# ============================================================

if set(DSES_MAPPING_RAW.keys()).intersection(KNOWN_REACTIONS):
    DSES_MAPPING = {}
    for reaction_name, disease_list in DSES_MAPPING_RAW.items():
        if reaction_name not in KNOWN_REACTIONS:
            continue
        if disease_list is None:
            continue
        for disease_name in disease_list:
            DSES_MAPPING.setdefault(
                str(disease_name),
                [],
            ).append(str(reaction_name))
else:
    DSES_MAPPING = {
        str(disease): list(reactions or [])
        for disease, reactions in DSES_MAPPING_RAW.items()
    }

DSES_MAPPING = {
    disease: sorted(set(reactions))
    for disease, reactions in DSES_MAPPING.items()
}

# Healthy state: all known reactions at baseline (factor 1.0).
# This lets the ranking include P(Healthy) alongside disease hypotheses.
DSES_MAPPING[HEALTHY_LABEL] = sorted(KNOWN_REACTIONS)
DISEASE_REACTION_FACTORS.setdefault(HEALTHY_LABEL, {})
for rxn in KNOWN_REACTIONS:
    DISEASE_REACTION_FACTORS[HEALTHY_LABEL].setdefault(rxn, 1.0)

if not DSES_MAPPING:
    st.error("The disease entropy signature mapping is empty.")
    st.stop()

st.success(
    f"Loaded DSES model and {len(DSES_MAPPING)} states "
    f"(including {HEALTHY_LABEL}). "
    "No disease-specific RF classifier files are required."
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
    # Standard thallium stress-test categories (Cleveland / clinical).
    # Training data only contained "Normal", so the RF has no thal dummies;
    # the UI must still expose the full clinical set.
    thal = st.selectbox(
        "Thallium Stress Test Result (thal)",
        [
            "Normal",
            "Fixed defect",
            "Reversible defect",
        ],
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
# ML INPUT
# ============================================================

def build_ml_row(disease_name, reaction_name):
    return pd.DataFrame(
        [
            {
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
                "Disease": disease_name,
                "Biochemical Reaction": reaction_name,
            }
        ]
    )


def prepare_model_input(raw_df):
    """
    Encode a single patient row for the DSES RF regressor.

    CRITICAL: do NOT use drop_first=True on a single-row frame.
    With only one level present, drop_first removes that level
    entirely → every disease collapses to the same RF input.
    """
    categorical = [
        c for c in DSES_CATEGORICAL_COLS
        if c in raw_df.columns
    ]

    encoded = pd.get_dummies(
        raw_df,
        columns=categorical,
        drop_first=False,
        dtype=float,
    )

    aligned = encoded.reindex(
        columns=DSES_MODEL_COLUMNS,
        fill_value=0.0,
    )

    for col in aligned.columns:
        if aligned[col].isna().any():
            aligned[col] = aligned[col].fillna(
                safe_float(
                    DSES_MODEL_MEDIANS.get(col, 0.0),
                    0.0,
                )
            )

    return aligned


def rf_expected_dses_and_uncertainty(disease_name, reaction_name):
    """
    Single RF DSES regressor: predicts expected DSES for this
    patient's clinical parameters under a disease/reaction pair.
    Tree spread is only an uncertainty proxy.
    """
    # Healthy is not a trained Disease level — use a neutral disease
    # label so clinical features still condition the RF.
    rf_disease = (
        "Heart block"
        if disease_name == HEALTHY_LABEL
        else disease_name
    )

    raw = build_ml_row(rf_disease, reaction_name)
    x = prepare_model_input(raw)

    mean_prediction = float(
        DSES_RF_MODEL.predict(x)[0]
    )

    tree_predictions = []

    if hasattr(DSES_RF_MODEL, "estimators_"):
        for tree in DSES_RF_MODEL.estimators_:
            try:
                value = float(tree.predict(x)[0])
                if np.isfinite(value):
                    tree_predictions.append(value)
            except Exception:
                continue

    if len(tree_predictions) >= 2:
        uncertainty = float(
            np.std(tree_predictions, ddof=1)
        )
    else:
        uncertainty = np.nan

    return mean_prediction, uncertainty


# ============================================================
# RUN
# ============================================================

run_dt = st.button(
    "RUN DIGITAL TWIN",
    type="primary",
    use_container_width=True,
)

if run_dt:

    if edv <= esv:
        st.error("EDV must be greater than ESV.")
        st.stop()

    if sbp <= dbp:
        st.error("SBP must be greater than DBP.")
        st.stop()

    # ========================================================
    # HEMODYNAMICS
    # ========================================================

    sv = edv - esv
    ef = sv / soft_denominator(edv)

    map_pressure = dbp + (sbp - dbp) / 3.0
    co = sv * hr
    rpp = hr * sbp
    lvsw = sv * map_pressure * 0.000133322

    # ========================================================
    # METABOLIC LAYER
    # ========================================================

    RPP_ref = 10000.0
    LVSW_ref = 0.9
    MVO2_rest = 8.0
    O2_ENERGY = 20.2
    MYOCARDIAL_MASS = myo_mass_input

    rpp_component = rpp / soft_denominator(RPP_ref)
    lvsw_component = lvsw / soft_denominator(LVSW_ref)
    metabolic_demand = 0.7 * rpp_component + 0.3 * lvsw_component
    mvo2 = MVO2_rest * metabolic_demand
    o2_consumption = mvo2 * MYOCARDIAL_MASS / 100.0
    chemical_power = o2_consumption * O2_ENERGY / 60.0
    mechanical_power = lvsw * hr / 60.0
    heat_production = max(chemical_power - mechanical_power, 0.0)

    # ========================================================
    # ATP
    # ========================================================

    ATP_ENERGY = 30500.0
    ATP_COUPLING_EFFICIENCY = 0.60

    atp_production = (
        chemical_power * 60.0 * ATP_COUPLING_EFFICIENCY
        / soft_denominator(ATP_ENERGY)
    )
    mechanical_energy = mechanical_power * 60.0
    atp_utilization = mechanical_energy / soft_denominator(ATP_ENERGY)
    atp_fraction = atp_utilization / soft_denominator(atp_production)
    atp_balance = atp_production - atp_utilization

    # ========================================================
    # THERMODYNAMIC PROXY LAYER
    # ========================================================

    safe_atp_balance = soft_positive_input(atp_balance)

    Q_metabolism = soft_positive_input(
        ((fbs + 1.0) * chol)
        / soft_denominator(spo2 * safe_atp_balance)
    )
    dg_metabolism = (
        -2870000.0 + GAS_CONSTANT_R * T * np.log(Q_metabolism)
    )

    Q_atp = soft_positive_input(
        atp_utilization / soft_denominator(atp_production)
    )
    dg_atp = -30500.0 + GAS_CONSTANT_R * T * np.log(Q_atp)

    pH_factor = 10 ** (7.4 - ph)
    Q_ion = soft_positive_input(atp_fraction * pH_factor)
    dg_ion = -50000.0 + GAS_CONSTANT_R * T * np.log(Q_ion)

    calcium_factor = ph * hr / soft_denominator(hr)
    Q_calcium = soft_positive_input(atp_fraction * calcium_factor)
    dg_calcium = -50000.0 + GAS_CONSTANT_R * T * np.log(Q_calcium)

    redox_factor = spo2 / 100.0
    Q_redox = soft_positive_input(atp_fraction * redox_factor)
    dg_redox = -220000.0 + GAS_CONSTANT_R * T * np.log(Q_redox)

    no_factor = ph * spo2 / 100.0
    Q_no = soft_positive_input(atp_fraction * no_factor)
    dg_no = -100000.0 + GAS_CONSTANT_R * T * np.log(Q_no)

    total_dg = (
        dg_metabolism + dg_atp + dg_ion
        + dg_calcium + dg_redox + dg_no
    )

    # ========================================================
    # ENTROPY FROM ENERGY / GIBBS BALANCE
    # ========================================================

    chem_input_j = chemical_power * 60.0
    mech_work_j = mechanical_power * 60.0
    heat_j = heat_production * 60.0

    ion_energy = mech_work_j * atp_fraction
    calcium_energy = mech_work_j * atp_fraction
    redox_energy = chem_input_j * atp_fraction
    no_energy = chem_input_j * atp_fraction

    dG_dn_sum = 0.0

    dn_metab = chem_input_j / max(abs(dg_metabolism), DG_FLOOR)
    dG_dn_sum += dg_metabolism * dn_metab

    dn_atp = atp_utilization
    dG_dn_sum += dg_atp * dn_atp

    dn_ion = ion_energy / max(abs(dg_ion), DG_FLOOR)
    dG_dn_sum += dg_ion * dn_ion

    dn_calcium = calcium_energy / max(abs(dg_calcium), DG_FLOOR)
    dG_dn_sum += dg_calcium * dn_calcium

    dn_redox = redox_energy / max(abs(dg_redox), DG_FLOOR)
    dG_dn_sum += dg_redox * dn_redox

    dn_no = no_energy / max(abs(dg_no), DG_FLOOR)
    dG_dn_sum += dg_no * dn_no

    entropy_flow = (chem_input_j + mech_work_j - dG_dn_sum) / T
    entropy = entropy_flow

    # ========================================================
    # NORMALIZED STRESS FEATURES
    # ========================================================

    entropy_stress = (
        entropy_flow / soft_denominator(refs["Entropy_Flow_median"])
    )
    metabolic_stress = (
        mvo2 / soft_denominator(refs["MVO2_median"])
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
        / soft_denominator(refs["ATP_Utilization_Fraction_median"])
    )

    # ========================================================
    # PATIENT REACTION-LEVEL DSES COMPONENTS
    # ========================================================

    reaction_values = {
        "Metabolism": {"dg": dg_metabolism, "energy": chem_input_j},
        "ATP Utilization": {"dg": dg_atp, "energy": mechanical_energy},
        "Ion Transport": {"dg": dg_ion, "energy": ion_energy},
        "Calcium Handling": {"dg": dg_calcium, "energy": calcium_energy},
        "Redox Metabolism": {"dg": dg_redox, "energy": redox_energy},
        "Nitric Oxide Metabolism": {"dg": dg_no, "energy": no_energy},
    }

    patient_components = {}

    for reaction, vals in reaction_values.items():
        if reaction not in REACTION_REFS:
            continue

        ref = REACTION_REFS[reaction]
        dg_val = float(vals["dg"])
        energy_val = float(vals["energy"])
        dn_val = energy_val / max(abs(dg_val), DG_FLOOR)

        reaction_entropy_flow = max(
            abs(dg_val) * dn_val / T,
            SOFT_FLOOR,
        )

        r_met = mvo2 / max(
            safe_float(ref.get("mvo2_median", 1.0), 1.0),
            SOFT_FLOOR,
        )
        r_mech = mechanical_power / max(
            safe_float(ref.get("mechanical_power_median", 1.0), 1.0),
            SOFT_FLOOR,
        )
        r_thermo = abs(dg_val) / max(
            safe_float(ref.get("dg_abs_median", 1.0), 1.0),
            SOFT_FLOOR,
        )
        r_atp = atp_fraction / max(
            safe_float(ref.get("atp_fraction_median", 1.0), 1.0),
            SOFT_FLOOR,
        )
        r_entropy = reaction_entropy_flow / max(
            safe_float(ref.get("entropy_flow_median", 1.0), 1.0),
            SOFT_FLOOR,
        )

        stress_values = [r_met, r_mech, r_thermo, r_atp, r_entropy]

        patient_components[reaction] = {
            "entropy_ratio": max(r_entropy, SOFT_FLOOR),
            "stress_mean": max(float(np.mean(stress_values)), SOFT_FLOOR),
            "metabolic_stress": r_met,
            "mechanical_stress": r_mech,
            "thermodynamic_stress": r_thermo,
            "atp_stress": r_atp,
            "entropy_stress": r_entropy,
        }

    # ========================================================
    # PATIENT DSES FOR EVERY STATE (diseases + Healthy)
    # ========================================================

    patient_dses = {}
    patient_dses_details = {}

    raw_dses_min = safe_float(signature.get("raw_dses_min", 0.0), 0.0)
    raw_dses_max = safe_float(signature.get("raw_dses_max", 100.0), 100.0)
    dses_low = safe_float(signature.get("dses_range_low", 1.0), 1.0)
    dses_high = safe_float(signature.get("dses_range_high", 100.0), 100.0)

    for disease, reactions in DSES_MAPPING.items():
        components = []
        used_reactions = []

        for reaction in reactions:
            if reaction not in patient_components:
                continue

            factor = safe_float(
                DISEASE_REACTION_FACTORS
                .get(disease, {})
                .get(reaction, 1.0),
                1.0,
            )

            component_value = (
                patient_components[reaction]["entropy_ratio"]
                * patient_components[reaction]["stress_mean"]
                * max(factor, SOFT_FLOOR)
            )
            components.append(component_value)
            used_reactions.append(reaction)

        raw_dses = max(
            float(np.mean(components)) if components else SOFT_FLOOR,
            SOFT_FLOOR,
        )

        if raw_dses_max - raw_dses_min <= SOFT_FLOOR:
            scaled_dses = (dses_low + dses_high) / 2.0
        else:
            scaled_dses = (
                dses_low
                + (
                    (raw_dses - raw_dses_min)
                    * (dses_high - dses_low)
                    / (raw_dses_max - raw_dses_min)
                )
            )

        patient_dses[disease] = float(scaled_dses)
        patient_dses_details[disease] = {
            "raw_dses": raw_dses,
            "mapped_reactions": reactions,
            "used_reactions": used_reactions,
            "missing_reactions": [
                r for r in reactions if r not in used_reactions
            ],
        }

    # ========================================================
    # PATIENT-CONDITIONED EXPECTED DSES
    #
    # expected_d = disease_signature_scalar_d
    #              + (RF_mean_d - median_RF_across_states)
    # ========================================================

    expected_dses = {}
    expected_dses_uncertainty = {}
    expected_dses_details = {}
    raw_rf_means = {}

    progress = st.progress(
        0,
        text="Calculating patient-conditioned expected DSES...",
    )

    diseases_sorted = sorted(DSES_MAPPING.keys())

    for idx, disease in enumerate(diseases_sorted, start=1):
        reactions = DSES_MAPPING.get(disease, [])
        predictions = []
        uncertainties = []
        rows = []

        for reaction in reactions:
            try:
                predicted, uncertainty = rf_expected_dses_and_uncertainty(
                    disease, reaction
                )
                if np.isfinite(predicted):
                    predictions.append(predicted)
                if np.isfinite(uncertainty):
                    uncertainties.append(uncertainty)
                rows.append(
                    {
                        "Biochemical Reaction": reaction,
                        "RF Expected DSES": predicted,
                        "RF Predictive Spread": uncertainty,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "Biochemical Reaction": reaction,
                        "RF Expected DSES": np.nan,
                        "RF Predictive Spread": np.nan,
                        "Error": str(exc),
                    }
                )

        rf_mean = (
            float(np.mean(predictions)) if predictions else np.nan
        )
        raw_rf_means[disease] = rf_mean
        expected_dses_uncertainty[disease] = (
            float(np.mean(uncertainties)) if uncertainties else np.nan
        )
        expected_dses_details[disease] = rows

        progress.progress(
            idx / max(len(diseases_sorted), 1),
            text=f"Calculating expected DSES: {idx}/{len(diseases_sorted)}",
        )

    progress.empty()

    finite_rf = [v for v in raw_rf_means.values() if np.isfinite(v)]
    rf_center = float(np.median(finite_rf)) if finite_rf else 0.0

    scalar_values = [
        safe_float(v, np.nan)
        for v in DISEASE_SIGNATURE_SCALARS.values()
    ]
    scalar_values = [v for v in scalar_values if np.isfinite(v) and v > 0]
    median_scalar = (
        float(np.median(scalar_values)) if scalar_values else rf_center
    )

    for disease in diseases_sorted:
        rf_mean = raw_rf_means.get(disease, np.nan)
        scalar = safe_float(
            DISEASE_SIGNATURE_SCALARS.get(disease, median_scalar),
            median_scalar,
        )
        if np.isfinite(rf_mean):
            expected_dses[disease] = float(scalar + (rf_mean - rf_center))
        else:
            expected_dses[disease] = float(scalar)

    # ========================================================
    # DSES DISTANCE + PROBABILITY (includes Healthy)
    # ========================================================

    valid_diseases = [
        disease
        for disease in diseases_sorted
        if (
            disease in patient_dses
            and np.isfinite(expected_dses.get(disease, np.nan))
        )
    ]

    if not valid_diseases:
        st.error(
            "No patient-conditioned expected DSES values could be calculated."
        )
        st.stop()

    distances = {
        disease: abs(patient_dses[disease] - expected_dses[disease])
        for disease in valid_diseases
    }

    uncertainty_values = [
        safe_float(expected_dses_uncertainty.get(disease, np.nan), np.nan)
        for disease in valid_diseases
    ]
    positive_uncertainties = np.asarray(
        [x for x in uncertainty_values if np.isfinite(x) and x > SOFT_FLOOR],
        dtype=float,
    )

    expected_values = np.asarray(
        [
            expected_dses[d]
            for d in valid_diseases
            if np.isfinite(expected_dses[d])
        ],
        dtype=float,
    )

    if expected_values.size >= 2:
        q1, q3 = np.percentile(expected_values, [25, 75])
        expected_iqr = float(q3 - q1)
        expected_mad = float(
            np.median(np.abs(expected_values - np.median(expected_values)))
        )
    else:
        expected_iqr = 0.0
        expected_mad = 0.0

    robust_rf_spread = (
        float(np.median(positive_uncertainties))
        if positive_uncertainties.size
        else 0.0
    )

    uncertainty_floor_candidates = [
        robust_rf_spread * 0.50,
        expected_iqr * 0.25,
        expected_mad * 1.4826,
        1.0,
    ]
    uncertainty_floor = max(
        [x for x in uncertainty_floor_candidates if np.isfinite(x)]
        + [SOFT_FLOOR]
    )

    standardized_distances = {}
    error_sources = {}

    for disease in valid_diseases:
        rf_spread = safe_float(
            expected_dses_uncertainty.get(disease, np.nan), np.nan
        )
        if np.isfinite(rf_spread) and rf_spread > uncertainty_floor:
            scale = float(rf_spread)
            source = "Disease-specific RF tree spread"
        else:
            scale = float(uncertainty_floor)
            source = "Robust uncertainty floor"

        error_sources[disease] = {"scale": scale, "source": source}
        standardized_distances[disease] = (
            distances[disease] / max(scale, SOFT_FLOOR)
        )

    STUDENT_T_DF = 4.0
    log_likelihoods = {}
    for disease in valid_diseases:
        z = standardized_distances[disease]
        log_likelihoods[disease] = (
            -0.5 * (STUDENT_T_DF + 1.0)
            * np.log1p((z ** 2) / STUDENT_T_DF)
        )

    log_values = np.asarray(
        [log_likelihoods[d] for d in valid_diseases], dtype=float
    )
    max_log = float(np.max(log_values))
    exp_values = np.exp(log_values - max_log)
    probability_denominator = float(np.sum(exp_values))

    dses_probability = {}
    for disease, exp_value in zip(valid_diseases, exp_values):
        dses_probability[disease] = float(
            exp_value / max(probability_denominator, SOFT_FLOOR)
        )

    dses_match_index = {
        disease: float(1.0 / (1.0 + standardized_distances[disease] ** 2))
        for disease in valid_diseases
    }

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    result_rows = []
    for disease in valid_diseases:
        result_rows.append(
            {
                "Disease": disease,
                "Patient DSES": patient_dses[disease],
                "Expected DSES (RF)": expected_dses[disease],
                "DSES Distance": distances[disease],
                "RF Predictive Spread": error_sources[disease]["scale"],
                "Standardized Distance": standardized_distances[disease],
                "DSES Relative Likelihood": float(
                    np.exp(log_likelihoods[disease] - max_log)
                ),
                "DSES Probability": dses_probability[disease],
                "DSES Match Index": dses_match_index[disease],
            }
        )

    disease_results = (
        pd.DataFrame(result_rows)
        .sort_values("DSES Probability", ascending=False)
        .reset_index(drop=True)
    )
    disease_results.insert(
        0, "Rank", np.arange(1, len(disease_results) + 1)
    )

    top = disease_results.iloc[0]
    top_disease = top["Disease"]
    top_probability = top["DSES Probability"]
    top_patient_dses = top["Patient DSES"]
    top_expected_dses = top["Expected DSES (RF)"]
    top_distance = top["DSES Distance"]
    top_std_distance = top["Standardized Distance"]

    healthy_row = disease_results[
        disease_results["Disease"] == HEALTHY_LABEL
    ]
    healthy_probability = (
        float(healthy_row["DSES Probability"].iloc[0])
        if not healthy_row.empty
        else np.nan
    )

    # ========================================================
    # DIGITAL TWIN RESULTS
    # ========================================================

    st.header("Digital Twin Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stroke Volume", f"{sv:.2f} mL")
    m2.metric("Ejection Fraction", f"{ef * 100:.2f}%")
    m3.metric("Entropy", f"{entropy:.4f} J/K")
    m4.metric("Entropy Stress", f"{entropy_stress:.3f}")

    # ========================================================
    # MAIN RANKING (includes Healthy)
    # ========================================================

    st.header("DSES Disease Ranking")

    st.caption(
        "Expected DSES combines each state's signature scalar "
        "(unique center, including Healthy) with a patient-specific "
        "RF shift. Probabilities are normalized across all states "
        "so they sum to 100%, including P(Healthy)."
    )

    display_cols = [
        "Rank",
        "Disease",
        "Patient DSES",
        "Expected DSES (RF)",
        "DSES Distance",
        "RF Predictive Spread",
        "Standardized Distance",
        "DSES Probability",
        "DSES Match Index",
    ]

    st.dataframe(
        disease_results[display_cols].style.format(
            {
                "Patient DSES": "{:.3f}",
                "Expected DSES (RF)": "{:.3f}",
                "DSES Distance": "{:.3f}",
                "RF Predictive Spread": "{:.3f}",
                "Standardized Distance": "{:.3f}",
                "DSES Probability": "{:.2%}",
                "DSES Match Index": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if top_disease == HEALTHY_LABEL:
        st.success(
            f"Top DSES-matched state: **Healthy** "
            f"(DSES-derived probability = {top_probability:.2%})"
        )
    else:
        st.success(
            f"Top DSES-matched disease: {top_disease} "
            f"(DSES-derived probability = {top_probability:.2%})"
        )

    if np.isfinite(healthy_probability):
        st.info(
            f"**Healthy-state probability** = {healthy_probability:.2%}  ·  "
            f"Patient DSES = {top_patient_dses:.3f} | "
            f"Expected DSES = {top_expected_dses:.3f} | "
            f"Distance = {top_distance:.3f} | "
            f"Standardized distance = {top_std_distance:.3f}"
        )
    else:
        st.info(
            f"Patient DSES = {top_patient_dses:.3f} | "
            f"Expected DSES = {top_expected_dses:.3f} | "
            f"Distance = {top_distance:.3f} | "
            f"Standardized distance = {top_std_distance:.3f}"
        )

    if top_std_distance <= 0.5:
        match_band = "Very high DSES match"
    elif top_std_distance <= 1.0:
        match_band = "High DSES match"
    elif top_std_distance <= 2.0:
        match_band = "Moderate DSES match"
    elif top_std_distance <= 3.0:
        match_band = "Low DSES match"
    else:
        match_band = "Very low DSES match"

    st.caption(
        f"Interpretation of the leading standardized distance: "
        f"**{match_band}**. "
        "This is a model-based DSES interpretation, not a diagnosis."
    )

    st.info(
        "DSES-derived probability uses a Student-t distance likelihood "
        "and uniform priors over all evaluated states (diseases + Healthy). "
        "Probabilities sum to 100% across this candidate set. "
        "This is NOT a clinically calibrated probability."
    )

    # ========================================================
    # CHARTS
    # ========================================================

    fig_compat = go.Figure(
        go.Bar(
            x=disease_results["DSES Match Index"],
            y=disease_results["Disease"],
            orientation="h",
        )
    )
    fig_compat.update_layout(
        title="Independent DSES Match Index by State",
        xaxis_title="DSES Match Index (0–1, higher = closer)",
        yaxis_title="State",
        yaxis={
            "categoryorder": "array",
            "categoryarray": disease_results["Disease"].tolist(),
        },
    )
    st.plotly_chart(fig_compat, use_container_width=True)

    fig_probability = go.Figure(
        go.Bar(
            x=disease_results["DSES Probability"] * 100.0,
            y=disease_results["Disease"],
            orientation="h",
        )
    )
    fig_probability.update_layout(
        title="DSES-Derived State Probability (includes Healthy)",
        xaxis_title="DSES-derived probability (%)",
        yaxis_title="State",
        xaxis={"range": [0, 100]},
        yaxis={
            "categoryorder": "array",
            "categoryarray": disease_results["Disease"].tolist(),
        },
    )
    st.plotly_chart(fig_probability, use_container_width=True)

    fig_distance = go.Figure(
        go.Bar(
            x=disease_results["Disease"],
            y=disease_results["DSES Distance"],
        )
    )
    fig_distance.update_layout(
        title="Patient DSES vs Patient-Conditioned Expected DSES",
        xaxis_title="State",
        yaxis_title="Absolute DSES Distance",
        xaxis={"tickangle": -60},
    )
    st.plotly_chart(fig_distance, use_container_width=True)

    fig_std = go.Figure(
        go.Bar(
            x=disease_results["Disease"],
            y=disease_results["Standardized Distance"],
        )
    )
    fig_std.add_hline(
        y=1.0, line_dash="dash", annotation_text="1 RF-spread unit"
    )
    fig_std.update_layout(
        title="Uncertainty-Adjusted DSES Distance",
        xaxis_title="State",
        yaxis_title="Standardized Distance",
        xaxis={"tickangle": -60},
    )
    st.plotly_chart(fig_std, use_container_width=True)

    # ========================================================
    # WHY TOP RANKED
    # ========================================================

    st.header("Why Was This State Ranked First?")
    st.write(
        f"**{top_disease}** has the smallest uncertainty-adjusted "
        f"mismatch between the patient's DSES and the patient-conditioned "
        f"expected DSES. Absolute distance = **{top_distance:.3f}**, "
        f"standardized distance = **{top_std_distance:.3f}**, "
        f"model probability = **{top_probability:.2%}**."
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
        stress_df.style.format({"Value": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    fig_stress = go.Figure(
        go.Bar(x=stress_df["Stress Component"], y=stress_df["Value"])
    )
    fig_stress.add_hline(
        y=1.0, line_dash="dash", annotation_text="Reference = 1.0"
    )
    fig_stress.update_layout(
        title=f"Patient Stress Profile Supporting {top_disease}",
        xaxis_title="Stress Component",
        yaxis_title="Normalized Stress",
    )
    st.plotly_chart(fig_stress, use_container_width=True)

    # ========================================================
    # TOP STATE REACTION EXPLANATION
    # ========================================================

    st.subheader(f"{top_disease}: Reaction-Level DSES Explanation")

    top_reactions = DSES_MAPPING.get(top_disease, [])
    explanation_rows = []

    for reaction in top_reactions:
        if reaction not in patient_components:
            continue

        component = patient_components[reaction]
        factor = safe_float(
            DISEASE_REACTION_FACTORS
            .get(top_disease, {})
            .get(reaction, 1.0),
            1.0,
        )
        component_dses = (
            component["entropy_ratio"]
            * component["stress_mean"]
            * max(factor, SOFT_FLOOR)
        )

        expected_value = np.nan
        expected_spread = np.nan
        for row in expected_dses_details.get(top_disease, []):
            if row.get("Biochemical Reaction") == reaction:
                expected_value = row.get("RF Expected DSES", np.nan)
                expected_spread = row.get("RF Predictive Spread", np.nan)
                break

        explanation_rows.append(
            {
                "Biochemical Reaction": reaction,
                "Metabolic Stress": component["metabolic_stress"],
                "Mechanical Stress": component["mechanical_stress"],
                "Thermodynamic Stress": component["thermodynamic_stress"],
                "ATP Stress": component["atp_stress"],
                "Entropy Stress": component["entropy_stress"],
                "Entropy Ratio": component["entropy_ratio"],
                "Disease Factor": factor,
                "Patient DSES Component": component_dses,
                "RF Expected DSES": expected_value,
                "RF Predictive Spread": expected_spread,
            }
        )

    explanation_df = pd.DataFrame(explanation_rows)

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
                    "Disease Factor": "{:.3f}",
                    "Patient DSES Component": "{:.3f}",
                    "RF Expected DSES": "{:.3f}",
                    "RF Predictive Spread": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        fig_reaction = go.Figure()
        fig_reaction.add_bar(
            x=explanation_df["Biochemical Reaction"],
            y=explanation_df["Patient DSES Component"],
            name="Patient DSES Component",
        )
        fig_reaction.add_bar(
            x=explanation_df["Biochemical Reaction"],
            y=explanation_df["RF Expected DSES"],
            name="RF Expected DSES",
        )
        fig_reaction.update_layout(
            title=f"Patient DSES Component vs RF Expected DSES — {top_disease}",
            xaxis_title="Biochemical Reaction",
            yaxis_title="DSES",
            barmode="group",
        )
        st.plotly_chart(fig_reaction, use_container_width=True)

    # ========================================================
    # PER-REACTION RF INSPECTION
    # ========================================================

    with st.expander("Show Patient-Conditioned RF Expected DSES"):
        selected_reactions = DSES_MAPPING.get(disease_choice, [])
        inspection_rows = []
        for reaction in selected_reactions:
            try:
                predicted, uncertainty = rf_expected_dses_and_uncertainty(
                    disease_choice, reaction
                )
                inspection_rows.append(
                    {
                        "Biochemical Reaction": reaction,
                        "RF Expected DSES": predicted,
                        "RF Predictive Spread": uncertainty,
                    }
                )
            except Exception as exc:
                inspection_rows.append(
                    {
                        "Biochemical Reaction": reaction,
                        "RF Expected DSES": np.nan,
                        "RF Predictive Spread": np.nan,
                        "Error": str(exc),
                    }
                )
        st.dataframe(
            pd.DataFrame(inspection_rows).style.format(
                {
                    "RF Expected DSES": "{:.3f}",
                    "RF Predictive Spread": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # DIGITAL TWIN CALCULATIONS
    # ========================================================

    with st.expander("Show Digital Twin Calculations"):
        calculation_df = pd.DataFrame(
            {
                "Variable": [
                    "Temperature (K)",
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
                    "MAP",
                    "Cardiac Output",
                    "RPP",
                    "LV Stroke Work",
                    "MVO₂",
                    "Chemical Power",
                    "Mechanical Power",
                    "Heat Production",
                    "ATP Production",
                    "ATP Utilization",
                    "ATP Fraction",
                    "ATP Balance",
                    "Entropy Flow",
                    "Entropy",
                    "Entropy Stress",
                    "Metabolic Stress",
                    "Mechanical Stress",
                    "Thermodynamic Stress",
                    "ATP Stress",
                    "Total ΔG",
                ],
                "Value": [
                    T, hr, sbp, dbp, edv, esv, ph, spo2, fbs, chol,
                    sv, ef, map_pressure, co, rpp, lvsw, mvo2,
                    chemical_power, mechanical_power, heat_production,
                    atp_production, atp_utilization, atp_fraction,
                    atp_balance, entropy_flow, entropy, entropy_stress,
                    metabolic_stress, mechanical_stress,
                    thermodynamic_stress, atp_stress, total_dg,
                ],
            }
        )
        st.dataframe(
            calculation_df.style.format({"Value": "{:.6f}"}),
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # FULL DETAILS
    # ========================================================

    st.subheader("Full State DSES / Distance Details")
    st.dataframe(
        disease_results[
            [
                "Rank",
                "Disease",
                "Patient DSES",
                "Expected DSES (RF)",
                "DSES Distance",
                "RF Predictive Spread",
                "Standardized Distance",
                "DSES Relative Likelihood",
                "DSES Probability",
                "DSES Match Index",
            ]
        ].style.format(
            {
                "Patient DSES": "{:.3f}",
                "Expected DSES (RF)": "{:.3f}",
                "DSES Distance": "{:.3f}",
                "RF Predictive Spread": "{:.3f}",
                "Standardized Distance": "{:.3f}",
                "DSES Relative Likelihood": "{:.3f}",
                "DSES Probability": "{:.2%}",
                "DSES Match Index": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
