import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="Cardio-Thermodynamic Digital Twin",
    layout="wide"
)

st.title("Cardio-Thermodynamic Digital Twin")

# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------

clf = joblib.load("disease_model.pkl")
le = joblib.load("disease_label_encoder.pkl")
refs = joblib.load("dt_reference_values.pkl")

T = 310.15


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
        step=1.0
    )

    sbp = st.number_input(
        "Systolic BP (mmHg)",
        min_value=50.0,
        max_value=250.0,
        value=120.0,
        step=1.0
    )

    dbp = st.number_input(
        "Diastolic BP (mmHg)",
        min_value=30.0,
        max_value=150.0,
        value=80.0,
        step=1.0
    )

with c2:

    edv = st.number_input(
        "End-Diastolic Volume (mL)",
        min_value=20.0,
        max_value=500.0,
        value=140.0,
        step=1.0
    )

    esv = st.number_input(
        "End-Systolic Volume (mL)",
        min_value=5.0,
        max_value=400.0,
        value=60.0,
        step=1.0
    )

    ph = st.number_input(
        "pH",
        min_value=6.5,
        max_value=8.0,
        value=7.40,
        step=0.01
    )

with c3:

    spo2 = st.number_input(
        "SpO₂ (%)",
        min_value=50.0,
        max_value=100.0,
        value=98.0,
        step=1.0
    )

    fbs = st.number_input(
        "Fasting Blood Sugar",
        min_value=20.0,
        max_value=600.0,
        value=90.0,
        step=1.0
    )

    chol = st.number_input(
        "Serum Cholesterol",
        min_value=50.0,
        max_value=600.0,
        value=180.0,
        step=1.0
    )


run_dt = st.button(
    "RUN DIGITAL TWIN",
    type="primary",
    use_container_width=True
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

    ef = sv / edv

    map_pressure = (
        dbp + (sbp - dbp) / 3.0
    )

    co = sv * hr

    rpp = hr * sbp

    lvsw = sv * map_pressure * 0.000133322

    # --------------------------------------------------------
    # Metabolic layer
    # --------------------------------------------------------

    RPP_ref = 10000
    LVSW_ref = 0.9
    MVO2_rest = 8.0
    MYOCARDIAL_MASS = 300
    O2_ENERGY = 20.2

    rpp_component = rpp / RPP_ref

    lvsw_component = lvsw / LVSW_ref

    metabolic_demand = (
        0.7 * rpp_component
        + 0.3 * lvsw_component
    )

    mvo2 = MVO2_rest * metabolic_demand

    o2_consumption = (
        mvo2 * MYOCARDIAL_MASS / 100
    )

    chemical_power = (
        o2_consumption * O2_ENERGY / 60.0
    )

    mechanical_power = (
        lvsw * hr / 60.0
    )

    heat_production = max(
        chemical_power - mechanical_power,
        0
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
        / ATP_ENERGY
    )

    mechanical_energy = mechanical_power * 60.0

    atp_utilization = (
        mechanical_energy / ATP_ENERGY
    )

    atp_fraction = (
        atp_utilization / atp_production
        if atp_production != 0
        else np.nan
    )

    atp_balance = (
        atp_production - atp_utilization
    )

    # --------------------------------------------------------
    # Thermodynamic proxy layer
    # Uses the same formulas as the notebook.
    # --------------------------------------------------------

    R = 8.314

    safe_atp_balance = max(
        atp_balance,
        1e-6
    )

    Q_metabolism = (
        ((fbs + 1.0) * chol)
        / (spo2 * safe_atp_balance)
    )

    Q_metabolism = max(
        Q_metabolism,
        1e-12
    )

    dg_metabolism = (
        -2870000
        + R * T * np.log(Q_metabolism)
    )

    Q_atp = (
        atp_utilization / atp_production
        if atp_production > 0
        else 1e-12
    )

    dg_atp = (
        -30500
        + R * T * np.log(
            max(Q_atp, 1e-12)
        )
    )

    pH_factor = 10 ** (7.4 - ph)

    Q_ion = max(
        atp_fraction * pH_factor,
        1e-12
    )

    dg_ion = (
        -50000
        + R * T * np.log(Q_ion)
    )

    calcium_factor = (
        ph
        * hr
        / hr
    )

    Q_calcium = max(
        atp_fraction * calcium_factor,
        1e-12
    )

    dg_calcium = (
        -50000
        + R * T * np.log(Q_calcium)
    )

    redox_factor = spo2 / 100.0

    Q_redox = max(
        atp_fraction * redox_factor,
        1e-12
    )

    dg_redox = (
        -220000
        + R * T * np.log(Q_redox)
    )

    no_factor = (
        ph * spo2 / 100.0
    )

    Q_no = max(
        atp_fraction * no_factor,
        1e-12
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
    # ENTROPY
    # --------------------------------------------------------

    entropy_flow = (
        heat_production * 60.0 / T
    )

    entropy = entropy_flow * 1.0

    entropy_stress = (
        entropy_flow
        / refs["Entropy_Flow_median"]
    )

    # --------------------------------------------------------
    # Other ML stress features
    # --------------------------------------------------------

    metabolic_stress = (
        mvo2
        / refs["MVO2_median"]
    )

    mechanical_stress = (
        mechanical_power
        / refs["Mechanical_Power_median"]
    )

    thermodynamic_stress = (
        abs(total_dg)
        / refs["Total_DG_abs_median"]
    )

    atp_stress = (
        atp_fraction
        / refs["ATP_Utilization_Fraction_median"]
    )

    # --------------------------------------------------------
    # Feature vector
    # --------------------------------------------------------

    patient_features = pd.DataFrame([{
        "Metabolic Stress": metabolic_stress,
        "Mechanical Stress": mechanical_stress,
        "Thermodynamic Stress": thermodynamic_stress,
        "ATP Stress": atp_stress,
        "Entropy Stress": entropy_stress
    }])

    # --------------------------------------------------------
    # ALL DISEASE PROBABILITIES
    # --------------------------------------------------------

    probabilities = clf.predict_proba(
        patient_features
    )[0]

    disease_results = pd.DataFrame({
        "Disease": le.classes_,
        "Probability": probabilities
    })

    disease_results["Probability (%)"] = (
        disease_results["Probability"] * 100.0
    )

    disease_results = (
        disease_results
        .sort_values(
            "Probability",
            ascending=False
        )
        .reset_index(drop=True)
    )

    disease_results.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(disease_results) + 1
        )
    )

    # ========================================================
    # DISPLAY DT OUTPUTS
    # ========================================================

    st.header("Digital Twin Results")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Stroke Volume",
        f"{sv:.2f} mL"
    )

    m2.metric(
        "Ejection Fraction",
        f"{ef * 100:.2f}%"
    )

    m3.metric(
        "Entropy",
        f"{entropy:.4f} J/K"
    )

    m4.metric(
        "Entropy Stress",
        f"{entropy_stress:.3f}"
    )

    # ========================================================
    # DISEASE RANKING
    # ========================================================

    st.header(
        "Disease Probability Ranking"
    )

    st.dataframe(
        disease_results[
            [
                "Rank",
                "Disease",
                "Probability (%)"
            ]
        ].style.format({
            "Probability (%)": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # DISEASE BAR CHART
    # ========================================================

    fig = go.Figure(
        go.Bar(
            x=disease_results["Probability (%)"],
            y=disease_results["Disease"],
            orientation="h"
        )
    )

    fig.update_layout(
        title="All Predicted Disease Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="Disease",
        yaxis={
            "categoryorder": "array",
            "categoryarray": disease_results["Disease"].tolist()
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.success(
        f"Highest model-predicted disease probability: "
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
            pd.DataFrame({
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
                    "Total ΔG"
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
                    total_dg
                ]
            }),
            use_container_width=True,
            hide_index=True
        )


