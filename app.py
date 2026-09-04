# ============================================================
# app.py
# Cardiac DSES Digital Twin - FINAL 28-CLASS ARCHITECTURE
#
# This app uses the final multiclass architecture:
#
#   Patient at rest
#        ↓
#   Digital Twin
#        ↓
#   28 Patient-DSES values
#        +
#   Conditional Expected-DSES RF
#        ↓
#   28 Expected-DSES values
#        ↓
#   28 DSES residuals
#        +
#   5 global stress features
#        +
#   5 Digital-Twin physiological features
#        +
#   compact clinical feature layer
#        ↓
#   Final 28-class ExtraTrees classifier
#        ↓
#   28 disease probabilities
#
# IMPORTANT:
# - This replaces the older 50/50 DT + ML ranking.
# - This replaces the older Student-t DSES-distance ranking.
# - Disease is the prediction target, not a final classifier input.
# - Patient is assumed to be at rest.
# - Smoking is fixed to NO.
# ============================================================

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. ROBUST PATH
# ============================================================

BASE_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)


# ============================================================
# 2. REQUIRED FILE CHECK
# ============================================================

REQUIRED_FILES = [
    "digital_twin.py",
    "ml_final_28class_FIXED.py",
    "disease_dses_rf.joblib",
    "disease_dses_model_columns.joblib",
    "disease_dses_model_medians.joblib",
    "final_28class_dses_classifier.joblib",
    "final_28class_feature_columns.joblib",
]

missing = [
    filename
    for filename in REQUIRED_FILES
    if not (BASE_DIR / filename).exists()
]


# ============================================================
# 3. IMPORT MODEL MODULES
# ============================================================

if not missing:
    try:
        # The existing final classifier module loads the Digital Twin
        # itself. It is kept separate from Streamlit.
        sys.path.insert(0, str(BASE_DIR))

        from digital_twin import (
            DISEASE_REACTION_MAP,
            build_or_load_references,
            compute_patient_dses_scores,
        )

        from ml_final_28class_FIXED import (
            build_features,
            predict_patient,
        )

    except Exception as exc:
        st.set_page_config(
            page_title="Cardiac DSES Digital Twin",
            layout="wide",
        )
        st.title("Cardiac DSES Digital Twin")
        st.error(
            "The 87% architecture could not be loaded."
        )
        st.exception(exc)
        st.stop()
else:
    st.set_page_config(
        page_title="Cardiac DSES Digital Twin",
        layout="wide",
    )
    st.title("Cardiac DSES Digital Twin")

    st.error(
        "Required 87% architecture files are missing."
    )

    st.write(
        "Place these files in the same folder as app.py:"
    )

    for filename in missing:
        st.write(f"- `{filename}`")

    st.stop()


# ============================================================
# 4. PAGE CONFIG
# ============================================================

# set_page_config may already have been called above in an error
# branch. In normal execution, it has not been called yet.
try:
    st.set_page_config(
        page_title="Cardiac DSES Digital Twin",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

st.title("Cardio-Thermodynamic Digital Twin")
st.caption(
    "Final 28-class DSES model | Resting patient | Smoking fixed to NO"
)


# ============================================================
# 5. REFERENCES / DISEASES
# ============================================================

@st.cache_resource(show_spinner=False)
def get_references():
    return build_or_load_references()


@st.cache_resource(show_spinner=False)
def get_diseases():
    return [
        disease
        for disease in DISEASE_REACTION_MAP.keys()
        if str(disease).strip().lower() != "healthy"
    ]


try:
    refs = get_references()
    diseases = get_diseases()
except Exception as exc:
    st.error(
        f"Could not load Digital Twin reference data: {exc}"
    )
    st.stop()


# ============================================================
# 6. SIDEBAR - PATIENT INPUT
# ============================================================

with st.sidebar:

    st.header("Patient Input")

    age = st.number_input(
        "Age (years)",
        min_value=18.0,
        max_value=120.0,
        value=55.0,
        step=1.0,
    )

    sex = st.selectbox(
        "Biological Sex",
        ["Female", "Male"],
    )

    height = st.number_input(
        "Height (cm)",
        min_value=80.0,
        max_value=230.0,
        value=170.0,
        step=0.1,
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=20.0,
        max_value=250.0,
        value=75.0,
        step=0.1,
    )

    st.divider()

    st.subheader("Hemodynamics")

    hr = st.number_input(
        "Resting Heart Rate (bpm)",
        min_value=30.0,
        max_value=200.0,
        value=70.0,
        step=1.0,
    )

    sbp = st.number_input(
        "Systolic BP (mmHg)",
        min_value=70.0,
        max_value=250.0,
        value=120.0,
        step=1.0,
    )

    dbp = st.number_input(
        "Diastolic BP (mmHg)",
        min_value=40.0,
        max_value=160.0,
        value=80.0,
        step=1.0,
    )

    edv = st.number_input(
        "End-Diastolic Volume (mL)",
        min_value=20.0,
        max_value=500.0,
        value=150.0,
        step=1.0,
    )

    esv = st.number_input(
        "End-Systolic Volume (mL)",
        min_value=5.0,
        max_value=400.0,
        value=60.0,
        step=1.0,
    )

    pawp = st.number_input(
        "PAWP (mmHg)",
        min_value=0.0,
        max_value=40.0,
        value=12.0,
        step=0.1,
    )

    myocardial_mass = st.number_input(
        "Myocardial Mass (g)",
        min_value=50.0,
        max_value=500.0,
        value=150.0,
        step=1.0,
    )

    core_temp = st.number_input(
        "Core Temperature (°C)",
        min_value=34.0,
        max_value=41.0,
        value=37.0,
        step=0.01,
    )

    spo2 = st.number_input(
        "SpO₂ (%)",
        min_value=70.0,
        max_value=100.0,
        value=98.0,
        step=0.1,
    )

    ph = st.number_input(
        "Arterial pH",
        min_value=6.8,
        max_value=7.8,
        value=7.40,
        step=0.001,
    )

    fbs = st.number_input(
        "Fasting Blood Sugar (mg/dL)",
        min_value=50.0,
        max_value=500.0,
        value=90.0,
        step=1.0,
    )

    chol = st.number_input(
        "Serum Cholesterol (mg/dL)",
        min_value=80.0,
        max_value=500.0,
        value=200.0,
        step=1.0,
    )

    st.divider()

    st.subheader("Clinical Phenotype")

    chest_pain = st.selectbox(
        "Chest Pain Type",
        [
            "asymptomatic",
            "typical angina",
            "atypical angina",
            "non-anginal pain",
        ],
    )

    rest_ecg = st.selectbox(
        "Resting ECG",
        [
            "Normal",
            "Left ventricular hypertrophy",
            "ST-T wave abnormality",
        ],
    )

    run_button = st.button(
        "RUN 28-CLASS MODEL",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# 7. PATIENT DICTIONARY
# ============================================================

patient = {
    "age": age,
    "sex": sex,
    "height_cm": height,
    "weight_kg": weight,
    "hr": hr,
    "sbp": sbp,
    "dbp": dbp,
    "edv": edv,
    "esv": esv,
    "pawp": pawp,
    "myocardial_mass_g": myocardial_mass,
    "temp_c": core_temp,
    "spo2": spo2,
    "ph": ph,
    "fbs": fbs,
    "chol": chol,
    "chest_pain": chest_pain,
    "rest_ecg": rest_ecg,
}


# ============================================================
# 8. RUN MODEL
# ============================================================

if run_button:

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if edv <= esv:
        st.error("EDV must be greater than ESV.")
        st.stop()

    if sbp <= dbp:
        st.error("Systolic BP must be greater than DBP.")
        st.stop()

    if height <= 0:
        st.error("Height must be greater than zero.")
        st.stop()

    if weight <= 0:
        st.error("Weight must be greater than zero.")
        st.stop()

    # --------------------------------------------------------
    # Run Digital Twin + final classifier
    # --------------------------------------------------------

    with st.spinner(
        "Running Digital Twin and final 28-class classifier..."
    ):

        try:
            dt_result = compute_patient_dses_scores(
                patient,
                refs,
            )

            final_predictions = predict_patient(
                patient,
                refs,
            )

            # Build the same final feature vector used during
            # final-classifier development so the intermediate
            # DSES/stress/physiology features can be displayed.
            model_input = pd.DataFrame([
                {
                    "Age_years": age,
                    "Biological_Sex": sex,
                    "Height_cm": height,
                    "Weight_kg": weight,
                    "Heart_Rate_bpm": hr,
                    "Systolic_BP_mmHg": sbp,
                    "Diastolic_BP_mmHg": dbp,
                    "EDV_mL": edv,
                    "ESV_mL": esv,
                    "PAWP_mmHg": pawp,
                    "Core_Temp_C": core_temp,
                    "SpO2_pct": spo2,
                    "pH": ph,
                    "Fasting_Blood_Sugar": fbs,
                    "Serum_Cholesterol": chol,
                    "Myocardial_Mass_g": myocardial_mass,
                    "Chest_Pain_Type": chest_pain,
                    "Resting_ECG": rest_ecg,
                }
            ])

            final_features = build_features(
                model_input,
                refs,
            )

        except Exception as exc:
            st.error(
                "Model execution failed."
            )
            st.exception(exc)
            st.stop()

    # ========================================================
    # TOP PREDICTION
    # ========================================================

    top_row = final_predictions.iloc[0]

    top_disease = str(
        top_row["Disease"]
    )

    top_probability = float(
        top_row["Probability"]
    )

    common = dt_result["common"]

    # ========================================================
    # TOP METRICS
    # ========================================================

    st.header("Final Prediction")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Predicted Disease",
        top_disease,
    )

    c2.metric(
        "Model Probability",
        f"{top_probability:.2%}",
    )

    c3.metric(
        "Stroke Volume",
        f"{common['SV']:.2f} mL",
    )

    c4.metric(
        "Ejection Fraction",
        f"{common['EF'] * 100:.2f}%",
    )

    c5.metric(
        "MAP",
        f"{common['MAP']:.2f} mmHg",
    )

    st.success(
        f"Top 28-class prediction: **{top_disease}** "
        f"with model probability **{top_probability:.2%}**."
    )

    # ========================================================
    # ARCHITECTURE PIPELINE
    # ========================================================

    st.header("87% Architecture")

    pipeline = go.Figure(
        go.Sankey(
            node=dict(
                label=[
                    "Patient at Rest",
                    "Digital Twin",
                    "28 Patient DSES",
                    "Expected-DSES RF",
                    "28 Expected DSES",
                    "28 DSES Residuals",
                    "5 Global Stresses",
                    "5 DT Physiology",
                    "Clinical Layer",
                    "Final 28-Class Classifier",
                    "28 Disease Probabilities",
                ]
            ),
            link=dict(
                source=[
                    0, 1, 2, 3, 4,
                    5, 6, 7, 8, 9
                ],
                target=[
                    1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10
                ],
                value=[1] * 10,
            ),
        )
    )

    pipeline.update_layout(
        title="Final Cardiac DSES Prediction Pipeline",
        height=400,
    )

    st.plotly_chart(
        pipeline,
        use_container_width=True,
    )

    # ========================================================
    # 28-CLASS RANKING TABLE
    # ========================================================

    st.header("28-Class Disease Ranking")

    ranking = final_predictions.copy()

    ranking.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(ranking) + 1,
        ),
    )

    ranking["Probability"] = (
        ranking["Probability"] * 100.0
    )

    st.dataframe(
        ranking[
            [
                "Rank",
                "Disease",
                "Probability",
            ]
        ].style.format(
            {
                "Probability": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # TOP 10 CHART
    # ========================================================

    top10 = (
        ranking
        .head(10)
        .iloc[::-1]
    )

    fig_top10 = go.Figure(
        go.Bar(
            x=top10["Probability"],
            y=top10["Disease"],
            orientation="h",
        )
    )

    fig_top10.update_layout(
        title="Top 10 Final 28-Class Predictions",
        xaxis_title="Classifier Probability (%)",
        yaxis_title="Disease",
        height=520,
    )

    st.plotly_chart(
        fig_top10,
        use_container_width=True,
    )

    # ========================================================
    # PATIENT DSES / EXPECTED DSES / RESIDUALS
    # ========================================================

    st.header(
        "DSES Features Used by the Final Classifier"
    )

    dses_rows = []

    for disease in diseases:

        p_col = f"P_{disease}"
        e_col = f"E_{disease}"
        r_col = f"R_{disease}"

        dses_rows.append(
            {
                "Disease": disease,
                "Patient DSES": (
                    float(
                        final_features.iloc[0][p_col]
                    )
                    if p_col in final_features.columns
                    else np.nan
                ),
                "Expected DSES": (
                    float(
                        final_features.iloc[0][e_col]
                    )
                    if e_col in final_features.columns
                    else np.nan
                ),
                "DSES Residual": (
                    float(
                        final_features.iloc[0][r_col]
                    )
                    if r_col in final_features.columns
                    else np.nan
                ),
            }
        )

    dses_table = pd.DataFrame(
        dses_rows
    )

    dses_table = dses_table.merge(
        final_predictions,
        on="Disease",
        how="left",
    )

    dses_table["Probability"] = (
        dses_table["Probability"] * 100.0
    )

    dses_table = (
        dses_table
        .sort_values(
            "Probability",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    dses_table.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(dses_table) + 1,
        ),
    )

    st.dataframe(
        dses_table.style.format(
            {
                "Patient DSES": "{:.3f}",
                "Expected DSES": "{:.3f}",
                "DSES Residual": "{:.3f}",
                "Probability": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # PATIENT VS EXPECTED DSES
    # ========================================================

    fig_dses = go.Figure()

    selected_dses = (
        dses_table
        .head(10)
        .iloc[::-1]
    )

    fig_dses.add_trace(
        go.Bar(
            x=selected_dses["Patient DSES"],
            y=selected_dses["Disease"],
            orientation="h",
            name="Patient DSES",
        )
    )

    fig_dses.add_trace(
        go.Bar(
            x=selected_dses["Expected DSES"],
            y=selected_dses["Disease"],
            orientation="h",
            name="Expected DSES",
        )
    )

    fig_dses.update_layout(
        title="Patient DSES vs Expected DSES - Top 10",
        xaxis_title="DSES",
        yaxis_title="Disease",
        barmode="group",
        height=520,
    )

    st.plotly_chart(
        fig_dses,
        use_container_width=True,
    )

    # ========================================================
    # DSES RESIDUAL CHART
    # ========================================================

    residual_plot = (
        dses_table
        .head(15)
        .iloc[::-1]
    )

    fig_residual = go.Figure(
        go.Bar(
            x=residual_plot["DSES Residual"],
            y=residual_plot["Disease"],
            orientation="h",
        )
    )

    fig_residual.add_vline(
        x=0,
        line_dash="dash",
    )

    fig_residual.update_layout(
        title="DSES Residuals - Top 15 Predicted Diseases",
        xaxis_title="Patient DSES - Expected DSES",
        yaxis_title="Disease",
        height=620,
    )

    st.plotly_chart(
        fig_residual,
        use_container_width=True,
    )

    # ========================================================
    # GLOBAL STRESS FEATURES
    # ========================================================

    st.header("Global Digital-Twin Stress Features")

    stress_names = [
        "Metabolic Stress",
        "Mechanical Stress",
        "Thermodynamic Stress",
        "ATP Stress",
        "Entropy Stress",
    ]

    stress_values = []

    for name in stress_names:
        column = f"G_{name}"

        stress_values.append(
            float(
                final_features.iloc[0].get(
                    column,
                    np.nan,
                )
            )
        )

    stress_df = pd.DataFrame(
        {
            "Stress Component": stress_names,
            "Value": stress_values,
        }
    )

    st.dataframe(
        stress_df.style.format(
            {"Value": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig_stress = go.Figure(
        go.Bar(
            x=stress_df["Stress Component"],
            y=stress_df["Value"],
        )
    )

    fig_stress.add_hline(
        y=1.0,
        line_dash="dash",
        annotation_text="Reference = 1.0",
    )

    fig_stress.update_layout(
        title="Global Stress Profile",
        xaxis_title="Stress",
        yaxis_title="Normalized Value",
    )

    st.plotly_chart(
        fig_stress,
        use_container_width=True,
    )

    # ========================================================
    # DIGITAL TWIN PHYSIOLOGY
    # ========================================================

    st.header("Digital Twin Physiological State")

    physiology_rows = [
        (
            "Stroke Volume",
            common["SV"],
            "mL",
        ),
        (
            "Ejection Fraction",
            common["EF"] * 100.0,
            "%",
        ),
        (
            "Mean Arterial Pressure",
            common["MAP"],
            "mmHg",
        ),
        (
            "Cardiac Output",
            common["CO"],
            "mL/min proxy",
        ),
        (
            "Rate-Pressure Product",
            common["RPP"],
            "mmHg·bpm",
        ),
        (
            "LV Stroke Work",
            common["LVSW"],
            "J/beat",
        ),
        (
            "MVO₂",
            common["MVO2"],
            "DT units",
        ),
        (
            "Chemical Power",
            common["Chemical Power (W)"],
            "W",
        ),
        (
            "Mechanical Power",
            common["Mechanical Power (W)"],
            "W",
        ),
        (
            "Heat Production",
            common["Heat Production (W)"],
            "W",
        ),
        (
            "ATP Production",
            common["ATP Production (mol/min)"],
            "mol/min",
        ),
        (
            "ATP Utilization",
            common["ATP Utilization (mol/min)"],
            "mol/min",
        ),
        (
            "ATP Utilization Fraction",
            common["ATP Utilization Fraction"],
            "ratio",
        ),
        (
            "ATP Balance",
            common["ATP Balance (mol/min)"],
            "mol/min",
        ),
    ]

    physiology_df = pd.DataFrame(
        physiology_rows,
        columns=[
            "Variable",
            "Value",
            "Unit",
        ],
    )

    st.dataframe(
        physiology_df.style.format(
            {"Value": "{:.6f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # REACTION-LEVEL DETAIL FOR TOP DISEASE
    # ========================================================

    st.header(
        f"Reaction-Level Detail — {top_disease}"
    )

    top_reaction_details = (
        dt_result
        .get("reaction_details", {})
        .get(top_disease, {})
    )

    reaction_rows = []

    for reaction, detail in top_reaction_details.items():

        row = {
            "Biochemical Reaction": reaction
        }

        if isinstance(detail, dict):
            row.update(detail)

        reaction_rows.append(row)

    if reaction_rows:

        reaction_df = pd.DataFrame(
            reaction_rows
        )

        st.dataframe(
            reaction_df,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "No reaction-level details were available "
            "for the top predicted disease."
        )

    # ========================================================
    # FULL DIGITAL-TWIN CALCULATION DETAILS
    # ========================================================

    with st.expander(
        "Show Full Digital-Twin Calculations"
    ):

        calculation = []

        for key, value in common.items():

            if isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):

                calculation.append(
                    {
                        "Variable": key,
                        "Value": float(value),
                    }
                )

        calculation_df = pd.DataFrame(
            calculation
        )

        if not calculation_df.empty:
            st.dataframe(
                calculation_df.style.format(
                    {"Value": "{:.8f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

    # ========================================================
    # MODEL FEATURE COUNT
    # ========================================================

    with st.expander(
        "Show Final Classifier Feature Summary"
    ):

        feature_summary = pd.DataFrame(
            {
                "Feature Group": [
                    "Patient DSES",
                    "Expected DSES",
                    "DSES residuals",
                    "Global stress summaries",
                    "DT physiological features",
                    "Compact clinical layer",
                ],
                "Features": [
                    28,
                    28,
                    28,
                    5,
                    5,
                    12,
                ],
                "Total": [
                    "",
                    "",
                    "",
                    "",
                    "",
                    106,
                ],
            }
        )

        st.dataframe(
            feature_summary,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "The compact clinical layer contains the raw patient "
            "fields used by the final model after categorical encoding."
        )

    # ========================================================
    # MODEL INTERPRETATION NOTICE
    # ========================================================

    st.info(
        "The displayed probabilities come from the trained final "
        "28-class classifier. They are model probabilities, not "
        "clinically calibrated diagnostic probabilities."
    )

else:

    st.info(
        "Enter the patient's resting parameters in the sidebar "
        "and click **RUN 28-CLASS MODEL**."
    )
