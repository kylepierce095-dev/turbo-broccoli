# digital_twin.py
# Cardiac DSES Digital Twin — deterministic patient-side engine.
# Assumption for this build: patient is at rest and non-smoking.
import math
import numpy as np
from typing import Dict, Any

import os
import re
import pickle
import pandas as pd
from pathlib import Path
SOFT_FLOOR = 1e-6
DG_FLOOR = 1.0

RPP_REF = 10000.0
LVSW_REF = 0.9
MVO2_REST = 8.0
MYOCARDIAL_MASS = 300.0
O2_ENERGY = 20.2

ATP_ENERGY = 30500.0
ATP_COUPLING_EFFICIENCY = 0.60
R_GAS = 8.314
TEMPERATURE_K = 310.15

REFERENCE_WORKBOOK = (
    "Cardiac_Disease_Synchronized_Dataset_Patient_1_"
    "Literature_Conditioned_CLEAN (4)_2.xlsx"
)

REFERENCE_CACHE = "dt_reference_values_UPDATED.pkl"


REACTIONS = [
    "Metabolism",
    "ATP Utilization",
    "Ion Transport",
    "Calcium Handling",
    "Redox Metabolism",
    "Nitric Oxide Metabolism",
]


# ============================================================
# DISEASE -> REACTION MAPPING
# ============================================================

DISEASE_REACTION_MAP = {
    "Ischemic heart disease": ["Metabolism"],
    "Heart failure": [
        "Metabolism",
        "ATP Utilization",
        "Calcium Handling",
        "Redox Metabolism",
        "Nitric Oxide Metabolism",
    ],
    "Diabetic cardiomyopathy": ["Metabolism", "Redox Metabolism"],
    "Obesity-related cardiomyopathy": ["Metabolism"],
    "Metabolic cardiomyopathy": ["Metabolism"],
    "Mitochondrial cardiomyopathies": ["Metabolism"],
    "Ischemia": ["ATP Utilization"],
    "Myocardial infarction": ["ATP Utilization", "Redox Metabolism"],
    "Hypertrophic cardiomyopathy": ["ATP Utilization", "Calcium Handling"],
    "Mitochondrial diseases": ["ATP Utilization"],
    "ATP-sensitive potassium channel disorders": ["ATP Utilization"],
    "Cardiac arrhythmias": ["Ion Transport"],
    "Long QT syndrome": ["Ion Transport"],
    "Brugada syndrome": ["Ion Transport"],
    "Short QT syndrome": ["Ion Transport"],
    "Atrial fibrillation": ["Ion Transport"],
    "Heart block": ["Ion Transport"],
    "Sudden cardiac death": ["Ion Transport"],
    "Dilated cardiomyopathy": ["Calcium Handling"],
    "Arrhythmias": ["Calcium Handling"],
    "CPVT": ["Calcium Handling"],
    "Ischemia-reperfusion injury": ["Redox Metabolism"],
    "Atherosclerosis": ["Redox Metabolism", "Nitric Oxide Metabolism"],
    "Oxidative-stress cardiomyopathy": ["Redox Metabolism"],
    "Hypertension": ["Nitric Oxide Metabolism"],
    "Coronary artery disease": ["Nitric Oxide Metabolism"],
    "Endothelial dysfunction": ["Nitric Oxide Metabolism"],
    "Pulmonary hypertension": ["Nitric Oxide Metabolism"],
}


# Exact literature-conditioned factors used by the existing DT.
DISEASE_REACTION_FACTORS = {
    "ATP-sensitive potassium channel disorders": {
        "ATP Utilization": 1.2744,
    },
    "Arrhythmias": {
        "Calcium Handling": 1.3750,
    },
    "Atherosclerosis": {
        "Nitric Oxide Metabolism": 1.4560,
        "Redox Metabolism": 1.4950,
    },
    "Atrial fibrillation": {
        "Ion Transport": 1.4560,
    },
    "Brugada syndrome": {
        "Ion Transport": 1.4000,
    },
    "Cardiac arrhythmias": {
        "Ion Transport": 1.4336,
    },
    "CPVT": {
        "Calcium Handling": 1.5180,
    },
    "Coronary artery disease": {
        "Nitric Oxide Metabolism": 1.4560,
    },
    "Diabetic cardiomyopathy": {
        "Metabolism": 1.2800,
        "Redox Metabolism": 1.4720,
    },
    "Dilated cardiomyopathy": {
        "Calcium Handling": 1.4300,
    },
    "Endothelial dysfunction": {
        "Nitric Oxide Metabolism": 1.5456,
    },
    "Heart block": {
        "Ion Transport": 1.4336,
    },
    "Heart failure": {
        "ATP Utilization": 1.4040,
        "Calcium Handling": 1.4300,
        "Metabolism": 1.3000,
        "Nitric Oxide Metabolism": 1.4560,
        "Redox Metabolism": 1.4950,
    },
    "Hypertension": {
        "Nitric Oxide Metabolism": 1.4784,
    },
    "Hypertrophic cardiomyopathy": {
        "ATP Utilization": 1.3500,
        "Calcium Handling": 1.3750,
    },
    "Ischemia": {
        "ATP Utilization": 1.2744,
    },
    "Ischemia-reperfusion injury": {
        "Redox Metabolism": 1.6675,
    },
    "Ischemic heart disease": {
        "Metabolism": 1.1200,
    },
    "Long QT syndrome": {
        "Ion Transport": 1.3664,
    },
    "Metabolic cardiomyopathy": {
        "Metabolism": 1.2200,
    },
    "Mitochondrial cardiomyopathies": {
        "Metabolism": 1.3000,
    },
    "Mitochondrial diseases": {
        "ATP Utilization": 1.3824,
    },
    "Myocardial infarction": {
        "ATP Utilization": 1.4256,
        "Redox Metabolism": 1.5180,
    },
    "Obesity-related cardiomyopathy": {
        "Metabolism": 1.2200,
    },
    "Oxidative-stress cardiomyopathy": {
        "Redox Metabolism": 1.6675,
    },
    "Pulmonary hypertension": {
        "Nitric Oxide Metabolism": 1.6576,
    },
    "Short QT syndrome": {
        "Ion Transport": 1.3664,
    },
    "Sudden cardiac death": {
        "Ion Transport": 1.5680,
    },
}


# Stored disease-level signature scalars from the current model.
DISEASE_SIGNATURE_SCALARS = {
    "ATP-sensitive potassium channel disorders": 44.165400,
    "Arrhythmias": 68.794107,
    "Atherosclerosis": 58.513057,
    "Atrial fibrillation": 86.464036,
    "Brugada syndrome": 79.692878,
    "Cardiac arrhythmias": 83.755835,
    "CPVT": 85.252288,
    "Coronary artery disease": 48.168431,
    "Diabetic cardiomyopathy": 39.867094,
    "Dilated cardiomyopathy": 75.125979,
    "Endothelial dysfunction": 56.692023,
    "Heart block": 83.755835,
    "Heart failure": 53.040198,
    "Hypertension": 50.299553,
    "Hypertrophic cardiomyopathy": 60.670576,
    "Ischemia": 44.165400,
    "Ischemia-reperfusion injury": 87.274409,
    "Ischemic heart disease": 1.000632,
    "Long QT syndrome": 75.629118,
    "Metabolic cardiomyopathy": 9.213487,
    "Mitochondrial cardiomyopathies": 15.784194,
    "Mitochondrial diseases": 55.464860,
    "Myocardial infarction": 65.637029,
    "Obesity-related cardiomyopathy": 9.213487,
    "Oxidative-stress cardiomyopathy": 87.274409,
    "Pulmonary hypertension": 67.343278,
    "Short QT syndrome": 75.629118,
    "Sudden cardiac death": 99.984602,
}


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def finite_positive(x, floor=SOFT_FLOOR):
    x = float(x)
    if not np.isfinite(x):
        return float(floor)
    return max(x, float(floor))


def safe_ratio(a, b, floor=SOFT_FLOOR):
    return float(a) / finite_positive(b, floor)


def safe_log_q(q):
    return math.log(finite_positive(q, 1e-12))


# ============================================================
# COMMON DIGITAL-TWIN LAYER
# ============================================================

def compute_common_layer(
    hr,
    sbp,
    dbp,
    edv,
    esv,
):
    if edv <= esv:
        raise ValueError("EDV must be greater than ESV.")
    if sbp <= dbp:
        raise ValueError("SBP must be greater than DBP.")

    sv = edv - esv
    ef = sv / edv
    map_pressure = dbp + (sbp - dbp) / 3.0
    co = sv * hr
    rpp = hr * sbp
    lvsw = sv * map_pressure * 0.000133322

    rpp_component = safe_ratio(rpp, RPP_REF)
    lvsw_component = safe_ratio(lvsw, LVSW_REF)
    metabolic_demand = (
        0.7 * rpp_component
        + 0.3 * lvsw_component
    )

    mvo2 = MVO2_REST * metabolic_demand
    mvo2_total = mvo2 * MYOCARDIAL_MASS / 100.0
    chemical_power = mvo2_total * O2_ENERGY / 60.0
    mechanical_power = lvsw * hr / 60.0
    heat_production = max(
        chemical_power - mechanical_power,
        0.0,
    )

    atp_energy_production = (
        chemical_power
        * 60.0
        * ATP_COUPLING_EFFICIENCY
    )
    atp_production = (
        atp_energy_production
        / ATP_ENERGY
    )
    mechanical_energy = mechanical_power * 60.0
    atp_utilization = mechanical_energy / ATP_ENERGY

    atp_fraction = safe_ratio(
        atp_utilization,
        atp_production,
    )
    atp_balance = atp_production - atp_utilization

    return {
        "SV": sv,
        "EF": ef,
        "MAP": map_pressure,
        "CO": co,
        "RPP": rpp,
        "LVSW": lvsw,
        "RPP_component": rpp_component,
        "LVSW_component": lvsw_component,
        "Metabolic Demand Index": metabolic_demand,
        "MVO2": mvo2,
        "MVO2_total": mvo2_total,
        "O2 Consumption (mL/min)": mvo2_total,
        "Chemical Power (W)": chemical_power,
        "Mechanical Power (W)": mechanical_power,
        "Heat Production (W)": heat_production,
        "ATP Energy Production (J/min)": atp_energy_production,
        "ATP Production (mol/min)": atp_production,
        "Mechanical Energy Demand (J/min)": mechanical_energy,
        "ATP Utilization (mol/min)": atp_utilization,
        "ATP Utilization Fraction": atp_fraction,
        "ATP Balance (mol/min)": atp_balance,
    }


# ============================================================
# REACTION THERMODYNAMICS
# ============================================================

def compute_reaction_state(common, ph, spo2, fbs, chol, factor):
    hr = common["HR"]
    # Stored for clarity; common layer does not mutate HR.
    _ = hr

    atp_fraction = common["ATP Utilization Fraction"]
    atp_utilization = common["ATP Utilization (mol/min)"]
    atp_production = common["ATP Production (mol/min)"]
    atp_balance = common["ATP Balance (mol/min)"]

    chemical_power = common["Chemical Power (W)"]
    mechanical_energy = common["Mechanical Energy Demand (J/min)"]

    # These are the same pathway allocations used in the source DT.
    reaction_energy = {
        "Metabolism": chemical_power * 60.0,
        "ATP Utilization": mechanical_energy,
        "Ion Transport": mechanical_energy * atp_fraction,
        "Calcium Handling": mechanical_energy * atp_fraction,
        "Redox Metabolism": chemical_power * 60.0 * atp_fraction,
        "Nitric Oxide Metabolism": chemical_power * 60.0 * atp_fraction,
    }

    q = {}

    q["Metabolism"] = (
        ((fbs + 1.0) * chol)
        / (
            spo2
            * finite_positive(atp_balance)
        )
    ) * factor
    q["ATP Utilization"] = (
        safe_ratio(atp_utilization, atp_production)
        * factor
    )
    q["Ion Transport"] = (
        atp_fraction
        * (10.0 ** (7.4 - ph))
        * factor
    )
    calcium_hr_reference = common.get(
        "Calcium HR Reference",
        common["HR"],
    )
    q["Calcium Handling"] = (
        atp_fraction
        * (
            ph
            * common["HR"]
            / finite_positive(calcium_hr_reference)
        )
        * factor
    )
    q["Redox Metabolism"] = (
        atp_fraction
        * (spo2 / 100.0)
        * factor
    )
    q["Nitric Oxide Metabolism"] = (
        atp_fraction
        * (ph * spo2 / 100.0)
        * factor
    )

    dg0 = {
        "Metabolism": -2870000.0,
        "ATP Utilization": -30500.0,
        "Ion Transport": -50000.0,
        "Calcium Handling": -50000.0,
        "Redox Metabolism": -220000.0,
        "Nitric Oxide Metabolism": -100000.0,
    }

    dg = {
        reaction: dg0[reaction]
        + R_GAS
        * TEMPERATURE_K
        * safe_log_q(q[reaction])
        for reaction in REACTIONS
    }

    reaction_entropy_flow = {
        reaction: max(
            abs(dg[reaction])
            * reaction_energy[reaction]
            / max(abs(dg[reaction]), DG_FLOOR)
            / TEMPERATURE_K,
            SOFT_FLOOR,
        )
        for reaction in REACTIONS
    }

    return {
        "Q": q,
        "dG": dg,
        "Energy Demand (J/min)": reaction_energy,
        "Entropy Flow (J/K/min)": reaction_entropy_flow,
    }


# ============================================================
# BUILD FIXED TRAINING REFERENCES
# ============================================================

def build_training_references(workbook_path):
    """
    Reconstruct the fixed reaction reference medians and the
    disease-specific raw-DSES medians from the same CLEAN workbook
    used to construct the DSES training layer.
    """
    if not os.path.exists(workbook_path):
        raise FileNotFoundError(
            f"Reference workbook not found: {workbook_path}"
        )

    reaction_rows = {r: [] for r in REACTIONS}
    disease_reaction_medians = {
        disease: {} for disease in DISEASE_REACTION_MAP
    }

    xls = pd.ExcelFile(workbook_path)

    required = [
        "Heart Rate (HR)",
        "Systolic Blood Pressure (SBP)",
        "Diastolic Blood Pressure (DBP)",
        "End-Diastolic Volume (EDV)",
        "End-Systolic Volume (ESV)",
        "pH_a",
        "SpO2",
        "Fasting blood sugar (fbs)",
        "Serum cholesterol (chol)",
        "Literature Scenario Factor",
        "Associated Disease Condition",
    ]

    # First pass: reaction-specific derived values.
    for reaction in REACTIONS:
        if reaction not in xls.sheet_names:
            continue

        df = pd.read_excel(workbook_path, sheet_name=reaction)

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{reaction}: missing columns {missing}"
            )

        for col in required[:-1]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        hr = df["Heart Rate (HR)"]
        sbp = df["Systolic Blood Pressure (SBP)"]
        dbp = df["Diastolic Blood Pressure (DBP)"]
        edv = df["End-Diastolic Volume (EDV)"]
        esv = df["End-Systolic Volume (ESV)"]
        ph = df["pH_a"]
        spo2 = df["SpO2"]
        fbs = df["Fasting blood sugar (fbs)"]
        chol = df["Serum cholesterol (chol)"]
        factor = df["Literature Scenario Factor"].fillna(1.0)

        sv = edv - esv
        map_pressure = dbp + (sbp - dbp) / 3.0
        rpp = hr * sbp
        lvsw = sv * map_pressure * 0.000133322

        mdi = (
            0.7 * (rpp / RPP_REF)
            + 0.3 * (lvsw / LVSW_REF)
        )
        mvo2 = MVO2_REST * mdi
        mvo2_total = mvo2 * MYOCARDIAL_MASS / 100.0
        chemical_power = mvo2_total * O2_ENERGY / 60.0
        mechanical_power = lvsw * hr / 60.0
        mechanical_energy = mechanical_power * 60.0

        atp_production = (
            chemical_power
            * 60.0
            * ATP_COUPLING_EFFICIENCY
            / ATP_ENERGY
        )
        atp_utilization = mechanical_energy / ATP_ENERGY
        atp_fraction = (
            atp_utilization
            / np.maximum(atp_production, SOFT_FLOOR)
        )
        atp_balance = (
            atp_production - atp_utilization
        )

        if reaction == "Metabolism":
            q = (
                ((fbs + 1.0) * chol)
                / (
                    spo2
                    * np.maximum(atp_balance, SOFT_FLOOR)
                )
            ) * factor
            dg = -2870000.0 + R_GAS * TEMPERATURE_K * np.log(
                np.maximum(q, 1e-12)
            )
            energy = chemical_power * 60.0

        elif reaction == "ATP Utilization":
            q = (
                atp_utilization
                / np.maximum(atp_production, SOFT_FLOOR)
            ) * factor
            dg = -30500.0 + R_GAS * TEMPERATURE_K * np.log(
                np.maximum(q, 1e-12)
            )
            energy = mechanical_energy

        elif reaction == "Ion Transport":
            q = (
                atp_fraction
                * (10.0 ** (7.4 - ph))
                * factor
            )
            dg = -50000.0 + R_GAS * TEMPERATURE_K * np.log(
                np.maximum(q, 1e-12)
            )
            energy = mechanical_energy * atp_fraction

        elif reaction == "Calcium Handling":
            # The original DT used the sheet mean HR in this factor.
            mean_hr = float(hr.mean())
            calcium_factor = (
                ph
                * hr
                / max(mean_hr, SOFT_FLOOR)
            )
            q = (
                atp_fraction
                * calcium_factor
                * factor
            )
            dg = -50000.0 + R_GAS * TEMPERATURE_K * np.log(
                np.maximum(q, 1e-12)
            )
            energy = mechanical_energy * atp_fraction

        elif reaction == "Redox Metabolism":
            q = (
                atp_fraction
                * (spo2 / 100.0)
                * factor
            )
            dg = -220000.0 + R_GAS * TEMPERATURE_K * np.log(
                np.maximum(q, 1e-12)
            )
            energy = (
                chemical_power
                * 60.0
                * atp_fraction
            )

        else:
            q = (
                atp_fraction
                * (ph * spo2 / 100.0)
                * factor
            )
            dg = -100000.0 + R_GAS * TEMPERATURE_K * np.log(
                np.maximum(q, 1e-12)
            )
            energy = (
                chemical_power
                * 60.0
                * atp_fraction
            )

        entropy = np.maximum(
            np.abs(dg)
            * energy
            / np.maximum(np.abs(dg), DG_FLOOR)
            / TEMPERATURE_K,
            SOFT_FLOOR,
        )

        reaction_rows[reaction].append(
            pd.DataFrame(
                {
                    "mvo2": mvo2,
                    "mechanical_power": mechanical_power,
                    "dg_abs": np.abs(dg),
                    "atp_fraction": atp_fraction,
                    "entropy_flow": entropy,
                    "disease": (
                        df["Associated Disease Condition"]
                        .astype(str)
                        .str.strip()
                    ),
                    "factor": factor,
                }
            )
        )

    reaction_refs = {}

    # Reaction reference medians.
    for reaction in REACTIONS:
        if not reaction_rows[reaction]:
            raise ValueError(
                f"No reference rows found for reaction: {reaction}"
            )

        d = pd.concat(
            reaction_rows[reaction],
            ignore_index=True,
        )

        # Preserve the original DT convention for the Calcium
        # Handling factor: normalize HR by the mean HR of that
        # reaction sheet.
        try:
            source_df = pd.read_excel(
                workbook_path,
                sheet_name=reaction,
                usecols=["Heart Rate (HR)"],
            )
            source_hr = pd.to_numeric(
                source_df["Heart Rate (HR)"],
                errors="coerce",
            ).dropna()
            d.attrs["mean_hr"] = (
                float(source_hr.mean())
                if len(source_hr)
                else 75.0
            )
        except Exception:
            d.attrs["mean_hr"] = 75.0

        reaction_refs[reaction] = {
            "mvo2_median": float(d["mvo2"].median()),
            "mechanical_power_median": float(
                d["mechanical_power"].median()
            ),
            "dg_abs_median": float(
                d["dg_abs"].median()
            ),
            "atp_fraction_median": float(
                d["atp_fraction"].median()
            ),
            "entropy_flow_median": float(
                d["entropy_flow"].median()
            ),
            "calcium_hr_reference": float(
                d.attrs.get("mean_hr", 75.0)
            ),
        }

        # Compute the exact reaction-level DSES component.
        d["metabolic_stress"] = (
            d["mvo2"]
            / finite_positive(
                reaction_refs[reaction]["mvo2_median"]
            )
        )
        d["mechanical_stress"] = (
            d["mechanical_power"]
            / finite_positive(
                reaction_refs[reaction][
                    "mechanical_power_median"
                ]
            )
        )
        d["thermodynamic_stress"] = (
            d["dg_abs"]
            / finite_positive(
                reaction_refs[reaction]["dg_abs_median"]
            )
        )
        d["atp_stress"] = (
            d["atp_fraction"]
            / finite_positive(
                reaction_refs[reaction][
                    "atp_fraction_median"
                ]
            )
        )
        d["entropy_stress"] = (
            d["entropy_flow"]
            / finite_positive(
                reaction_refs[reaction][
                    "entropy_flow_median"
                ]
            )
        )
        d["all_stress_mean"] = d[
            [
                "metabolic_stress",
                "mechanical_stress",
                "thermodynamic_stress",
                "atp_stress",
                "entropy_stress",
            ]
        ].mean(axis=1)
        d["entropy_ratio"] = d[
            "entropy_stress"
        ]
        d["component"] = (
            d["entropy_ratio"]
            * d["all_stress_mean"]
            * d["factor"]
        )

        # Store reaction-level disease anchors for only the
        # disease/reaction combinations that actually belong together.
        for disease in DISEASE_REACTION_MAP:
            if reaction not in DISEASE_REACTION_MAP[disease]:
                continue

            mask = (
                d["disease"].map(normalize_disease_name)
                == normalize_disease_name(disease)
            )
            vals = pd.to_numeric(
                d.loc[mask, "component"],
                errors="coerce",
            ).dropna()

            if len(vals):
                disease_reaction_medians[disease][reaction] = float(
                    vals.median()
                )

    # Disease-specific raw-DSES anchors.
    # Only the reactions mapped to each disease are included.
    disease_raw_medians = {}

    for disease, reaction_map in DISEASE_REACTION_MAP.items():
        values = [
            disease_reaction_medians[disease][reaction]
            for reaction in reaction_map
            if reaction in disease_reaction_medians[disease]
        ]

        if values:
            disease_raw_medians[disease] = float(
                np.mean(values)
            )

    result = {
        "temperature_K": TEMPERATURE_K,
        "reaction_references": reaction_refs,
        "disease_raw_medians": disease_raw_medians,
        "disease_signature_scalars": DISEASE_SIGNATURE_SCALARS,
        "mapping": DISEASE_REACTION_MAP,
        "factors": DISEASE_REACTION_FACTORS,
    }

    return result


def load_reference_bundle():
    # Prefer a cached fixed reference bundle.
    if os.path.exists(REFERENCE_CACHE):
        try:
            with open(REFERENCE_CACHE, "rb") as fh:
                bundle = pickle.load(fh)
            return bundle
        except Exception:
            pass

    bundle = build_training_references(
        REFERENCE_WORKBOOK
    )

    with open(REFERENCE_CACHE, "wb") as fh:
        pickle.dump(bundle, fh)

    return bundle


# ============================================================
# NAME NORMALIZATION
# ============================================================

def normalize_disease_name(name):
    s = str(name).strip().lower()
    s = s.replace(
        "catecholaminergic polymorphic ventricular tachycardia (cpvt)",
        "cpvt",
    )
    s = s.replace("catecholaminergic polymorphic ventricular tachycardia", "cpvt")
    return re.sub(r"\s+", " ", s)


# ============================================================
# PATIENT DSES ENGINE
# ============================================================

def compute_patient_dses(
    hr,
    sbp,
    dbp,
    edv,
    esv,
    ph,
    spo2,
    fbs,
    chol,
    reference_bundle,
):
    # Common layer.
    common = compute_common_layer(
        hr=hr,
        sbp=sbp,
        dbp=dbp,
        edv=edv,
        esv=esv,
    )
    common["HR"] = hr

    refs = reference_bundle[
        "reaction_references"
    ]

    common["Calcium HR Reference"] = refs[
        "Calcium Handling"
    ].get("calcium_hr_reference", common["HR"])

    disease_raw_medians = reference_bundle[
        "disease_raw_medians"
    ]

    signatures = reference_bundle[
        "disease_signature_scalars"
    ]

    factors = reference_bundle[
        "factors"
    ]

    reaction_output = {}

    # Use the disease-independent "factor = 1" reaction state.
    # Disease conditioning is applied only when building a disease
    # signature below.
    base_reactions = {}

    for reaction in REACTIONS:
        state = compute_reaction_state(
            common,
            ph,
            spo2,
            fbs,
            chol,
            factor=1.0,
        )
        # One call creates all reactions; keep only this reaction.
        base_reactions[reaction] = {
            "dG": state["dG"][reaction],
            "Q": state["Q"][reaction],
            "energy": state["Energy Demand (J/min)"][reaction],
            "entropy": state["Entropy Flow (J/K/min)"][reaction],
        }

    def reaction_component(
        reaction,
        disease_factor,
    ):
        # Recompute only the reaction with its disease factor so
        # literature conditioning is applied consistently.
        state = compute_reaction_state(
            common,
            ph,
            spo2,
            fbs,
            chol,
            factor=disease_factor,
        )

        dg = state["dG"][reaction]
        entropy = state[
            "Entropy Flow (J/K/min)"
        ][reaction]

        ref = refs[reaction]

        metabolic_stress = safe_ratio(
            common["MVO2"],
            ref["mvo2_median"],
        )
        mechanical_stress = safe_ratio(
            common["Mechanical Power (W)"],
            ref["mechanical_power_median"],
        )
        thermodynamic_stress = safe_ratio(
            abs(dg),
            ref["dg_abs_median"],
        )
        atp_stress = safe_ratio(
            common["ATP Utilization Fraction"],
            ref["atp_fraction_median"],
        )
        entropy_stress = safe_ratio(
            entropy,
            ref["entropy_flow_median"],
        )

        all_stress_mean = (
            metabolic_stress
            + mechanical_stress
            + thermodynamic_stress
            + atp_stress
            + entropy_stress
        ) / 5.0

        component = (
            entropy_stress
            * all_stress_mean
            * max(disease_factor, SOFT_FLOOR)
        )

        return {
            "Q": state["Q"][reaction],
            "dG (J/mol)": dg,
            "Entropy Flow (J/K/min)": entropy,
            "Metabolic Stress": metabolic_stress,
            "Mechanical Stress": mechanical_stress,
            "Thermodynamic Stress": thermodynamic_stress,
            "ATP Stress": atp_stress,
            "Entropy Stress": entropy_stress,
            "All Stress Mean": all_stress_mean,
            "Raw Reaction DSES": max(
                component,
                SOFT_FLOOR,
            ),
            "Literature Scenario Factor": disease_factor,
        }

    # Disease-specific patient signatures.
    disease_rows = []

    for disease, reactions in DISEASE_REACTION_MAP.items():
        reaction_values = []
        per_reaction = {}

        for reaction in reactions:
            disease_factor = float(
                factors.get(disease, {}).get(
                    reaction,
                    1.0,
                )
            )

            detail = reaction_component(
                reaction,
                disease_factor,
            )
            per_reaction[reaction] = detail
            reaction_values.append(
                detail["Raw Reaction DSES"]
            )

        raw_dses = (
            float(np.mean(reaction_values))
            if reaction_values
            else SOFT_FLOOR
        )

        training_anchor = disease_raw_medians.get(
            disease,
            np.nan,
        )

        signature = signatures.get(
            disease,
            np.nan,
        )

        if (
            np.isfinite(training_anchor)
            and training_anchor > SOFT_FLOOR
            and np.isfinite(signature)
        ):
            # NEW CALIBRATION:
            # patient / training-anchor * disease signature.
            calibrated = (
                signature
                * (
                    raw_dses
                    / training_anchor
                )
            )
        else:
            calibrated = raw_dses

        calibrated = float(
            np.clip(
                calibrated,
                1.0,
                100.0,
            )
        )

        disease_rows.append(
            {
                "Disease": disease,
                "Mapped Reactions": ", ".join(reactions),
                "Raw DSES": raw_dses,
                "Training Raw Anchor": training_anchor,
                "Disease Signature": signature,
                "Patient DSES (1-100)": calibrated,
            }
        )

        reaction_output[disease] = per_reaction

    disease_table = (
        pd.DataFrame(disease_rows)
        .sort_values(
            "Patient DSES (1-100)",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # Full-system entropy diagnostic, matching the existing DT's
    # energy/Gibbs balance calculation.
    reaction0 = compute_reaction_state(
        common,
        ph,
        spo2,
        fbs,
        chol,
        factor=1.0,
    )

    dn_metabolism = (
        common["Chemical Power (W)"] * 60.0
        / max(
            abs(reaction0["dG"]["Metabolism"]),
            DG_FLOOR,
        )
    )
    dn_atp = common[
        "ATP Utilization (mol/min)"
    ]
    dn_ion = (
        common["Mechanical Power (W)"] * 60.0
        * common["ATP Utilization Fraction"]
        / max(
            abs(reaction0["dG"]["Ion Transport"]),
            DG_FLOOR,
        )
    )
    dn_calcium = (
        common["Mechanical Power (W)"] * 60.0
        * common["ATP Utilization Fraction"]
        / max(
            abs(reaction0["dG"]["Calcium Handling"]),
            DG_FLOOR,
        )
    )
    dn_redox = (
        common["Chemical Power (W)"] * 60.0
        * common["ATP Utilization Fraction"]
        / max(
            abs(reaction0["dG"]["Redox Metabolism"]),
            DG_FLOOR,
        )
    )
    dn_no = (
        common["Chemical Power (W)"] * 60.0
        * common["ATP Utilization Fraction"]
        / max(
            abs(reaction0["dG"]["Nitric Oxide Metabolism"]),
            DG_FLOOR,
        )
    )

    gibbs_sum = (
        reaction0["dG"]["Metabolism"]
        * dn_metabolism
        + reaction0["dG"]["ATP Utilization"]
        * dn_atp
        + reaction0["dG"]["Ion Transport"]
        * dn_ion
        + reaction0["dG"]["Calcium Handling"]
        * dn_calcium
        + reaction0["dG"]["Redox Metabolism"]
        * dn_redox
        + reaction0["dG"]["Nitric Oxide Metabolism"]
        * dn_no
    )

    entropy_flow_full = (
        common["Chemical Power (W)"] * 60.0
        + common["Mechanical Power (W)"] * 60.0
        - gibbs_sum
    ) / TEMPERATURE_K

    # Reaction summary for display.
    top_disease = disease_table.iloc[0].to_dict()

    result = {
        "common": common,
        "reaction_details": reaction_output,
        "disease_table": disease_table,
        "full_entropy_flow": float(
            max(entropy_flow_full, SOFT_FLOOR)
        ),
        "top_disease": top_disease,
    }

    return result



DEFAULT_REFERENCE_WORKBOOK = (
    Path(__file__).resolve().parent /
    "Cardiac_Disease_Synchronized_Dataset_Patient_1_"
    "Literature_Conditioned_CLEAN (4).xlsx"
)
DEFAULT_CACHE = Path(__file__).resolve().parent / "dt_reference_values.pkl"

def build_or_load_references(workbook_path=None, cache_path=None):
    workbook_path = str(workbook_path or DEFAULT_REFERENCE_WORKBOOK)
    cache_path = str(cache_path or DEFAULT_CACHE)
    if Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    bundle = build_training_references(workbook_path)
    with open(cache_path, "wb") as f:
        pickle.dump(bundle, f)
    return bundle

def compute_patient_state(**patient):
    return compute_common_layer(
        hr=patient["hr"], sbp=patient["sbp"], dbp=patient["dbp"],
        edv=patient["edv"], esv=patient["esv"]
    )

def compute_patient_dses_scores(patient, reference_bundle):
    return compute_patient_dses(
        hr=patient["hr"], sbp=patient["sbp"], dbp=patient["dbp"],
        edv=patient["edv"], esv=patient["esv"], ph=patient["ph"],
        spo2=patient["spo2"], fbs=patient["fbs"], chol=patient["chol"],
        reference_bundle=reference_bundle
    )
