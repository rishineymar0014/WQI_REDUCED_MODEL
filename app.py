
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from wqi_reduced_model import synthetic_data, compute_wqi, wqi_class

st.set_page_config(page_title="WQI Predictor", page_icon="💧")

st.title("💧 Reduced-Parameter WQI Predictor")
st.caption(
    "Can a few cheap measurements predict the Water Quality Index? "
    "Models are trained on simulated river data."
)

SETS = {
    "3 parameters (pH, BOD, Total Coliform)": [
        "ph", "bod", "total_coliform"
    ],
    "4 parameters (+ Dissolved Oxygen)": [
        "ph", "do", "bod", "total_coliform"
    ],
    "7 parameters (all)": [
        "temp", "ph", "do", "bod", "conductivity",
        "nitrate", "total_coliform"
    ],
}

@st.cache_resource
def train():
    df = synthetic_data(n_st=40, n_per=150)
    df["wqi"] = compute_wqi(df)

    return {
        name: RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=0
        ).fit(df[columns], df["wqi"])
        for name, columns in SETS.items()
    }

models = train()

choice = st.radio("Choose model", list(SETS))
cols = SETS[choice]

ranges = {
    "temp": ("Temperature (°C)", 10.0, 40.0, 25.0),
    "ph": ("pH", 6.0, 9.0, 7.5),
    "do": ("Dissolved Oxygen (mg/L)", 0.5, 12.0, 6.0),
    "bod": ("BOD (mg/L)", 0.5, 30.0, 3.0),
    "conductivity": ("Conductivity (µmho/cm)", 50.0, 2500.0, 500.0),
    "nitrate": ("Nitrate (mg/L)", 0.05, 30.0, 2.0),
    "total_coliform": (
        "Total Coliform (MPN/100ml)", 5.0, 100000.0, 500.0
    ),
}

vals = {}

for c in cols:
    label, lo, hi, default = ranges[c]
    vals[c] = st.number_input(
        label,
        min_value=lo,
        max_value=hi,
        value=default
    )

if st.button("Predict WQI"):
    input_df = pd.DataFrame([vals], columns=cols)
    w = float(models[choice].predict(input_df)[0])

    st.metric("Predicted WQI", f"{w:.1f}")
    st.success(
        f"Water quality class: **{wqi_class(pd.Series([w]))[0]}**"
    )

st.info(
    "Lower WQI = cleaner water. Demo classes: Excellent <25, Good <50, "
    "Poor <75, Very poor <100, Unfit ≥100."
)

st.warning(
    "This version uses simulated data. Predictions are for demonstration "
    "only and are not a validated assessment of real river-water safety."
)
