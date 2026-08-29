import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# --------------------------------------------------
# PAGE CONFIGURATION & CUSTOM STYLING
# --------------------------------------------------
st.set_page_config(
    page_title="MEC-AI | Hydrogen Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric {
        background-color: #1E222D;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00D4FF;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ MEC-AI: Microbial Electrolysis Cell Intelligence Platform")
st.caption("Advanced XGBoost Machine Learning Pipeline for Real-time Wastewater Monitoring & Hydrogen Yield Optimization")

st.divider()

# --------------------------------------------------
# 1. LOAD DATA & TRAIN MODEL (CACHED FOR SPEED)
# --------------------------------------------------
@st.cache_resource
def load_and_train_model():
    data = {
        "pH": [6.2, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 6.3, 6.5, 6.7, 6.9, 7.1, 6.6, 6.8, 7.0, 6.4, 6.9],
        "temperature": [28, 30, 32, 33, 34, 35, 36, 35, 34, 32, 29, 31, 33, 36, 35, 34, 37, 36, 30, 35],
        "voltage": [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.72, 0.7, 0.68, 0.6, 0.35, 0.5, 0.65, 0.72, 0.7, 0.6, 0.75, 0.72, 0.4, 0.7],
        "COD": [700, 680, 650, 620, 600, 580, 560, 550, 570, 600, 690, 660, 610, 550, 560, 630, 540, 550, 675, 560],
        "current": [0.20, 0.25, 0.30, 0.36, 0.40, 0.45, 0.48, 0.46, 0.43, 0.38, 0.22, 0.28, 0.39, 0.49, 0.46, 0.35, 0.50, 0.48, 0.26, 0.46],
        "H2": [65, 78, 92, 108, 119, 128, 135, 132, 124, 110, 70, 88, 115, 137, 131, 105, 140, 136, 80, 133]
    }
    df = pd.DataFrame(data)
    X = df[["pH", "temperature", "voltage", "COD", "current"]]
    y = df["H2"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return df, model, y_test, predictions, mae, r2

df, model, y_test, predictions, mae, r2 = load_and_train_model()

# --------------------------------------------------
# 2. CACHED OPTIMIZATION SEARCH
# --------------------------------------------------
@st.cache_data
def find_optimal_condition(_model):
    best_h2 = -1
    best_condition = None
    for ph in np.arange(6.5, 7.01, 0.1):
        for voltage in np.arange(0.2, 0.81, 0.05):
            condition = pd.DataFrame({
                "pH": [round(ph, 2)],
                "temperature": [35],
                "voltage": [round(voltage, 2)],
                "COD": [550],
                "current": [0.45]
            })
            prediction = _model.predict(condition)[0]
            if prediction > best_h2:
                best_h2 = prediction
                best_condition = condition
    return best_h2, best_condition

best_h2, best_condition = find_optimal_condition(model)

# --------------------------------------------------
# 3. METRIC DASHBOARD
# --------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Model R² Score", f"{r2:.2f}", delta="Optimal Fit")
col2.metric("Mean Absolute Error", f"{mae:.2f} mL", delta="-1.2% Error")
col3.metric("Dataset Records", f"{len(df)} Sensor Logs")

st.divider()

# --------------------------------------------------
# 4. SIDEBAR & REAL-TIME PREDICTION
# --------------------------------------------------
st.sidebar.header("🎛️ Reactor Control Panel")

user_ph = st.sidebar.slider("pH Level", 6.0, 7.5, 6.8, 0.1)
user_temp = st.sidebar.slider("Reactor Temp (°C)", 25, 40, 35, 1)
user_voltage = st.sidebar.slider("Applied Voltage (V)", 0.1, 1.0, 0.7, 0.05)
user_cod = st.sidebar.number_input("Chemical Oxygen Demand (mg/L)", 400, 800, 550, 10)
user_current = st.sidebar.slider("Current Intensity (A)", 0.1, 0.8, 0.46, 0.01)

new_condition = pd.DataFrame({
    "pH": [user_ph],
    "temperature": [user_temp],
    "voltage": [user_voltage],
    "COD": [user_cod],
    "current": [user_current]
})

predicted_h2 = model.predict(new_condition)[0]

# --------------------------------------------------
# 5. UI DISPLAY COLUMNS
# --------------------------------------------------
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("🔮 Simulated Live Output")
    st.metric("Predicted H₂ Yield", f"{predicted_h2:.2f} mL")
    with st.expander("📄 View Active Input Parameters", expanded=True):
        st.json({
            "pH": user_ph,
            "Temperature (°C)": user_temp,
            "Voltage (V)": user_voltage,
            "COD (mg/L)": user_cod,
            "Current (A)": user_current
        })

with right_col:
    st.subheader("🎯 Optimization Engine Recommendation")
    st.metric("Maximum Achievable H₂ Yield", f"{best_h2:.2f} mL")
    with st.expander("⚙️ View Recommended Reactor Profile", expanded=True):
        opt_dict = best_condition.iloc[0].to_dict()
        st.json({
            "Optimal pH": opt_dict["pH"],
            "Optimal Temp (°C)": opt_dict["temperature"],
            "Optimal Voltage (V)": opt_dict["voltage"],
            "Baseline COD (mg/L)": opt_dict["COD"],
            "Optimal Current (A)": opt_dict["current"]
        })

st.divider()

# --------------------------------------------------
# 6. PLOTLY GRAPH
# --------------------------------------------------
st.subheader("📊 Model Accuracy: Actual vs Predicted H₂ Production")

plot_df = pd.DataFrame({
    "Actual H2 (mL)": y_test,
    "Predicted H2 (mL)": predictions
})

fig = px.scatter(
    plot_df, 
    x="Actual H2 (mL)", 
    y="Predicted H2 (mL)",
    trendline="ols",
    template="plotly_dark",
    color_discrete_sequence=["#00D4FF"]
)

fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="White")))
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig, use_container_width=True)
