import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(
    page_title="MEC-AI Platform",
    page_icon="⚡",
    layout="wide"
)

# Load data & train model
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

    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return df, model, y_test, predictions, mae, r2

df, model, y_test, predictions, mae, r2 = load_and_train_model()

# Title
st.title("⚡ MEC-AI: Hydrogen Intelligence Platform")

# Sidebar
st.sidebar.header("Reactor Controls")
user_ph = st.sidebar.slider("pH Level", 6.0, 7.5, 6.8, 0.1)
user_temp = st.sidebar.slider("Reactor Temp (°C)", 25, 40, 35, 1)
user_voltage = st.sidebar.slider("Applied Voltage (V)", 0.1, 1.0, 0.7, 0.05)
user_cod = st.sidebar.number_input("Chemical Oxygen Demand (mg/L)", 400, 800, 550, 10)
user_current = st.sidebar.slider("Current Intensity (A)", 0.1, 0.8, 0.46, 0.01)

new_condition = pd.DataFrame({
    "pH": [user_ph], "temperature": [user_temp], "voltage": [user_voltage],
    "COD": [user_cod], "current": [user_current]
})
predicted_h2 = model.predict(new_condition)[0]

# Display
col1, col2 = st.columns(2)
col1.metric("Predicted H2 Yield", f"{predicted_h2:.2f} mL")
col2.metric("Model R² Score", f"{r2:.2f}")

st.subheader("Predicted Output Meter")
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=predicted_h2,
    title={'text': "Predicted H2 (mL)"},
    gauge={'axis': {'range': [0, 200]}}
))
st.plotly_chart(fig_gauge, use_container_width=True)
