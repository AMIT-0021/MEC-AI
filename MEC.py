import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Page configuration
st.set_page_config(
    page_title="MEC-AI", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.title("MEC-AI: Hydrogen Production & Wastewater Monitoring")
st.markdown("Predict hydrogen production and find optimal operating conditions using machine learning.")

# --------------------------------------------------
# 1. SAMPLE MEC DATA
# --------------------------------------------------
data = {
    "pH": [
        6.2, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2,
        6.3, 6.5, 6.7, 6.9, 7.1, 6.6, 6.8, 7.0, 6.4, 6.9
    ],
    "temperature": [
        28, 30, 32, 33, 34, 35, 36, 35, 34, 32,
        29, 31, 33, 36, 35, 34, 37, 36, 30, 35
    ],
    "voltage": [
        0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.72, 0.7, 0.68, 0.6,
        0.35, 0.5, 0.65, 0.72, 0.7, 0.6, 0.75, 0.72, 0.4, 0.7
    ],
    "COD": [
        700, 680, 650, 620, 600, 580, 560, 550, 570, 600,
        690, 660, 610, 550, 560, 630, 540, 550, 675, 560
    ],
    "current": [
        0.20, 0.25, 0.30, 0.36, 0.40, 0.45, 0.48, 0.46, 0.43, 0.38,
        0.22, 0.28, 0.39, 0.49, 0.46, 0.35, 0.50, 0.48, 0.26, 0.46
    ],
    "H2": [
        65, 78, 92, 108, 119, 128, 135, 132, 124, 110,
        70, 88, 115, 137, 131, 105, 140, 136, 80, 133
    ]
}

df = pd.DataFrame(data)

# Show raw dataset option
with st.expander("View Training Dataset"):
    st.dataframe(df)

# --------------------------------------------------
# 2. INPUTS AND OUTPUT
# --------------------------------------------------
X = df[["pH", "temperature", "voltage", "COD", "current"]]
y = df["H2"]

# --------------------------------------------------
# 3. SPLIT DATA & TRAIN MODEL
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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

# Display Model Metrics
st.subheader("Model Performance")
col_m1, col_m2 = st.columns(2)
col_m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f} mL")
col_m2.metric("R² Score", f"{r2:.2f}")

st.divider()

# --------------------------------------------------
# 4. INTERACTIVE SIDEBAR PREDICTION
# --------------------------------------------------
st.sidebar.header("Input Parameters")

user_ph = st.sidebar.slider("pH", 6.0, 7.5, 6.8, 0.1)
user_temp = st.sidebar.slider("Temperature (°C)", 25, 40, 35, 1)
user_voltage = st.sidebar.slider("Voltage (V)", 0.1, 1.0, 0.7, 0.05)
user_cod = st.sidebar.number_input("COD (mg/L)", 400, 800, 550, 10)
user_current = st.sidebar.slider("Current (A)", 0.1, 0.8, 0.46, 0.01)

new_condition = pd.DataFrame({
    "pH": [user_ph],
    "temperature": [user_temp],
    "voltage": [user_voltage],
    "COD": [user_cod],
    "current": [user_current]
})

predicted_h2 = model.predict(new_condition)[0]

# Display Predictions & Optimization in Main Screen
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Prediction for Current Input")
    st.metric("Predicted H₂ Production", f"{predicted_h2:.2f} mL")
    st.write("**Current Parameters:**")
    st.json(new_condition.iloc[0].to_dict())

# --------------------------------------------------
# 5. OPTIMAL CONDITION SEARCH
# --------------------------------------------------
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
        prediction = model.predict(condition)[0]
        if prediction > best_h2:
            best_h2 = prediction
            best_condition = condition

with col_right:
    st.subheader("Recommended Optimal Conditions")
    st.metric("Max Predicted H₂", f"{best_h2:.2f} mL")
    st.write("**Optimal Operating Values (at Temp=35°C, COD=550, Current=0.45A):**")
    st.json(best_condition.iloc[0].to_dict())

st.divider()

# --------------------------------------------------
# 6. GRAPH
# --------------------------------------------------
st.subheader("Actual vs Predicted Hydrogen Production")

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(y_test, predictions, color="blue", alpha=0.7)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=1.5)
ax.set_xlabel("Actual H2 Production (mL)")
ax.set_ylabel("Predicted H2 Production (mL)")
ax.grid(True)

st.pyplot(fig)
