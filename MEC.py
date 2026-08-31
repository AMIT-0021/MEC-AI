import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MEC-AI Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

[data-testid="stMetricValue"] {
    font-size: 30px;
}

.status-box {
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    background-color: rgba(100,100,100,0.12);
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TRAINING DATA
# =========================================================

DATA = {
    "pH": [
        6.2, 6.4, 6.5, 6.6, 6.7,
        6.8, 6.9, 7.0, 7.1, 7.2,
        6.3, 6.5, 6.7, 6.9, 7.1,
        6.6, 6.8, 7.0, 6.4, 6.9
    ],

    "temperature": [
        28, 30, 32, 33, 34,
        35, 36, 35, 34, 32,
        29, 31, 33, 36, 35,
        34, 37, 36, 30, 35
    ],

    "voltage": [
        0.30, 0.40, 0.50, 0.60, 0.65,
        0.70, 0.72, 0.70, 0.68, 0.60,
        0.35, 0.50, 0.65, 0.72, 0.70,
        0.60, 0.75, 0.72, 0.40, 0.70
    ],

    "COD": [
        700, 680, 650, 620, 600,
        580, 560, 550, 570, 600,
        690, 660, 610, 550, 560,
        630, 540, 550, 675, 560
    ],

    "current": [
        0.20, 0.25, 0.30, 0.36, 0.40,
        0.45, 0.48, 0.46, 0.43, 0.38,
        0.22, 0.28, 0.39, 0.49, 0.46,
        0.35, 0.50, 0.48, 0.26, 0.46
    ],

    "H2": [
        65, 78, 92, 108, 119,
        128, 135, 132, 124, 110,
        70, 88, 115, 137, 131,
        105, 140, 136, 80, 133
    ]
}


# =========================================================
# TRAIN MODEL
# =========================================================

@st.cache_resource
def train_model():

    df = pd.DataFrame(DATA)

    features = [
        "pH",
        "temperature",
        "voltage",
        "COD",
        "current"
    ]

    X = df[features]
    y = df["H2"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    validation_df = pd.DataFrame({
        "Actual H2 (mL)": y_test.values,
        "Predicted H2 (mL)": predictions
    })

    return df, model, validation_df, mae, r2


df, model, validation_df, mae, r2 = train_model()


# =========================================================
# HEADER
# =========================================================

st.title("⚡ MEC-AI")
st.subheader("Microbial Electrolysis Cell Intelligence Platform")

st.markdown(
    "### 🧠 AI-powered hydrogen prediction, reactor monitoring and optimization"
)

st.markdown("""
<div class="info-box">

<b>Current Mode:</b> Simulation / AI Demonstration

The values entered below are currently used as model inputs.
For the final physical prototype, these values should come from
real ESP32 sensors.

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR - REACTOR INPUTS
# =========================================================

st.sidebar.title("🎛️ Reactor Controls")

st.sidebar.markdown("### Reactor Conditions")

user_ph = st.sidebar.slider(
    "pH Level",
    min_value=6.0,
    max_value=7.5,
    value=6.8,
    step=0.1
)

user_temp = st.sidebar.slider(
    "Temperature (°C)",
    min_value=25,
    max_value=40,
    value=35,
    step=1
)

user_voltage = st.sidebar.slider(
    "Applied Voltage (V)",
    min_value=0.1,
    max_value=1.0,
    value=0.70,
    step=0.05
)

user_cod = st.sidebar.number_input(
    "COD (mg/L)",
    min_value=400,
    max_value=800,
    value=550,
    step=10
)

user_current = st.sidebar.slider(
    "Current (A)",
    min_value=0.1,
    max_value=0.8,
    value=0.46,
    step=0.01
)


# =========================================================
# AI PREDICTION
# =========================================================

input_data = pd.DataFrame({
    "pH": [user_ph],
    "temperature": [user_temp],
    "voltage": [user_voltage],
    "COD": [user_cod],
    "current": [user_current]
})

predicted_h2 = float(model.predict(input_data)[0])


# =========================================================
# REACTOR STATUS
# =========================================================

def reactor_status(ph, temp, voltage, current):

    score = 0

    if 6.5 <= ph <= 7.0:
        score += 1

    if 30 <= temp <= 37:
        score += 1

    if 0.5 <= voltage <= 0.8:
        score += 1

    if 0.35 <= current <= 0.55:
        score += 1

    if score >= 4:
        return "🟢 OPTIMAL"

    elif score >= 2:
        return "🟡 MODERATE"

    else:
        return "🔴 NEEDS ATTENTION"


status = reactor_status(
    user_ph,
    user_temp,
    user_voltage,
    user_current
)


# =========================================================
# TOP METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Predicted H₂",
    f"{predicted_h2:.2f} mL"
)

col2.metric(
    "Model R²",
    f"{r2:.2f}"
)

col3.metric(
    "Mean Absolute Error",
    f"{mae:.2f} mL"
)

col4.metric(
    "Reactor Status",
    status
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Live Prediction",
    "🎯 AI Optimization",
    "📈 Model Validation",
    "🧠 AI Insights",
    "📄 Report"
])


# =========================================================
# TAB 1 - LIVE PREDICTION
# =========================================================

with tab1:

    st.header("🔮 Live H₂ Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Current Reactor Inputs")

        input_display = pd.DataFrame({
            "Parameter": [
                "pH",
                "Temperature",
                "Voltage",
                "COD",
                "Current"
            ],

            "Value": [
                user_ph,
                f"{user_temp} °C",
                f"{user_voltage:.2f} V",
                f"{user_cod} mg/L",
                f"{user_current:.2f} A"
            ]
        })

        st.table(input_display)

    with col2:

        st.subheader("Predicted Hydrogen")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=predicted_h2,
                number={
                    "suffix": " mL"
                },
                title={
                    "text": "Predicted H₂ Yield"
                },
                gauge={
                    "axis": {
                        "range": [0, 200]
                    },
                    "bar": {
                        "thickness": 0.7
                    },
                    "steps": [
                        {
                            "range": [0, 70],
                        },
                        {
                            "range": [70, 130],
                        },
                        {
                            "range": [130, 200],
                        }
                    ]
                }
            )
        )

        gauge.update_layout(
            height=350,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

    st.success(
        f"AI prediction: approximately **{predicted_h2:.2f} mL H₂** "
        f"under the current simulated conditions."
    )


# =========================================================
# TAB 2 - AI OPTIMIZATION
# =========================================================

with tab2:

    st.header("🎯 AI Reactor Optimization")

    st.write(
        "The system searches different operating conditions and "
        "identifies the combination that gives the highest predicted H₂ output."
    )

    if st.button("🚀 Find Best Operating Condition"):

        progress = st.progress(0)

        best_h2 = -np.inf
        best_values = None

        total = 10 * 5 * 5

        counter = 0

        for ph in np.arange(6.3, 7.2, 0.1):

            for temp in np.arange(30, 38, 2):

                for voltage in np.arange(0.5, 0.81, 0.05):

                    # Estimate current based on available training range
                    current = 0.46

                    test_input = pd.DataFrame({
                        "pH": [round(ph, 2)],
                        "temperature": [temp],
                        "voltage": [round(voltage, 2)],
                        "COD": [user_cod],
                        "current": [current]
                    })

                    prediction = float(
                        model.predict(test_input)[0]
                    )

                    if prediction > best_h2:

                        best_h2 = prediction

                        best_values = {
                            "pH": round(ph, 2),
                            "Temperature": temp,
                            "Voltage": round(voltage, 2),
                            "COD": user_cod,
                            "Current": current
                        }

                    counter += 1

                    progress.progress(
                        min(counter / total, 1.0)
                    )

        st.success("✅ Optimization completed!")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Best Predicted H₂",
                f"{best_h2:.2f} mL"
            )

        with col2:

            st.metric(
                "Improvement",
                f"{best_h2 - predicted_h2:.2f} mL"
            )

        st.subheader("🎯 Recommended Conditions")

        recommendation = pd.DataFrame({
            "Parameter": [
                "pH",
                "Temperature",
                "Applied Voltage",
                "COD",
                "Current"
            ],

            "Recommended Value": [
                best_values["pH"],
                f'{best_values["Temperature"]} °C',
                f'{best_values["Voltage"]} V',
                f'{best_values["COD"]} mg/L',
                f'{best_values["Current"]} A'
            ]
        })

        st.table(recommendation)

        st.info(
            "⚠️ These are AI/model recommendations, not experimentally "
            "validated operating instructions. Verify safe operating limits "
            "with your physical MEC before applying changes."
        )


# =========================================================
# TAB 3 - MODEL VALIDATION
# =========================================================

with tab3:

    st.header("📈 Model Validation")

    st.write(
        "Comparison between actual H₂ values and values predicted by the model "
        "on the held-out test set."
    )

    fig = px.scatter(
        validation_df,
        x="Actual H2 (mL)",
        y="Predicted H2 (mL)",
        title="Actual vs Predicted H₂"
    )

    # Perfect prediction line
    min_value = min(
        validation_df["Actual H2 (mL)"].min(),
        validation_df["Predicted H2 (mL)"].min()
    )

    max_value = max(
        validation_df["Actual H2 (mL)"].max(),
        validation_df["Predicted H2 (mL)"].max()
    )

    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="Perfect Prediction"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "R² Score",
        f"{r2:.3f}"
    )

    col2.metric(
        "MAE",
        f"{mae:.2f} mL"
    )

    st.dataframe(
        validation_df,
        use_container_width=True
    )

    st.warning(
        "Important: this demonstration uses only 20 training records. "
        "The model should be retrained using substantially more real MEC "
        "experimental measurements before making scientific or engineering claims."
    )


# =========================================================
# TAB 4 - AI INSIGHTS
# =========================================================

with tab4:

    st.header("🧠 AI Insights")

    st.subheader("Feature Importance")

    features = [
        "pH",
        "temperature",
        "voltage",
        "COD",
        "current"
    ]

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Parameter": features,
        "Importance": importance
    }).sort_values(
        "Importance",
        ascending=False
    )

    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Parameter",
        orientation="h",
        title="XGBoost Feature Importance"
    )

    st.plotly_chart(
        fig_importance,
        use_container_width=True
    )

    st.subheader("🔍 Current Reactor Analysis")

    recommendations = []

    if user_ph < 6.5:
        recommendations.append(
            "pH is below the target operating range."
        )

    elif user_ph > 7.0:
        recommendations.append(
            "pH is above the target operating range."
        )

    else:
        recommendations.append(
            "pH is within the target range."
        )

    if user_temp < 30:
        recommendations.append(
            "Temperature is relatively low."
        )

    elif user_temp > 37:
        recommendations.append(
            "Temperature is relatively high."
        )

    else:
        recommendations.append(
            "Temperature is within the selected operating range."
        )

    if user_voltage < 0.5:
        recommendations.append(
            "Applied voltage is relatively low."
        )

    elif user_voltage > 0.8:
        recommendations.append(
            "Applied voltage is relatively high."
        )

    else:
        recommendations.append(
            "Applied voltage is within the selected operating range."
        )

    for item in recommendations:
        st.write("•", item)

    st.subheader("💡 AI Summary")

    st.info(
        f"Under the current simulated input conditions, the XGBoost model "
        f"predicts approximately {predicted_h2:.2f} mL of H₂. "
        f"The current reactor status is {status}."
    )


# =========================================================
# TAB 5 - REPORT
# =========================================================

with tab5:

    st.header("📄 Simulation Report")

    report_data = pd.DataFrame({
        "Parameter": [
            "pH",
            "Temperature (°C)",
            "Applied Voltage (V)",
            "COD (mg/L)",
            "Current (A)",
            "Predicted H₂ (mL)",
            "Model R²",
            "MAE (mL)",
            "Reactor Status"
        ],

        "Value": [
            user_ph,
            user_temp,
            user_voltage,
            user_cod,
            user_current,
            round(predicted_h2, 2),
            round(r2, 3),
            round(mae, 2),
            status
        ]
    })

    st.dataframe(
        report_data,
        use_container_width=True
    )

    csv_data = report_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Simulation Report",
        data=csv_data,
        file_name="MEC_AI_Report.csv",
        mime="text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "MEC-AI | Microbial Electrolysis Cell Intelligence Platform"
)

st.caption(
    "Current version: AI simulation prototype. "
    "Real-time hardware integration can be added through ESP32 + sensors."
)
