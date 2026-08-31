import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MEC-AI | Hydrogen Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PREMIUM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0,255,200,0.08), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(100,80,255,0.08), transparent 30%),
        radial-gradient(circle at 50% 90%, rgba(0,180,255,0.06), transparent 35%),
        #070b14;
}

/* Main title */
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    letter-spacing: 3px;
    margin-top: 5px;
    margin-bottom: 0px;
    background: linear-gradient(
        90deg,
        #00f5d4,
        #00bbf9,
        #9b5de5,
        #00f5d4
    );
    background-size: 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientMove 6s ease infinite;
}

@keyframes gradientMove {
    0% { background-position: 0%; }
    50% { background-position: 100%; }
    100% { background-position: 0%; }
}

.hero-subtitle {
    text-align: center;
    color: #9ba8c7;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Cards */
.card {
    background: rgba(17, 24, 39, 0.78);
    border: 1px solid rgba(0, 245, 212, 0.18);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 15px;
    box-shadow:
        0 8px 30px rgba(0,0,0,0.30),
        inset 0 1px 0 rgba(255,255,255,0.03);
    backdrop-filter: blur(12px);
}

.card:hover {
    border-color: rgba(0,245,212,0.45);
    box-shadow:
        0 10px 35px rgba(0,245,212,0.08);
}

/* Section headings */
.section-title {
    font-family: 'Orbitron', sans-serif;
    color: #00f5d4;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 12px;
}

/* Status */
.status-online {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 30px;
    background: rgba(0,245,212,0.12);
    border: 1px solid rgba(0,245,212,0.35);
    color: #00f5d4;
    font-weight: 700;
    font-size: 13px;
}

/* AI brain */
.ai-brain {
    text-align: center;
    padding: 28px;
    border-radius: 20px;
    background:
        radial-gradient(circle at center,
        rgba(0,245,212,0.13),
        rgba(17,24,39,0.85) 65%);
    border: 1px solid rgba(0,245,212,0.28);
    box-shadow: 0 0 35px rgba(0,245,212,0.07);
}

.brain-icon {
    font-size: 58px;
}

.brain-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 23px;
    color: #ffffff;
    margin-top: 8px;
}

.brain-text {
    color: #9ba8c7;
    font-size: 14px;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(17,24,39,0.75);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 16px;
    border-radius: 15px;
}

[data-testid="stMetricLabel"] {
    color: #8d9ab8;
}

[data-testid="stMetricValue"] {
    color: #ffffff;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid rgba(0,245,212,0.35);
    background: linear-gradient(
        135deg,
        rgba(0,245,212,0.15),
        rgba(0,187,249,0.10)
    );
    color: #ffffff;
    font-weight: 700;
    padding: 12px;
    transition: 0.25s;
}

.stButton > button:hover {
    border-color: #00f5d4;
    box-shadow: 0 0 20px rgba(0,245,212,0.18);
    transform: translateY(-1px);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #090e1a,
            #070b14
        );
    border-right: 1px solid rgba(0,245,212,0.10);
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.08);
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# GLITTER CURSOR
# =========================================================

st.markdown("""
<script>
(function() {

    const colors = [
        "#00f5d4",
        "#00bbf9",
        "#9b5de5",
        "#ffffff"
    ];

    function createSparkle(x, y) {

        const sparkle = document.createElement("div");

        const size = Math.random() * 7 + 3;
        const color = colors[
            Math.floor(Math.random() * colors.length)
        ];

        sparkle.style.position = "fixed";
        sparkle.style.left = x + "px";
        sparkle.style.top = y + "px";
        sparkle.style.width = size + "px";
        sparkle.style.height = size + "px";
        sparkle.style.background = color;
        sparkle.style.borderRadius = "50%";
        sparkle.style.pointerEvents = "none";
        sparkle.style.zIndex = "999999";
        sparkle.style.boxShadow =
            "0 0 8px " + color +
            ", 0 0 16px " + color;

        document.body.appendChild(sparkle);

        const dx = (Math.random() - 0.5) * 40;
        const dy = (Math.random() - 0.5) * 40;

        sparkle.animate(
            [
                {
                    transform: "translate(0,0) scale(1)",
                    opacity: 1
                },
                {
                    transform:
                        "translate(" +
                        dx +
                        "px," +
                        dy +
                        "px) scale(0)",
                    opacity: 0
                }
            ],
            {
                duration: 650,
                easing: "ease-out"
            }
        );

        setTimeout(() => {
            sparkle.remove();
        }, 700);
    }

    let lastTime = 0;

    document.addEventListener("mousemove", function(e) {

        const now = Date.now();

        if (now - lastTime > 35) {

            createSparkle(e.clientX, e.clientY);

            lastTime = now;
        }
    });

})();
</script>
""", unsafe_allow_html=True)


# =========================================================
# DATA
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
# MODEL
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

    prediction = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    r2 = r2_score(
        y_test,
        prediction
    )

    validation = pd.DataFrame({
        "Actual H₂": y_test.values,
        "Predicted H₂": prediction
    })

    return df, model, mae, r2, validation


df, model, mae, r2, validation = train_model()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="hero-title">⚡ MEC-AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'MICROBIAL ELECTROLYSIS • HYDROGEN INTELLIGENCE PLATFORM'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div style="text-align:center;">'
    '<span class="status-online">● SYSTEM ONLINE</span>'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    "## ⚡ MEC-AI CONTROL"
)

st.sidebar.markdown(
    "### 🧪 Reactor Parameters"
)

user_ph = st.sidebar.slider(
    "pH",
    6.0,
    7.5,
    6.8,
    0.1
)

user_temp = st.sidebar.slider(
    "Temperature (°C)",
    25,
    40,
    35,
    1
)

user_voltage = st.sidebar.slider(
    "Applied Voltage (V)",
    0.1,
    1.0,
    0.70,
    0.05
)

user_cod = st.sidebar.number_input(
    "COD (mg/L)",
    400,
    800,
    550,
    10
)

user_current = st.sidebar.slider(
    "Current (A)",
    0.1,
    0.8,
    0.46,
    0.01
)

st.sidebar.divider()

demo_mode = st.sidebar.toggle(
    "🎮 AI Demo Mode",
    value=True
)

st.sidebar.caption(
    "Demo Mode uses simulated data. "
    "For the final prototype, connect these parameters "
    "to ESP32 sensors."
)


# =========================================================
# INPUT DATA
# =========================================================

new_condition = pd.DataFrame({
    "pH": [user_ph],
    "temperature": [user_temp],
    "voltage": [user_voltage],
    "COD": [user_cod],
    "current": [user_current]
})

predicted_h2 = float(
    model.predict(new_condition)[0]
)


# =========================================================
# REACTOR STATUS
# =========================================================

def get_status():

    score = 0

    if 6.5 <= user_ph <= 7.0:
        score += 1

    if 30 <= user_temp <= 37:
        score += 1

    if 0.5 <= user_voltage <= 0.8:
        score += 1

    if 0.35 <= user_current <= 0.55:
        score += 1

    if score == 4:
        return "🟢 OPTIMAL"

    if score >= 2:
        return "🟡 MODERATE"

    return "🔴 ATTENTION"


status = get_status()


# =========================================================
# AI BRAIN CARD
# =========================================================

st.markdown("""
<div class="ai-brain">

<div class="brain-icon">🧠</div>

<div class="brain-title">
MEC-AI BRAIN
</div>

<div class="brain-text">
XGBoost-powered hydrogen prediction and reactor intelligence
</div>

</div>
""", unsafe_allow_html=True)

st.write("")


# =========================================================
# MAIN METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "⚡ Predicted H₂",
    f"{predicted_h2:.1f} mL"
)

c2.metric(
    "🧠 Model R²",
    f"{r2:.2f}"
)

c3.metric(
    "📉 MAE",
    f"{mae:.1f} mL"
)

c4.metric(
    "🧪 Reactor",
    status
)


# =========================================================
# DASHBOARD TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Reactor",
    "🧠 AI Optimization",
    "📊 Analytics",
    "🚨 Intelligence"
])


# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.markdown(
        '<div class="section-title">⚡ REACTOR COMMAND CENTER</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1, 1])

    with left:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### 🧪 Current Reactor Conditions"
        )

        params = pd.DataFrame({
            "Parameter": [
                "pH",
                "Temperature",
                "Voltage",
                "COD",
                "Current"
            ],

            "Value": [
                f"{user_ph:.1f}",
                f"{user_temp} °C",
                f"{user_voltage:.2f} V",
                f"{user_cod} mg/L",
                f"{user_current:.2f} A"
            ]
        })

        st.dataframe(
            params,
            hide_index=True,
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### ⚡ Hydrogen Output"
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=predicted_h2,
                number={
                    "suffix": " mL",
                    "font": {
                        "size": 42
                    }
                },

                title={
                    "text": "AI Predicted H₂"
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
                            "range": [0, 70]
                        },
                        {
                            "range": [70, 130]
                        },
                        {
                            "range": [130, 200]
                        }
                    ]
                }
            )
        )

        gauge.update_layout(
            height=300,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="white"
            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# TAB 2 — OPTIMIZATION
# =========================================================

with tab2:

    st.markdown(
        '<div class="section-title">🎯 AI OPTIMIZATION ENGINE</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The AI searches possible operating conditions and identifies "
        "the combination with the highest predicted hydrogen output."
    )

    if st.button(
        "🚀 RUN AI OPTIMIZATION"
    ):

        best_h2 = -999
        best = None

        progress = st.progress(0)

        combinations = 0

        ph_values = np.arange(
            6.3,
            7.2,
            0.1
        )

        temp_values = range(
            30,
            38,
            2
        )

        voltage_values = np.arange(
            0.5,
            0.81,
            0.05
        )

        total = (
            len(ph_values)
            * len(temp_values)
            * len(voltage_values)
        )

        for ph in ph_values:

            for temp in temp_values:

                for voltage in voltage_values:

                    test = pd.DataFrame({
                        "pH": [round(ph, 2)],
                        "temperature": [temp],
                        "voltage": [round(voltage, 2)],
                        "COD": [user_cod],
                        "current": [user_current]
                    })

                    result = float(
                        model.predict(test)[0]
                    )

                    if result > best_h2:

                        best_h2 = result

                        best = {
                            "pH": round(ph, 2),
                            "Temperature": temp,
                            "Voltage": round(voltage, 2),
                            "COD": user_cod,
                            "Current": user_current
                        }

                    combinations += 1

                    progress.progress(
                        min(
                            combinations / total,
                            1.0
                        )
                    )

        st.success(
            "🎯 AI optimization completed!"
        )

        a, b = st.columns(2)

        a.metric(
            "Current Prediction",
            f"{predicted_h2:.1f} mL"
        )

        b.metric(
            "Optimized Prediction",
            f"{best_h2:.1f} mL"
        )

        st.subheader(
            "🧠 AI Recommended Conditions"
        )

        recommended = pd.DataFrame({
            "Parameter": [
                "pH",
                "Temperature",
                "Voltage",
                "COD",
                "Current"
            ],

            "Recommended": [
                best["pH"],
                f'{best["Temperature"]} °C',
                f'{best["Voltage"]} V',
                f'{best["COD"]} mg/L',
                f'{best["Current"]:.2f} A'
            ]
        })

        st.table(recommended)

        st.warning(
            "⚠️ AI recommendations are model-based. "
            "They must be experimentally validated before changing "
            "a physical MEC reactor."
        )


# =========================================================
# TAB 3 — ANALYTICS
# =========================================================

with tab3:

    st.markdown(
        '<div class="section-title">📊 AI ANALYTICS</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🎯 Actual vs Predicted H₂"
        )

        fig = px.scatter(
            validation,
            x="Actual H₂",
            y="Predicted H₂",
            title="Model Validation"
        )

        minimum = min(
            validation["Actual H₂"].min(),
            validation["Predicted H₂"].min()
        )

        maximum = max(
            validation["Actual H₂"].max(),
            validation["Predicted H₂"].max()
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    minimum,
                    maximum
                ],

                y=[
                    minimum,
                    maximum
                ],

                mode="lines",
                name="Perfect Prediction"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "🧠 Feature Importance"
        )

        feature_names = [
            "pH",
            "temperature",
            "voltage",
            "COD",
            "current"
        ]

        importance = model.feature_importances_

        feature_df = pd.DataFrame({
            "Parameter": feature_names,
            "Importance": importance
        }).sort_values(
            "Importance",
            ascending=True
        )

        fig2 = px.bar(
            feature_df,
            x="Importance",
            y="Parameter",
            orientation="h",
            title="What Influences AI Prediction?"
        )

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


# =========================================================
# TAB 4 — INTELLIGENCE
# =========================================================

with tab4:

    st.markdown(
        '<div class="section-title">🚨 REACTOR INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    alerts = []

    if user_ph < 6.5:
        alerts.append(
            "🔴 pH is below the target operating range."
        )

    elif user_ph > 7.0:
        alerts.append(
            "🟡 pH is above the target operating range."
        )

    else:
        alerts.append(
            "🟢 pH is within the target range."
        )

    if 30 <= user_temp <= 37:
        alerts.append(
            "🟢 Temperature is within the selected range."
        )

    else:
        alerts.append(
            "🟡 Temperature is outside the selected range."
        )

    if 0.5 <= user_voltage <= 0.8:
        alerts.append(
            "🟢 Applied voltage is within the selected range."
        )

    else:
        alerts.append(
            "🟡 Applied voltage should be reviewed."
        )

    if 0.35 <= user_current <= 0.55:
        alerts.append(
            "🟢 Current is within the selected range."
        )

    else:
        alerts.append(
            "🟡 Current is outside the selected range."
        )

    for alert in alerts:

        st.markdown(
            f'<div class="card">{alert}</div>',
            unsafe_allow_html=True
        )

    st.subheader(
        "🧠 AI Summary"
    )

    st.info(
        f"The AI currently predicts approximately "
        f"**{predicted_h2:.1f} mL H₂** under the entered conditions. "
        f"The reactor condition is classified as **{status}**."
    )

    st.subheader(
        "🔌 Future Hardware Integration"
    )

    st.code("""
REAL MEC REACTOR
       ↓
pH Sensor
Temperature Sensor
Voltage Sensor
Current Sensor
Hydrogen Sensor
       ↓
     ESP32
       ↓
      Wi-Fi
       ↓
    MEC-AI
       ↓
Prediction + Optimization + Alerts
""", language="text")


# =========================================================
# DOWNLOAD REPORT
# =========================================================

st.divider()

st.subheader(
    "📥 Export AI Report"
)

report = pd.DataFrame({
    "Parameter": [
        "pH",
        "Temperature (°C)",
        "Applied Voltage (V)",
        "COD (mg/L)",
        "Current (A)",
        "Predicted H₂ (mL)",
        "R² Score",
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

csv = report.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "⬇️ Download MEC-AI Report",
    data=csv,
    file_name="MEC_AI_Report.csv",
    mime="text/csv"
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#6f7b96;font-size:12px;">
        ⚡ MEC-AI • Microbial Electrolysis Cell Intelligence Platform
        <br>
        AI Simulation Prototype • ESP32 Integration Ready
    </div>
    """,
    unsafe_allow_html=True
)
