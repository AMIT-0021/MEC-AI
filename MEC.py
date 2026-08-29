import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# --------------------------------------------------
# 1. PAGE CONFIG & ELECTRIC SPARK EFFECT
# --------------------------------------------------
st.set_page_config(
    page_title="MEC-AI | Hydrogen Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme & glowing UI card styling
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    
    .stMetric {
        background-color: #1E222D;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00D4FF;
        box-shadow: 0 0 12px rgba(0, 212, 255, 0.2);
    }
    
    .status-card {
        background-color: #1E222D;
        padding: 16px;
        border-radius: 12px;
        border-left: 5px solid #00E676;
        margin-bottom: 20px;
        box-shadow: 0 0 20px rgba(0, 230, 118, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# JavaScript for Original Cursor + Animated Electric Lightning Spark Trail
components.html("""
    <script>
    document.addEventListener('mousemove', function(e) {
        // Create 2-3 electric particles per mouse movement
        for (let i = 0; i < 2; i++) {
            let spark = document.createElement('div');
            
            // Randomize particle sizes and electric spark lengths
            let size = Math.random() * 3 + 1;
            let length = Math.random() * 12 + 6;
            let isBlue = Math.random() > 0.3;
            
            spark.style.position = 'fixed';
            spark.style.left = e.clientX + 'px';
            spark.style.top = e.clientY + 'px';
            spark.style.width = size + 'px';
            spark.style.height = length + 'px';
            
            // Cyan blue & white voltage glow
            spark.style.backgroundColor = isBlue ? '#00D4FF' : '#FFFFFF';
            spark.style.boxShadow = isBlue ? 
                '0 0 8px #00D4FF, 0 0 15px #00D4FF' : 
                '0 0 10px #FFFFFF, 0 0 20px #00D4FF';
            
            spark.style.pointerEvents = 'none';
            spark.style.borderRadius = '2px';
            spark.style.zIndex = '999999';
            
            // Scatter particles outward at random angles like lightning arcs
            let angle = Math.random() * 360;
            let distance = Math.random() * 25 + 5;
            let xOffset = Math.cos(angle) * distance;
            let yOffset = Math.sin(angle) * distance;
            
            spark.style.transform = `rotate(${angle}deg)`;
            spark.style.transition = 'all 0.35s cubic-bezier(0.1, 0.8, 0.3, 1)';
            
            window.parent.document.body.appendChild(spark);
            
            // Animate spark movement and fade-out
            setTimeout(() => {
                spark.style.opacity = '0';
                spark.style.left = (e.clientX + xOffset) + 'px';
                spark.style.top = (e.clientY + yOffset) + 'px';
                spark.style.transform += ' scaleY(0.2)';
            }, 20);
            
            // Clean up DOM elements
            setTimeout(() => {
                spark.remove();
            }, 350);
        }
    });
    </script>
""", height=0)

# --------------------------------------------------
# 2. MODEL TRAINING (CACHED FOR SPEED)
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
# 3. OPTIMIZATION CALCULATION (CACHED
