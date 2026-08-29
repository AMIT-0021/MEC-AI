import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor


# --------------------------------------------------
# 1. SAMPLE MEC DATA
# --------------------------------------------------
# Replace this dataset later with your real sensor data.

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


# --------------------------------------------------
# 2. INPUTS AND OUTPUT
# --------------------------------------------------

X = df[
    ["pH", "temperature", "voltage", "COD", "current"]
]

y = df["H2"]


# --------------------------------------------------
# 3. SPLIT DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 4. CREATE AI MODEL
# --------------------------------------------------

model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=42
)


# --------------------------------------------------
# 5. TRAIN
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# 6. TEST
# --------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print("\n==============================")
print(" MEC AI MODEL RESULTS")
print("==============================")

print(f"Mean Absolute Error: {mae:.2f} mL")
print(f"R² Score: {r2:.2f}")


# --------------------------------------------------
# 7. PREDICT H2 FOR NEW CONDITIONS
# --------------------------------------------------

new_condition = pd.DataFrame({
    "pH": [6.8],
    "temperature": [35],
    "voltage": [0.7],
    "COD": [550],
    "current": [0.46]
})

predicted_h2 = model.predict(new_condition)[0]

print("\n------------------------------")
print("NEW MEC CONDITION")
print("------------------------------")

print("pH:", new_condition["pH"].iloc[0])
print("Temperature:", new_condition["temperature"].iloc[0], "°C")
print("Voltage:", new_condition["voltage"].iloc[0], "V")
print("COD:", new_condition["COD"].iloc[0], "mg/L")
print("Current:", new_condition["current"].iloc[0], "A")

print(f"\nPredicted H2 Production: {predicted_h2:.2f} mL")


# --------------------------------------------------
# 8. FIND BEST CONDITION
# --------------------------------------------------

print("\n==============================")
print(" FINDING OPTIMAL CONDITION")
print("==============================")

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


print("\nRecommended operating condition:")

print("pH:",
      best_condition["pH"].iloc[0])

print("Temperature:",
      best_condition["temperature"].iloc[0],
      "°C")

print("Voltage:",
      best_condition["voltage"].iloc[0],
      "V")

print("COD:",
      best_condition["COD"].iloc[0],
      "mg/L")

print("Current:",
      best_condition["current"].iloc[0],
      "A")

print(f"\nPredicted maximum H2: {best_h2:.2f} mL")


# --------------------------------------------------
# 9. GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    predictions
)

plt.xlabel("Actual H2 Production (mL)")
plt.ylabel("Predicted H2 Production (mL)")
plt.title("MEC AI: Actual vs Predicted Hydrogen Production")

plt.grid(True)
plt.show()