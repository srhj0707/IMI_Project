import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# 🔹 1. Load dataset
# =========================
df = pd.read_excel("hydrogel_complete_dataset.xlsx")
df.columns = df.columns.str.strip()

print("Columns:", df.columns.tolist())

# =========================
# 🔹 2. Select important features
# =========================
df = df[[
    "Monomer_A", "Monomer_B",
    "Ratio_A", "Ratio_B",
    "MW_A", "MW_B",
    "Density", "Tg",
    "Crosslinker Type",
    "Crosslinker Concentration (%)",
    "Mesh Size (nm)",
    "Swelling_Ratio",
    "Adsorption Capacity (mg/g)",
    "Contact Angle (deg)",
    "Water Retention Capacity"
]]

# =========================
# 🔹 3. DEFINE numeric columns (FIXED)
# =========================
numeric_cols = [
    "Ratio_A", "Ratio_B",
    "MW_A", "MW_B",
    "Density", "Tg",
    "Crosslinker Concentration (%)",
    "Mesh Size (nm)",
    "Swelling_Ratio",
    "Adsorption Capacity (mg/g)",
    "Contact Angle (deg)",
    "Water Retention Capacity"
]

# Convert to numeric
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows only if targets missing
df = df.dropna(subset=[
    "Swelling_Ratio",
    "Adsorption Capacity (mg/g)",
    "Contact Angle (deg)",
    "Water Retention Capacity"
])

# Fill remaining NaN with mean
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

print("\nDataset shape after cleaning:", df.shape)

# =========================
# 🔹 4. One-hot encoding
# =========================
df = pd.get_dummies(df, columns=[
    "Monomer_A", "Monomer_B", "Crosslinker Type"
])

# =========================
# 🔹 5. Define inputs & outputs
# =========================
target_cols = [
    "Swelling_Ratio",
    "Adsorption Capacity (mg/g)",
    "Contact Angle (deg)",
    "Water Retention Capacity"
]

X = df.drop(columns=target_cols)
y = df[target_cols]

# =========================
# 🔹 6. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 🔹 7. Model
# =========================
model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
)

model.fit(X_train, y_train)

# =========================
# 🔹 8. Prediction
# =========================
y_pred = model.predict(X_test)

# =========================
# 🔹 9. Evaluation
# =========================
print("\n🔹 Model Performance:")

for i, col in enumerate(target_cols):
    print(f"{col} R2:", r2_score(y_test.iloc[:, i], y_pred[:, i]))

print("Overall MSE:", mean_squared_error(y_test, y_pred))

# =========================
# 🔹 10. Feature Importance
# =========================
print("\n🔹 Plotting Feature Importance...")

rf = model.estimators_[0]
importances = rf.feature_importances_
feature_names = X.columns

indices = importances.argsort()[-10:]

plt.figure()
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), feature_names[indices])
plt.title("Top Important Features")
plt.xlabel("Importance")
plt.show()

# =========================
# 🔹 11. Test with realistic sample
# =========================
print("\n🔹 Testing new sample...")

sample = dict.fromkeys(X.columns, 0)

# Numerical inputs
sample["Ratio_A"] = 0.6
sample["Ratio_B"] = 0.4
sample["MW_A"] = 50000
sample["MW_B"] = 60000
sample["Density"] = 1.2
sample["Tg"] = 120
sample["Crosslinker Concentration (%)"] = 0.5
sample["Mesh Size (nm)"] = 25

# ⚠️ IMPORTANT: Update these based on your dataset
# Run: print(X.columns) to confirm names
try:
    sample["Monomer_A_PAA"] = 1
    sample["Monomer_B_PAM"] = 1
    sample["Crosslinker Type_MBAA"] = 1
except:
    print("⚠️ Update one-hot column names based on your dataset")

sample_df = pd.DataFrame([sample])
sample_df = sample_df[X.columns]

prediction = model.predict(sample_df)

print("\nPredicted Properties:")
for i, col in enumerate(target_cols):
    print(f"{col}:", prediction[0][i])

# =========================
# 🔹 12. Save model
# =========================
joblib.dump(model, "forward_model.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("\n✅ Model saved successfully!")
