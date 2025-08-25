# ================================
# Task 1: Student Score Prediction
# ================================

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import os

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("StudentPerformanceFactors.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# -------------------------
# Data Cleaning
# -------------------------
# Fill missing categorical values with mode
for col in ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Check missing values again
print("Missing values after cleaning:\n", df.isnull().sum())

# -------------------------
# Exploratory Visualization
# -------------------------
plt.figure(figsize=(7,5))
sns.scatterplot(x=df["Hours_Studied"], y=df["Exam_Score"], alpha=0.6)
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.savefig("scatter_hours_vs_score.png")
plt.show()

# -------------------------
# Train/Test Split
# -------------------------
X = df[["Hours_Studied"]]      # feature
y = df["Exam_Score"]           # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Linear Regression
# -------------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Linear Regression Metrics:")
print(f" MAE: {mae:.2f}")
print(f" RMSE: {rmse:.2f}")
print(f" R²: {r2:.2f}")

# Save model
joblib.dump(lin_reg, "linear_hours_studied.joblib")

# Plot Actual vs Predicted
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs Predicted - Linear Regression (Hours_Studied)")
plt.savefig("actual_vs_pred_linear.png")
plt.show()

# -------------------------
# Polynomial Regression
# -------------------------
def polynomial_regression(degree):
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("linear", LinearRegression())
    ])
    model.fit(X_train, y_train)
    y_pred_poly = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred_poly)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_poly))
    r2 = r2_score(y_test, y_pred_poly)
    
    print(f"Polynomial Regression (degree={degree}) Metrics:")
    print(f" MAE: {mae:.2f}")
    print(f" RMSE: {rmse:.2f}")
    print(f" R²: {r2:.2f}")
    
    # Save model
    joblib.dump(model, f"poly_deg{degree}.joblib")
    
    # Plot
    plt.figure(figsize=(7,5))
    plt.scatter(y_test, y_pred_poly, alpha=0.6, color="green")
    plt.xlabel("Actual Exam Score")
    plt.ylabel("Predicted Exam Score")
    plt.title(f"Actual vs Predicted - Polynomial Regression (deg={degree})")
    plt.savefig(f"actual_vs_pred_poly_deg{degree}.png")
    plt.show()

# Try degree 2 and 3
polynomial_regression(2)
polynomial_regression(3)

# -------------------------
# Multi-feature Regression (Bonus)
# -------------------------
features = ["Hours_Studied", "Sleep_Hours", "Attendance"]
X_multi = df[features]

X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

lin_reg_multi = LinearRegression()
lin_reg_multi.fit(X_train, y_train)
y_pred_multi = lin_reg_multi.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_multi)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_multi))
r2 = r2_score(y_test, y_pred_multi)

print("Multi-feature Linear Regression Metrics:")
print(f" MAE: {mae:.2f}")
print(f" RMSE: {rmse:.2f}")
print(f" R²: {r2:.2f}")

# Save model
joblib.dump(lin_reg_multi, "linear_multi_features.joblib")

plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred_multi, alpha=0.6, color="red")
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs Predicted - Multi-feature Linear Regression")
plt.savefig("actual_vs_pred_linear_multifeature.png")
plt.show()
