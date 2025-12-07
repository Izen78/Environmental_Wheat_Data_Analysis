import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# === Load dataset ===
df = pd.read_csv("Clean/filtered_file.csv")

# === Define features and target ===
soil_features = ["Organic_Carbon", "Copper", "Manganese", "Boron"]
target = "Kg/Plot"

# Drop rows with missing values in selected columns
df = df.dropna(subset=soil_features + [target])

X = df[soil_features]
y = df[target]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 1. Random Forest Regressor ===
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# === 2. Lasso Regression (with cross-validation) ===
lasso_model = LassoCV(cv=5, random_state=42)
lasso_model.fit(X_train, y_train)
lasso_preds = lasso_model.predict(X_test)

# === Evaluation ===
def evaluate_model(name, y_test, y_pred):
    print(f"{name} Results:")
    print(f"  R2 Score: {r2_score(y_test, y_pred):.3f}")
    print(f"  RMSE: {mean_squared_error(y_test, y_pred):.3f}")
    print()

evaluate_model("Random Forest", y_test, rf_preds)
evaluate_model("Lasso Regression", y_test, lasso_preds)

# === Plot Feature Importances (Random Forest) ===
plt.figure(figsize=(8, 5))
sns.barplot(x=rf_model.feature_importances_, y=soil_features)
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# === Plot Lasso Coefficients ===
plt.figure(figsize=(8, 5))
sns.barplot(x=lasso_model.coef_, y=soil_features)
plt.title("Lasso Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()
