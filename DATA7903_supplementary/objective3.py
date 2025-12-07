# --- Step 1: Import libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from itertools import product
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Step 2: Load dataset ---
df = pd.read_csv("merged_final_dataset2.csv")

df = df.apply(pd.to_numeric, errors="coerce")

# Drop monthly columns (Jan–Dec)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

df = df.drop(columns=[col for col in months if col in df.columns], errors='ignore')


# --- Step 3: Define target and select environmental features only ---
target = "GrainYield_gm2"

# Identify environmental features (you can expand this list as needed)
env_features = [
    'GPSCoordsLatitude', 'GPSCoordsLongitude', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
    'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    'Boron', 'Conductivity_(EC)', 'Copper', 'Exchangeable_Ca', 'Exchangeable_K',
    'Exchangeable_Mg', 'Exchangeable_Na', 'Manganese', 'Organic_Carbon',
    'pH_(CaCl2)', 'pH_(water)', 'Phosphorous', 'Potassium_(K)', 'Sulphur', 'Zinc'
]
env_features = [f for f in env_features if f in df.columns]

# Drop missing target values
df = df.dropna(subset=[target])

# --- Step 4: Split data ---
X = df[env_features]
y = df[target]

# Drop rows where the target is missing
df = df.dropna(subset=[target])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 5: Impute missing numeric values ---
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')

X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# --- Step 6: Random Forest ---
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train_imputed, y_train)
y_pred_rf = rf.predict(X_test_imputed)

print("Random Forest R²:", r2_score(y_test, y_pred_rf))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

rf_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop 10 features (Random Forest):")
print(rf_importance.head(10))

plt.figure(figsize=(10, 6))
plt.barh(rf_importance["Feature"][:10][::-1], rf_importance["Importance"][:10][::-1])
plt.xlabel("Importance")
plt.title("Top Environmental Features (Random Forest)")
plt.show()

# --- Step 7: Lasso Regression ---
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

# Scale after imputation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
lasso.fit(X_train_scaled, y_train)

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lasso.coef_
}).sort_values("Coefficient", key=np.abs, ascending=False)

print("\nTop 10 features (Lasso):")
print(coef_df.head(10))


from sklearn.impute import SimpleImputer

# ------------------------------------------------------------
# 5. Lasso Regression (Feature Selection)
# ------------------------------------------------------------
# Impute missing numeric values with column medians
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_train)

# Standardize the imputed features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Fit LassoCV with 5-fold cross-validation
lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
lasso.fit(X_scaled, y_train)

# Create dataframe of nonzero coefficients
lasso_coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso.coef_
})
lasso_coef = lasso_coef[lasso_coef['Coefficient'] != 0].sort_values(by='Coefficient', ascending=False)

print("\nSelected Features (Lasso):")
print(lasso_coef)

# --- Step 8: Select top features for optimization ---
selected_features = list(set(
    rf_importance["Feature"].head(5).tolist() + 
    coef_df["Feature"].head(5).tolist()
))

print("\nSelected features for optimization:", selected_features)

# --- Step 9: Compute IQR ranges for selected features ---
iqr_ranges = {}
for f in selected_features:
    q1, q3 = np.percentile(X[f].dropna(), [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_ranges[f] = (lower, upper)

print("\nFeature ranges based on IQR:")
for f, (low, high) in iqr_ranges.items():
    print(f"{f}: {low:.3f} to {high:.3f}")

# --- Step 10: Grid Search over selected features (coarse search) ---
grid_points = 5  # adjust for finer search
grid_values = {
    f: np.linspace(iqr_ranges[f][0], iqr_ranges[f][1], grid_points)
    for f in selected_features
}

param_grid = list(product(*grid_values.values()))

best_yield = -np.inf
best_combo = None

# Create a template of median values for all model features
median_template = X_train_imputed.median().to_dict()

for combo in param_grid:
    combo_dict = dict(zip(selected_features, combo))
    
    # Start from median values
    full_input = median_template.copy()
    # Replace with our selected feature values
    full_input.update(combo_dict)
    
    # Build DataFrame with all model features in correct order
    x_input = pd.DataFrame([full_input])[X_train_imputed.columns]
    
    yield_pred = rf.predict(x_input)[0]
    if yield_pred > best_yield:
        best_yield = yield_pred
        best_combo = combo_dict

print("\nBest combination from grid search:")
print(best_combo)
print(f"Predicted yield: {best_yield:.2f} g/m²")

# --- Step 11: Constraint-Based Optimization ---
def objective(x):
    combo_dict = dict(zip(selected_features, x))
    full_input = median_template.copy()
    full_input.update(combo_dict)
    x_input = pd.DataFrame([full_input])[X_train_imputed.columns]
    return -rf.predict(x_input)[0]

bounds = [iqr_ranges[f] for f in selected_features]
x0 = [np.mean(b) for b in bounds]

result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")

optimized_values = dict(zip(selected_features, result.x))
optimized_yield = -result.fun

print("\nConstraint-based optimization results:")
print("Optimal environmental values:")
for f, val in optimized_values.items():
    print(f"{f}: {val:.3f}")
print(f"Predicted optimal yield: {optimized_yield:.2f} g/m²")

# --- Step 12: Export results for Power BI integration ---

# 1. Random Forest feature importance
rf_importance.to_csv("Objective3/rf_feature_importance.csv", index=False)

# 2. Lasso coefficients (all)
coef_df.to_csv("Objective3/lasso_coefficients.csv", index=False)

# 3. Lasso-selected nonzero features only
lasso_coef.to_csv("Objective3/lasso_selected_features.csv", index=False)

# 4. Grid search optimal combination
pd.DataFrame([{
    "Optimization_Method": "Grid Search",
    **best_combo,
    "Predicted_Yield_gm2": best_yield
}]).to_csv("Objective3/grid_search_results.csv", index=False)

# 5. Constraint-based optimization results
pd.DataFrame([{
    "Optimization_Method": "Constraint-Based",
    **optimized_values,
    "Predicted_Yield_gm2": optimized_yield
}]).to_csv("Objective3/constraint_opt_results.csv", index=False)

# 6. Optional: Combined summary table for Power BI
combined_results = pd.concat([
    pd.DataFrame([{"Optimization_Method": "Grid Search", **best_combo, "Predicted_Yield_gm2": best_yield}]),
    pd.DataFrame([{"Optimization_Method": "Constraint-Based", **optimized_values, "Predicted_Yield_gm2": optimized_yield}])
], ignore_index=True)

combined_results.to_csv("Objective3/optimization_summary.csv", index=False)

print("\n✅ All model outputs exported for Power BI:")
print(" - rf_feature_importance.csv")
print(" - lasso_coefficients.csv")
print(" - lasso_selected_features.csv")
print(" - grid_search_results.csv")
print(" - constraint_opt_results.csv")
print(" - optimization_summary.csv")
