# --- Step 1: Import libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 2: Load data ---
df = pd.read_csv("merged_final_dataset3.csv")

df = df.dropna(subset=['GrainYield_gm2'])

df=df.sample(n=100)


# --- Step 3: Define target and features ---
target = 'GrainYield_gm2'
y = df[target]

exclude_cols = [
    target, 'Kg/Plot', 'TrialCode', 'Name', 'State', 'RegionName', 
    'SiteDescription', 'SowingDate', 'HarvestDate',
    'SoilTestID'  # remove SoilTestID
]

# Drop columns that are in exclude_cols
X = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')

# --- Remove any month-like columns ---
# Example: if your dataset has columns named after months
month_cols = [c for c in X.columns if c.lower() in 
              ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']]
if month_cols:
    print(f"Excluding month columns: {month_cols}")
    X = X.drop(columns=month_cols)


# --- Step 4: Identify column types ---
categorical_cols = [c for c in X.columns if X[c].dtype == 'object']
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# --- Step 5: Preprocessing pipeline ---
from sklearn.impute import SimpleImputer

# --- Step 5: Preprocessing pipeline (with imputation added) ---
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),   # handles missing numeric values
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # fills missing categories
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# --- Step 6 onwards stays the same ---
linreg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# --- Step 6: Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linreg_pipeline.fit(X_train, y_train)
y_pred_lr = linreg_pipeline.predict(X_test)


# --- Step 7: Linear Regression ---
linreg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

linreg_pipeline.fit(X_train, y_train)
y_pred_lr = linreg_pipeline.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print("=== Multiple Linear Regression ===")
print(f"R²: {r2_lr:.3f}")
print(f"RMSE: {rmse_lr:.3f}")

# --- Step 8: Random Forest Regression ---
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=300, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("\n=== Random Forest Regression ===")
print(f"R²: {r2_rf:.3f}")
print(f"RMSE: {rmse_rf:.3f}")

# --- Step 9: Feature Importance (from Random Forest) ---
# Extract feature names after preprocessing
onehot = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
encoded_cat_names = onehot.get_feature_names_out(categorical_cols)
feature_names = np.concatenate([numeric_cols, encoded_cat_names])

importances = rf_pipeline.named_steps['model'].feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False).head(15)

plt.figure(figsize=(8,5))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature', color='skyblue')
plt.title("Top 15 Important Features (Random Forest)")
plt.tight_layout()
plt.show()

# --- Step 10: Cultivar Stability (Coefficient of Variation) ---
# Predict yields for all samples to measure cultivar stability
all_preds = rf_pipeline.predict(X)
df['PredictedYield'] = all_preds

cultivar_stats = (
    df.groupby('CultivarID')
      .agg(mean_yield=('PredictedYield', 'mean'),
           std_yield=('PredictedYield', 'std'))
)
cultivar_stats['cv'] = cultivar_stats['std_yield'] / cultivar_stats['mean_yield']
cultivar_stats = cultivar_stats.sort_values('cv')

print("\n=== Top 10 Most Stable (Climate-Resistant) Cultivars ===")
print(cultivar_stats.head(10))

# --- Step 11: K-Means Clustering on Environmental Variables ---
env_features = df[numeric_cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(env_features)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# PCA for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=df['Cluster'], palette='tab10')
plt.title("Environmental Clusters (PCA projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# Cluster-level yield statistics
cluster_yield = df.groupby('Cluster')['GrainYield_gm2'].agg(['mean', 'std'])
print("\n=== Cluster Yield Summary ===")
print(cluster_yield)
