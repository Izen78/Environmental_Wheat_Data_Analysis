# --- Step 1: Import libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 2: Load data ---
df = pd.read_csv("merged_final_dataset3.csv", low_memory=False)
TEST_MODE = True  # ✅ set to False on the cluster

if TEST_MODE:
    df = df.sample(n=1000, random_state=42)
    print(f"⚙️ Running in TEST MODE with {len(df)} rows")
else:
    print(f"Running full dataset with {len(df)} rows")

df = df.dropna(subset=['GrainYield_gm2'])

# --- Step 3: Define target and features ---
target = 'GrainYield_gm2'
y = df[target]

exclude_cols = [target, 'Kg/Plot', 'TrialCode', 'Name', 'State', 'RegionName', 
                'SiteDescription', 'SowingDate', 'HarvestDate']
X = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')

# --- Step 4: Identify column types ---
categorical_cols = [c for c in X.columns if X[c].dtype == 'object']
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# --- Step 5: Preprocessing pipelines ---
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
], n_jobs=-1)

# --- Step 6: Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 7: Linear Regression ---
linreg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression(n_jobs=-1))
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
    ('model', RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("\n=== Random Forest Regression ===")
print(f"R²: {r2_rf:.3f}")
print(f"RMSE: {rmse_rf:.3f}")

# --- Step 9: Feature Importance ---
preprocessor = rf_pipeline.named_steps['preprocessor']
model = rf_pipeline.named_steps['model']

num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out()
feature_names = np.concatenate([num_features, cat_features])

feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

plt.figure(figsize=(8, 5))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature', color='skyblue')
plt.title("Top 15 Important Features (Random Forest)")
plt.tight_layout()
#plt.show()
plt.savefig("obj1_feature_importance_rf.png")

# --- Step 10: Cultivar Stability ---
df['PredictedYield'] = rf_pipeline.predict(X)

if 'CultivarID' in df.columns:
    cultivar_stats = (
        df.groupby('CultivarID')
          .agg(mean_yield=('PredictedYield', 'mean'),
               std_yield=('PredictedYield', 'std'))
    )
    cultivar_stats['cv'] = cultivar_stats['std_yield'] / cultivar_stats['mean_yield']
    cultivar_stats = cultivar_stats.sort_values('cv')

    print("\n=== Top 10 Most Stable (Climate-Resistant) Cultivars ===")
    print(cultivar_stats.head(10))

# --- Step 11: K-Means Clustering ---
print("\n🔍 Running K-Means on environmental numeric features...")

# Select numeric columns only
env_features = df[numeric_cols].copy()

# Replace invalid or infinite values
env_features = env_features.replace([np.inf, -np.inf], np.nan)

# Fill missing values robustly
env_features = env_features.fillna(env_features.median(numeric_only=True))

# Drop columns that are still all NaN after imputation
env_features = env_features.dropna(axis=1, how='all')

# Sanity check
if env_features.isna().sum().sum() > 0:
    print("⚠️ Warning: NaNs remain after imputation, dropping remaining rows.")
    env_features = env_features.dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(env_features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# PCA visualization
pca = PCA(n_components=2, random_state=42)
pca_data = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=df['Cluster'], palette='tab10')
plt.title("Environmental Clusters (PCA projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
#plt.show()
plt.savefig("obj1_pca.png")

cluster_yield = df.groupby('Cluster')['GrainYield_gm2'].agg(['mean', 'std'])
print("\n=== Cluster Yield Summary ===")
print(cluster_yield)

# --- Step 12: Export Results for Power BI ---
output_data = df.copy()

# Optional: merge cultivar stats or cluster yield summaries
if 'cultivar_stats' in locals():
    cultivar_stats = cultivar_stats.reset_index()
    output_data = output_data.merge(cultivar_stats, on='CultivarID', how='left')

output_data.to_csv("obj1_powerbi_dashboard_data.csv", index=False)

print("✅ Exported Power BI dataset: obj1_powerbi_dashboard_data.csv")
feat_imp_df.to_csv("obj1_powerbi_feature_importance.csv", index=False)
pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
pca_df['Cluster'] = df['Cluster']
pca_df.to_csv("obj1_powerbi_pca_clusters.csv", index=False)
