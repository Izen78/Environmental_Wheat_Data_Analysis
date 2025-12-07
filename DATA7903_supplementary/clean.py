import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import numpy as np
import os

# Load the Excel file
file_path = "Simulation/apsim_out_daily_anonymized.csv"
df = pd.read_csv(file_path)

# Step 1: Preview structure
print("🔹 Shape of dataset:", df.shape)
print("🔹 First few rows:")
print(df.head())

# Step 2: Rename columns (if unnamed or unclear)
df.columns = df.columns.str.strip()  # remove leading/trailing whitespace
df.columns = df.columns.str.replace(' ', '_')  # optional: replace spaces with underscores

# Step 3: Drop completely empty rows or columns
df.dropna(how='all', inplace=True)
df.dropna(axis=1, how='all', inplace=True)

# Step 4: Identify and handle missing values
print("\n🔹 Missing values per column:")
print(df.isnull().sum())

# Step 5: Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Imputation for Categorical Columns using Mode Imputation
imputer = SimpleImputer(strategy='most_frequent')  # For categorical, impute with mode
df[categorical_cols] = imputer.fit_transform(df[categorical_cols])

# Step 6: Drop columns with more than 50% missing values
missing_percent = df.isnull().mean() * 100
cols_to_drop = missing_percent[missing_percent > 50].index
df.drop(columns=cols_to_drop, inplace=True)
print(f"\n🔹 Dropped columns (>{50}% missing):\n{list(cols_to_drop)}")

# Step 7: Linear Regression Imputation for numeric columns
numeric_df = df.select_dtypes(include=[np.number])
non_numeric_df = df.select_dtypes(exclude=[np.number])  # Preserve non-numeric

# Impute each numeric column using linear regression if it has missing values
for col in numeric_df.columns:
    if numeric_df[col].isnull().sum() == 0:
        continue

    # Define rows where this column is not null
    not_null = numeric_df[col].notnull()
    null = numeric_df[col].isnull()

    # Use all other numeric columns as predictors
    predictors = numeric_df.drop(columns=[col])

    # Only keep columns that are fully observed in the training set
    complete_cols = predictors.columns[predictors.notnull().all()]
    if complete_cols.empty:
        continue  # skip if no complete predictors

    X_train = predictors.loc[not_null, complete_cols]
    y_train = numeric_df.loc[not_null, col]
    X_pred = predictors.loc[null, complete_cols]

    if not X_pred.empty:
        model = LinearRegression()
        model.fit(X_train, y_train)
        imputed_values = model.predict(X_pred)
        numeric_df.loc[null, col] = imputed_values

# Combine numeric and non-numeric back
df_imputed = pd.concat([numeric_df, non_numeric_df], axis=1)

# Step 8: Export to CSV
output_file = "cleaned_simulated.csv"
df_imputed.to_csv(output_file, index=False)
print(f"\n✅ Cleaned dataset saved to: {output_file}")

# Step 9: Remove duplicates
df.drop_duplicates(inplace=True)

# Step 10: Final check
print("\n🔹 Cleaned dataset info:")
print(df.info())

# Count missing values per column
missing_counts = df.isnull().sum()

# Keep only columns with missing values and sort them
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

# Plot
plt.figure(figsize=(14, 6))
missing_counts.plot(kind='bar', color='steelblue')
plt.title('Missing Values per Column')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Calculate percentage of missing values per column
missing_percent = df.isnull().mean() * 100

# Filter only columns with missing values and sort
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
print(missing_percent)

# Plot
plt.figure(figsize=(14, 6))
missing_percent.plot(kind='bar', color='coral')
plt.title('Percentage of Missing Values per Column')
plt.ylabel('Missing Value Percentage (%)')
plt.xlabel('Columns')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
