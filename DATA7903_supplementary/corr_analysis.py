# correlation_analysis.py
# Perform correlation analysis on merged_final_dataset3.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load the dataset
# -----------------------------
# Replace with your actual file path
df = pd.read_csv("merged_final_dataset3.csv")

# Display dataset info
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# -----------------------------
# 2. Select numeric columns only
# -----------------------------
numeric_df = df.select_dtypes(include=[np.number])

print("\nNumber of numeric features:", numeric_df.shape[1])

# -----------------------------
# 3. Compute correlation matrix
# -----------------------------
corr_matrix = numeric_df.corr(method='pearson')

# Save correlation matrix to CSV for reference
corr_matrix.to_csv("correlation_matrix.csv", index=True)
print("\nCorrelation matrix saved as correlation_matrix.csv")

# -----------------------------
# 4. Visualize the correlation matrix
# -----------------------------
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Identify strongest correlations
# -----------------------------
# Convert correlation matrix to long format for easier filtering
corr_pairs = (
    corr_matrix.unstack()
    .reset_index()
    .rename(columns={"level_0": "Feature1", "level_1": "Feature2", 0: "Correlation"})
)

# Remove self-correlations (Feature1 == Feature2)
corr_pairs = corr_pairs[corr_pairs["Feature1"] != corr_pairs["Feature2"]]

# Drop duplicate pairs (since correlation is symmetric)
corr_pairs["Pair"] = corr_pairs.apply(
    lambda x: "-".join(sorted([x["Feature1"], x["Feature2"]])), axis=1
)
corr_pairs = corr_pairs.drop_duplicates(subset="Pair").drop(columns="Pair")

# Sort by absolute correlation
strongest = corr_pairs.reindex(
    corr_pairs["Correlation"].abs().sort_values(ascending=False).index
).head(10)

print("\nTop 10 strongest correlations:")
print(strongest.to_string(index=False))

# -----------------------------
# 6. (Optional) Focused correlation plot
# -----------------------------
# If you want to focus on yield or a key variable, uncomment and modify below:
# target_var = 'Yield'  # replace with your actual yield column name
# if target_var in numeric_df.columns:
#     plt.figure(figsize=(10, 6))
#     sns.barplot(
#         x=numeric_df.corr()[target_var].abs().sort_values(ascending=False).index,
#         y=numeric_df.corr()[target_var].abs().sort_values(ascending=False).values
#     )
#     plt.title(f"Correlation with {target_var}")
#     plt.xticks(rotation=90)
#     plt.tight_layout()
#     plt.show()
# else:
#     print(f"'{target_var}' not found in columns.")

