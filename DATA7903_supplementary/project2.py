# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

# # Load each dataset and add the class label
# dfs = []
# for i in [1, 2, 3, 4]:  # Assuming 4 severity levels
#     df_i = pd.read_csv(f"Clean/frost_damage_value_{i}.csv")
#     df_i["Frost damage"] = i  # Assign correct class label
#     dfs.append(df_i)




# # Combine all class datasets into one
# df = pd.concat(dfs, ignore_index=True)
# # Keep only the specified columns
# desired_columns = [
#     'Kg/Plot', 'Frost damage'
# ]
# df = df[[col for col in desired_columns if col in df.columns]]

# # Drop columns with more than 30% missing values
# threshold = 0.3
# df = df.loc[:, df.isnull().mean() < threshold]

# # Then fill the remaining missing values
# df = df.fillna(df.mean(numeric_only=True))


# # Define features and target
# # Define features and target
# X = df.drop('Frost damage', axis=1).select_dtypes(include=[np.number])
# Y = df['Frost damage']


# # Split into train/test
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

# # Fit multinomial logistic regression
# model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
# model.fit(X_train, Y_train)

# # Predict log-odds
# logits = model.predict_log_proba(X_test)

# # Plot logits against the first predictor
# plt.figure(figsize=(10, 6))
# for i in range(logits.shape[1]):
#     plt.plot(logits[:, i], X_test.iloc[:, 0], label=f"Class {i+1}")

# plt.title('Log-Odds (Logits) vs First Predictor')
# plt.xlabel('First Predictor')
# plt.ylabel('Log-Odds')
# plt.legend(title='Classes')
# plt.show()



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# === Load the dataset ===
df = pd.read_csv("Clean/merged_frost_damage.csv")  # update path if needed


# === Select only the relevant columns ===
columns = [
    "Year", "Name", "State", "MET Analysis Mega Region", "RegionName",
    "TrialCode", "SiteDescription", "SowingDate", "HarvestDate", "Abandoned",
    "GPSCoordsLatitude", "GPSCoordsLongitude", "RowSpacing2", "Range", "Row",
    "CultivarID", "Harvest Length", "Harvest Width", "Kg/Plot", "GrainYield_gm2",
    "Residual", "50% Flowering", "Establishment", "Frost damage", "Anthesis yearday"
]

df = df[columns]

# === Drop rows where Frost damage is missing ===
df = df.dropna(subset=["Frost damage"])

# === Convert target to integer and cast as categorical ===
df["Frost damage"] = df["Frost damage"].astype(int)

# === Handle missing values ===
for col in df.select_dtypes(include=["float64", "int64"]).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# === Encode dates ===
df["SowingDate"] = pd.to_datetime(df["SowingDate"], errors='coerce')
df["HarvestDate"] = pd.to_datetime(df["HarvestDate"], errors='coerce')
df["SowingDate"] = (df["SowingDate"] - df["SowingDate"].min()).dt.days
df["HarvestDate"] = (df["HarvestDate"] - df["HarvestDate"].min()).dt.days

# === Encode categorical features ===
categorical_cols = [
    "Name", "State", "MET Analysis Mega Region", "RegionName",
    "TrialCode", "SiteDescription", "CultivarID"
]
df = pd.get_dummies(df, columns=categorical_cols)

# === Prepare features and target ===
X = df.drop(columns=["Frost damage"])
y = df["Frost damage"] - 1

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Build XGBoost classifier ===
model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=4,  # 4 classes: 1, 2, 3, 4
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42
)

# === Train the model ===
model.fit(X_train, y_train)

# === Predict ===
y_pred = model.predict(X_test).astype(int) + 1
y_test_actual = y_test+1

# === Evaluation ===
print("Classification Report:")
print(classification_report(y_test_actual, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test_actual, y_pred))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Confusion matrix (y_test and y_pred should be in original 1–4 scale)
cm = confusion_matrix(y_test_actual, y_pred)

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Frost Damage Levels")
plt.tight_layout()
plt.show()
