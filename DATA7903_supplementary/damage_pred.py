import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load all CSV files
files = [
    "Clean/frost_damage_value_1.csv",
    "Clean/frost_damage_value_2.csv",
    "Clean/frost_damage_value_3.csv",
    "Clean/frost_damage_4_data.csv"
]

dfs = [pd.read_csv(file) for file in files]

# Merge all the files into one dataframe
df = pd.concat(dfs, ignore_index=True)

# Check the first few rows of the dataset
print(df.head())

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Fill missing values for numeric columns with the median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill missing values for categorical columns with the mode (most frequent value)
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Drop columns that aren't needed or are irrelevant for the analysis
columns_to_drop = ["Anthesis yearday", "BBCH Growth Stage Score", "Bird or Animal damage", "Chemical Damage Score", 
                  "Crown Rot score", "Drought Severity Early", "Drought Severity Late", "Early Growth Score", 
                  "Establishment Plant count", "Falling Number - Plot Level - Rep 1", "Falling Number - Plot Level - Rep 2", 
                  "Frost damage - seed percent", "General Comments (DO NOT USE)", "Hail damage", "Head Loss %", 
                  "Head Loss Score", "Head Retention Score", "Head Tipping score", "Heading yearday", "Height (cm)", 
                  "Herbicide damage", "Leaf Rust score", "Lodging score", "Missing row length within plots (m)", 
                  "Number of Inside rows blocked", "Number of poor rows per plot", "Number of Rows per Plot", 
                  "Patchiness", "Patchiness %", "Rhizoctonia damage", "Septoria Tritici score", "Shattering %", 
                  "Standability", "Tillering Score", "Vigour", "Waterlogging score", "Weed Contamination", 
                  "Wheel track", "Yellow Leaf Spot score", "Zadoks score"]  # Example, add more columns as needed

df.drop(columns=columns_to_drop, inplace=True)

# Check for missing values after handling them
print(df.isnull().sum())

# Target: Frost damage (levels 1-4), ensure it's categorical
df['Frost damage'] = df['Frost damage'].astype('category')

# Separate features (X) and target (y)
X = df.drop(columns=['Frost damage'])
y = df['Frost damage']

# Split the dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply one-hot encoding to categorical columns
categorical_cols = ['Name', 'State', 'MET Analysis Mega Region', 'RegionName', 'TrialCode', 'SiteDescription', 'CultivarID']

# Perform one-hot encoding
X_train = pd.get_dummies(X_train, columns=categorical_cols)
X_test = pd.get_dummies(X_test, columns=categorical_cols)

# Align the training and testing sets to ensure they have the same columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Convert dates to datetime format
X_train['SowingDate'] = pd.to_datetime(X_train['SowingDate'], errors='coerce')
X_test['SowingDate'] = pd.to_datetime(X_test['SowingDate'], errors='coerce')

X_train['HarvestDate'] = pd.to_datetime(X_train['HarvestDate'], errors='coerce')
X_test['HarvestDate'] = pd.to_datetime(X_test['HarvestDate'], errors='coerce')

# Convert dates to number of days from the earliest date
X_train['SowingDate'] = (X_train['SowingDate'] - X_train['SowingDate'].min()).dt.days
X_test['SowingDate'] = (X_test['SowingDate'] - X_test['SowingDate'].min()).dt.days

X_train['HarvestDate'] = (X_train['HarvestDate'] - X_train['HarvestDate'].min()).dt.days
X_test['HarvestDate'] = (X_test['HarvestDate'] - X_test['HarvestDate'].min()).dt.days

# Initialize the classification model (Random Forest Classifier)
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model using classification metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
