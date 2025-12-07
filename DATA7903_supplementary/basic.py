import pandas as pd

# Load the Excel file (first sheet by default)
# file_path = 'Weather_data/SILO_Point_Data/trial_716_SILOweather.csv'
# df = pd.read_csv(file_path, skiprows = list(range(31))+[32], sep='\s+')

file_path = 'Simulation/apsim_out_daily_anonymized.csv'
df = pd.read_csv(file_path )
rows, columns = df.shape

print(f"Number of rows: {rows}")
print(f"Number of columns: {columns}")

# Preview the first few rows
print("🔹 First 5 rows:")
print(df.head())

# Check column names
print("\n🔹 Column names:")
print(df.columns.tolist())

# Get basic info about each column (types, non-null count)
print("\n🔹 Dataset info:")
print(df.info())

# Describe numeric columns
print("\n🔹 Summary statistics (numerical columns):")
print(df.describe())

# Check for missing values
print("\n🔹 Missing values per column:")
print(df.isnull().sum())

# Look at unique values in categorical columns (optional)
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"\n🔹 Unique values in '{col}':")
    print(df[col].unique()[:10])  # Show only first 10 unique values
