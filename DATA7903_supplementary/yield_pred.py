# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# import statsmodels.api as sm

# Load the dataset
# df = pd.read_csv("merged_plot_data_with_weather.csv", low_memory=False)

# # Features (the last 6 weather columns)
# X = df[['Avg_Temp_Max', 'Avg_Temp_Min', 'Total_Rain', 'Total_Evap', 'Avg_Radn', 'Avg_VP', 'Avg_RHMaxT', 'Avg_RHMinT']]

# # Target variable
# y = df['Kg/Plot']

# # Drop rows where either X or y have missing values
# data = pd.concat([X, y], axis=1).dropna()
# X = data[['Avg_Temp_Max', 'Avg_Temp_Min', 'Total_Rain', 'Total_Evap', 'Avg_Radn', 'Avg_VP', 'Avg_RHMaxT', 'Avg_RHMinT']]
# y = data['Kg/Plot']
# X = sm.add_constant(X)
# # Split into train and test sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Fit the linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predict on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Model coefficients:", model.coef_)
# print("Model intercept:", model.intercept_)
# print(f"Mean Squared Error (MSE): {mse:.2f}")
# print(f"R² score: {r2:.2f}")

# NOT GOOD 0.3 R^2
# model2 = sm.OLS(y, X).fit()
# print(model2.summary()) 


# Load your data (with the warning fixed too)
# df = pd.read_csv("merged_plot_data_with_weather.csv", low_memory=False)

# # List of variables you want
# features = ['Avg_Temp_Max', 'Avg_Temp_Min', 'Total_Rain', 'Total_Evap', 'Avg_Radn', 'Avg_VP', 'Avg_RHMaxT', 'Avg_RHMinT',
#             'Harvest Length', 'Harvest Width', 'CultivarID']

# # Keep only the necessary columns
# df_model = df[features + ['Kg/Plot']].copy()

# # Convert all numeric columns (except CultivarID) to numeric types
# for col in ['Avg_Temp_Max', 'Avg_Temp_Min', 'Total_Rain', 'Total_Evap', 'Avg_Radn', 'Avg_VP', 
#             'Avg_RHMaxT', 'Avg_RHMinT', 'Harvest Length', 'Harvest Width', 'Kg/Plot']:
#     df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

# # Create dummy variables for CultivarID
# df_model = pd.get_dummies(df_model, columns=['CultivarID'], drop_first=True)
# for col in df_model.columns:
#     if df_model[col].dtype == 'bool':
#         df_model[col] = df_model[col].astype(int)
# # Drop missing values
# df_model = df_model.dropna()

# # Separate predictors and target
# X = df_model.drop('Kg/Plot', axis=1)
# y = df_model['Kg/Plot']

# # Add constant
# X = sm.add_constant(X)
# print(X.dtypes)
# print(X.head())

# # Fit the model
# model = sm.OLS(y, X).fit()

# # Summary
# print(model.summary())

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load your data
df = pd.read_csv("merged_plot_data_with_weather.csv")
df = df.dropna(subset=['Kg/Plot'])
# Step 1: One-hot encode categorical features (like CultivarID)
df_encoded = pd.get_dummies(df, drop_first=True)  # drop_first avoids multicollinearity

# Step 2: Define your target variable (Kg/Plot) and features (all other columns)
X = df_encoded.drop(columns=['Kg/Plot', 'GrainYield_gm2'])  # All columns except target
y = df_encoded['Kg/Plot']  # Target variable

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Step 7: Feature Importance (optional but useful)
importances = rf_model.feature_importances_
indices = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("\nTop 10 important features:")
print(indices.head(10))
