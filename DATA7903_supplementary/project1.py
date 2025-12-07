# import pandas as pd
# import seaborn as sns

# merged_df = pd.read_csv("filtered_file.csv")
# merged_df['TotalRainfall'] = merged_df[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']].sum(axis=1)

# top_cultivars = merged_df['CultivarID'].value_counts().head(10).index
# sns.lmplot(data=merged_df[merged_df['CultivarID'].isin(top_cultivars)],
#            x='TotalRainfall', y='GrainYield_gm2', hue='CultivarID', lowess=True)
# cv_df = merged_df.groupby('CultivarID')['GrainYield_gm2'].agg(['mean', 'std'])
# cv_df['CV'] = cv_df['std'] / cv_df['mean']
# resistant_cultivars = cv_df.sort_values('CV')  # Lower CV = more stable
# import statsmodels.formula.api as smf
# predictors = [
#     'TrialCode', 'TotalRainfall', 'MinDepth', 'MaxDepth', 'Boron', 'Conductivity_(EC)',
#     'Copper', 'Exchangeable_Ca', 'Exchangeable_K', 'Exchangeable_Mg',
#     'Exchangeable_Na', 'Manganese', 'Organic_Carbon', 'pH_(CaCl2)',
#     'pH_(water)', 'Phosphorous', 'Potassium_(K)', 'Sulphur', 'Texture',
#     'Zinc'
# ]


# merged_df['Rainfall_z'] = (merged_df['TotalRainfall'] - merged_df['TotalRainfall'].mean()) / merged_df['TotalRainfall'].std()

# # model = smf.ols('GrainYield_gm2 ~ C(CultivarID)*Rainfall_z', data=merged_df).fit()
# # Wrap predictors with Q() if they contain special characters
# def safe_var(varname):
#     return f'Q("{varname}")' if any(c in varname for c in "() ") else varname

# safe_predictors = [safe_var(p) for p in predictors]

# # Final formula
# formula = f'GrainYield_gm2 ~ C(CultivarID)*Rainfall_z + {" + ".join(safe_predictors)}'

# # Fit the model
# model = smf.ols(formula=formula, data=merged_df).fit()
# print(model.summary())


# summary_df = model.summary2().tables[0]
# print(model.rsquared)

# Filter out only cultivar rows with positive coefficients and p < 0.05
# filtered = summary_df[
#     (summary_df.index.str.contains('C\(CultivarID\)')) &  # only CultivarID rows
#     (summary_df['Coef.'] > 0) &                           # positive effect
#     (summary_df['P>|t|'] < 0.05)                          # statistically significant
# ]

# # Sort from highest to lowest positive effect
# sorted_filtered = filtered.sort_values(by='Coef.', ascending=False)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# Show the result
# print(sorted_filtered[['Coef.', 'P>|t|']])

# from statsmodels.stats.outliers_influence import variance_inflation_factor
# from patsy import dmatrices
# import pandas as pd
# merged_df = pd.read_csv("filtered_file.csv")
# print(merged_df.columns)
# merged_df['TotalRainfall'] = merged_df[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']].sum(axis=1)
# # Suppose you're interested in these predictors:
# predictors = [
#     'TotalRainfall', 'MinDepth', 'MaxDepth', 'Boron', 'Conductivity_(EC)',
#        'Copper', 'Exchangeable_Ca', 'Exchangeable_K', 'Exchangeable_Mg',
#        'Exchangeable_Na', 'Manganese', 'Organic_Carbon', 'pH_(CaCl2)',
#        'pH_(water)', 'Phosphorous', 'Potassium_(K)', 'Sulphur', 'Texture',
#        'Zinc'   # replace with your actual features
# ]

# # Drop rows with missing values in those columns
# vif_data = merged_df[predictors].dropna()

# # Optional: Check for constant columns
# print(vif_data.nunique())

# # Create design matrix
# from statsmodels.tools.tools import add_constant
# X = add_constant(vif_data)

# # Calculate VIF
# vif = pd.DataFrame()
# vif["Variable"] = X.columns
# vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# print(vif)

# import matplotlib.pyplot as plt
# import seaborn as sns
# import statsmodels.formula.api as smf

# model = smf.ols('GrainYield_gm2 ~ C(CultivarID)*TotalRainfall', data=merged_df).fit()
# # Predict values and residuals
# fitted_vals = model.fittedvalues
# residuals = model.resid

# sns.residplot(x=fitted_vals, y=residuals, lowess=True, line_kws={'color': 'red'})
# plt.xlabel("Fitted values")
# plt.ylabel("Residuals")
# plt.title("Residuals vs. Fitted")
# plt.axhline(0, linestyle='--', color='gray')
# plt.show()

import pandas as pd

df = pd.read_csv("Clean/frost_damage_value_1.csv")
print(df.columns)
print(df.head)