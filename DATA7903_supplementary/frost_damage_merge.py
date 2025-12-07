import pandas as pd
import glob

# # Step 1: Get all frost damage file paths (assuming they're in the 'Clean' folder)
# file_paths = glob.glob("Clean/frost_damage_value_*.csv")

# # Step 2: Read and store all DataFrames in a list
# dfs = [pd.read_csv(file) for file in file_paths]

# # Step 3: Concatenate all DataFrames into one
# merged_df = pd.concat(dfs, ignore_index=True)

# # Optional: Check the result
# print(merged_df['Frost damage'].value_counts())
# print(merged_df.shape)
# # Export the merged DataFrame to a CSV file
# merged_df.to_csv("merged_frost_damage.csv", index=False)

df = pd.read_csv("Clean/filtered_file.csv")
# Assuming your monthly rainfall columns are named exactly like this:
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create the new column by summing across the monthly columns
df['TotalRainfall'] = df[months].sum(axis=1)

df.to_csv("Clean/filtered_with_total_rainfall.csv", index=False)