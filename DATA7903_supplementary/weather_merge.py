import pandas as pd
import glob
import os

# Step 1: Load your main dataframe
main_df = pd.read_excel('Plot_Level/2015-2019 Wheat Main Plot Data (All traits)_anonymized.xlsx')

# Step 2: Find all weather files
weather_files = glob.glob('Weather_data/SILO_Point_Data/*.csv')

# Step 3: Create empty list for summarized weather
weather_summary_list = []

# Step 4: Read and summarize each weather file
for file in weather_files:
    try:
        print(f"📄 Reading {file}")
        weather = pd.read_csv(file, sep=r'\s+', skiprows=list(range(31)) + [32])
        print(weather.columns)
        # print(weather.head)

        # Extract the trial id from the filename
        basename = os.path.basename(file)
        trial_id = basename.split('_')[1]  # Gets the number after "trial_"

        # Summarize important weather info
        weather_summary = {
            'TrialCode': int(trial_id),
            'Avg_Temp_Max': weather['T.Max'].mean(),
            'Avg_Temp_Min': weather['T.Min'].mean(),
            'Total_Rain': weather['Rain'].sum(),
            'Total_Evap': weather['Evap'].sum(),
            'Avg_Radn': weather['Radn'].mean(),
            'Avg_VP': weather['VP'].mean(),
            'Avg_RHMaxT': weather['RHmaxT'].mean(),
            'Avg_RHMinT': weather['RHminT'].mean()
        }
        weather_summary_list.append(weather_summary)

    except Exception as e:
        print(f"❌ Error reading {file}: {e}")

# Step 5: Create weather summary DataFrame
weather_summary_df = pd.DataFrame(weather_summary_list)
main_df['TrialCode'] = main_df['TrialCode'].astype(str)
weather_summary_df['TrialCode'] = weather_summary_df['TrialCode'].astype(str)
weather_summary_df['TrialCode'] = weather_summary_df['TrialCode'].str.replace('trial_', '').astype(str)

weather_summary_df['TrialCode'] = 'trial_' + weather_summary_df['TrialCode'].astype(str)
# For main_df, ensure TrialCode is in string format (same format as weather)
main_df['TrialCode'] = main_df['TrialCode'].astype(str)

# Now, perform the merge
merged_df = main_df.merge(weather_summary_df, on='TrialCode', how='left')

# Check if the temperature columns are now populated
print(merged_df.head())
print(main_df['TrialCode'].unique()[:10])
print(weather_summary_df['TrialCode'].unique()[:10])
# Step 6: Merge with main dataframe
merged_df = main_df.merge(weather_summary_df, on='TrialCode', how='left')

# Step 7: Save to CSV
merged_df.to_csv('merged_plot_data_with_weather.csv', index=False)

print("✅ Merge complete! Saved as merged_plot_data_with_weather.csv")
