import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# Harvest Length, Harvest Width, Kg/Plot, GrainYield_gm2, Residual?? vs. Trial
# Create a model that determines if a wheat is going to take some sort of damage
# Create a model that predicts yield
# Visualise yield
# Descriptive Statistics
# Visualise variability within a trial
# Simulated HTP and GxE
# Cluster trials with similar characteristics
# Time-series data
# Compare simulated data with real data for anomaly detection (alerts)
df = pd.read_excel("Plot_Level/2015-2019 Wheat Main Plot Data (All traits)_anonymized.xlsx")
# print(df.head)
summary_stats = df.groupby("TrialCode")["Harvest Length"].describe()
# print(summary_stats)
# summary_stats.to_csv("summary_stats.csv")  # Open in Excel or a text editor

top_50_trials = df.groupby("TrialCode")["Harvest Length"].mean().nlargest(50).reset_index()
# plt.figure(figsize=(14, 6))
# sns.scatterplot(x="TrialCode", y="Harvest Length", data=top_50_trials, color="red", s=100)
# plt.xticks(rotation=90)
# plt.title("Top 50 Trials with the Greatest Mean Harvest Length")
# plt.xlabel("Trial Code")
# plt.ylabel("Mean Harvest Length")
# plt.show()

summary_stats_width = df.groupby("TrialCode")["Harvest Width"].describe()
summary_stats_kg_plot = df.groupby("TrialCode")["Kg/Plot"].describe()
summary_stats_yield = df.groupby("TrialCode")["GrainYield_gm2"].describe()

# Optionally, print out the summary stats to check
# print(summary_stats_width)
# print(summary_stats_kg_plot)
# print(summary_stats_yield)
# Top 50 trials by mean Harvest Width
top_50_trials_width = df.groupby("TrialCode")["Harvest Width"].mean().nlargest(50).reset_index()

# Top 50 trials by mean Kg/Plot
top_50_trials_kg_plot = df.groupby("TrialCode")["Kg/Plot"].mean().nlargest(50).reset_index()

# Top 50 trials by mean Grain Yield (gm²)
top_50_trials_yield = df.groupby("TrialCode")["GrainYield_gm2"].mean().nlargest(50).reset_index()
# Scatter plot for Top 50 Trials by mean Harvest Width
# plt.figure(figsize=(14, 6))
# sns.scatterplot(x="TrialCode", y="Harvest Width", data=top_50_trials_width, color="blue", s=100)
# plt.xticks(rotation=90)
# plt.title("Top 50 Trials with the Greatest Mean Harvest Width")
# plt.xlabel("Trial Code")
# plt.ylabel("Mean Harvest Width")
# plt.show()

# # Scatter plot for Top 50 Trials by mean Kg/Plot
# plt.figure(figsize=(14, 6))
# sns.scatterplot(x="TrialCode", y="Kg/Plot", data=top_50_trials_kg_plot, color="green", s=100)
# plt.xticks(rotation=90)
# plt.title("Top 50 Trials with the Greatest Mean Kg/Plot")
# plt.xlabel("Trial Code")
# plt.ylabel("Mean Kg/Plot")
# plt.show()

# # Scatter plot for Top 50 Trials by mean GrainYield_gm2
# plt.figure(figsize=(14, 6))
# sns.scatterplot(x="TrialCode", y="GrainYield_gm2", data=top_50_trials_yield, color="orange", s=100)
# plt.xticks(rotation=90)
# plt.title("Top 50 Trials with the Greatest Mean Grain Yield (gm²)")
# plt.xlabel("Trial Code")
# plt.ylabel("Mean Grain Yield (gm²)")
# plt.show()





############## SowingDate vs. Harvest Length ####################


# Convert to datetime
df["SowingDate"] = pd.to_datetime(df["SowingDate"])
df["HarvestDate"] = pd.to_datetime(df["HarvestDate"])

# Calculate Harvest Length in days
df["Harvest Length"] = (df["HarvestDate"] - df["SowingDate"]).dt.days



# plt.figure(figsize=(14, 6))
# sns.boxplot(x=df["SowingDate"].dt.strftime('%Y-%m-%d'), y=df["Harvest Length"], hue=df["TrialCode"], showfliers=False)

# plt.title("Boxplot of Sowing Date vs Harvest Length (Grouped by Trial)")
# plt.xlabel("Sowing Date")
# plt.ylabel("Harvest Length (days)")
# plt.xticks(rotation=45)
# plt.legend(title="Trial Code", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.grid()
# plt.show()



################ Comparing Simulation Data with Actual Data ################




################ Gathering all data affected by Frost Damage ################
# Filter the data for rows where FrostDamage is 4
frost_damage_4_df = df[df["Frost damage"] == 4]

# Display the filtered data
# print(frost_damage_4_df)
frost_damage_4_df.to_csv("frost_damage_4_data.csv", index=False)

################ Year vs. Harvest Length ################

# Convert SowingDate to datetime if it's not already
df["SowingDate"] = pd.to_datetime(df["SowingDate"])

# Extract Year from SowingDate
df["Year"] = df["SowingDate"].dt.year

# Group by Year and calculate the mean of Harvest Length for each year
yearly_avg_harvest_length = df.groupby("Year")["Harvest Length"].mean().reset_index()

# Display the result
print(yearly_avg_harvest_length)

# Plot Year vs Average Harvest Length
# plt.figure(figsize=(10, 6))
# sns.lineplot(x="Year", y="Harvest Length", data=yearly_avg_harvest_length, marker="o", color="blue")

# plt.title("Year vs Average Harvest Length")
# plt.xlabel("Year")
# plt.ylabel("Average Harvest Length (days)")
# plt.grid(True)
# plt.show()

################ MET Analysis Mega Region vs. Harvest Length ################
# Group by MegaRegion and calculate the mean of Harvest Length for each region
region_avg_harvest_length = df.groupby("MET Analysis Mega Region")["Harvest Length"].mean().reset_index()

# Display the result to check the average Harvest Length for each region
print(region_avg_harvest_length)

# Set the figure size for the plot
# plt.figure(figsize=(10, 6))

# Plot MegaRegion vs Average Harvest Length using seaborn
# sns.barplot(x="MET Analysis Mega Region", y="Harvest Length", data=region_avg_harvest_length, palette="viridis")

# # Add titles and labels
# plt.title("MET Analysis Mega Region vs Average Harvest Length")
# plt.xlabel("Mega Region")
# plt.ylabel("Average Harvest Length")
# plt.grid(True)

# Show the plot
# plt.show()




####################### DATA CLEANING #######################
print("DATA CLEANING PART")
print(df.isnull().sum())  # Count missing values per column
print(df.info())

# 99499 Entries
# Calculate percentage of non-null values for each column
non_null_percentage = (df.notnull().sum() / len(df)) * 100

# Convert to a DataFrame for better readability
non_null_df = pd.DataFrame(non_null_percentage, columns=["Non-Null Percentage"])

# Display the result
non_null_df = non_null_df.sort_values(by=["Non-Null Percentage"], ascending=True)

non_null_df.to_csv("non_null_perc1.csv")