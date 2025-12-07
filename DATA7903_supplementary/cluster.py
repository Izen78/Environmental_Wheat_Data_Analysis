import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("merged_plot_data_with_weather.csv")

# Assuming df is the original dataframe that contains 'TrialCode' and 'Kg/Plot' columns
df_mean = df.groupby('TrialCode')['Kg/Plot'].mean().reset_index()

# Check the result of the grouping
print(df_mean.head())

# Scale the 'Kg/Plot' values (since 'TrialCode' is categorical, we will only scale 'Kg/Plot')
# scaler = StandardScaler()
# df_mean['Kg/Plot_Scaled'] = scaler.fit_transform(df_mean[['Kg/Plot']])
# df_mean = df_mean.dropna(subset=['Kg/Plot', 'Kg/Plot_Scaled'])
df_mean = df_mean.dropna(subset=['Kg/Plot'])
# Check the scaled data
print(df_mean.head())
from sklearn.cluster import DBSCAN

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=5)  # You may need to adjust eps and min_samples based on your data
df_mean['Cluster'] = dbscan.fit_predict(df_mean[['Kg/Plot']])

# Check the results of clustering
print(df_mean[['TrialCode', 'Kg/Plot', 'Cluster']].head())
# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df_mean['TrialCode'], df_mean['Kg/Plot'], c=df_mean['Cluster'], cmap='viridis', marker='o')
plt.title('DBSCAN Clusters Based on Average Kg/Plot per TrialCode')
plt.xlabel('TrialCode')
plt.ylabel('Average Kg/Plot')
plt.colorbar(label='Cluster')
plt.xticks(rotation=90)  # Rotate the x-axis labels if they overlap
plt.show()

# Ensure 'Kg/Plot' and weather variables are present in the dataset
df = df.dropna(subset=['Kg/Plot', 'Avg_Temp_Max', 'Total_Rain', 'Avg_RHMaxT', 'Height (cm)', 'Total_Evap'])  # Drop rows where 'Kg/Plot' or weather variables are missing

# Select relevant columns (including 'Kg/Plot' and weather features)
weather_data = df[['Avg_Temp_Max', 'Total_Rain', 'Avg_RHMaxT', 'Height (cm)', 'Total_Evap']]
kg_plot = df[['Kg/Plot']]

# Concatenate 'Kg/Plot' with the weather features
data_for_clustering = pd.concat([weather_data, kg_plot], axis=1)

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_clustering)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(data_scaled)

# Visualize clusters with Kg/Plot on the y-axis
plt.scatter(df['Total_Rain'], df['Kg/Plot'], c=df['Cluster'], cmap='viridis')
plt.title('Clustering Based on Total Rain and Kg/Plot')
plt.xlabel('Total Rain')
plt.ylabel('Kg/Plot')
plt.colorbar(label='Cluster')
plt.show()

# Print top clusters with their average Kg/Plot
cluster_means = df.groupby('Cluster')['Kg/Plot'].mean()
print(cluster_means)