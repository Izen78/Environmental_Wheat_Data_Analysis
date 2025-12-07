import pandas as pd

# --- Load your dataset ---
file_path = r"C:\Users\Arpon\Desktop\Uni\2025\DATA7901\Objective2\obj2_logistic_predictions_all.csv"
df = pd.read_csv(file_path)

# --- Identify columns ---
id_vars = [col for col in df.columns if not (
    col.startswith('SowingDate_') or 
    col.startswith('HarvestDate_') or 
    col.startswith('CultivarID_')
)]

sowing_cols = [col for col in df.columns if col.startswith("SowingDate_")]
harvest_cols = [col for col in df.columns if col.startswith("HarvestDate_")]
cultivar_cols = [col for col in df.columns if col.startswith("CultivarID_")]

# --- Melt SowingDate columns ---
df_sowing = df.melt(id_vars=id_vars, value_vars=sowing_cols,
                     var_name='SowingDate_col', value_name='SowingDate')
df_sowing = df_sowing[df_sowing['SowingDate'] == True].copy()
df_sowing['SowingDate'] = pd.to_datetime(df_sowing['SowingDate_col'].str.replace('SowingDate_', '').str.split().str[0])

# --- Melt HarvestDate columns ---
df_harvest = df.melt(id_vars=id_vars, value_vars=harvest_cols,
                      var_name='HarvestDate_col', value_name='HarvestDate')
df_harvest = df_harvest[df_harvest['HarvestDate'] == True].copy()
df_harvest['HarvestDate'] = pd.to_datetime(df_harvest['HarvestDate_col'].str.replace('HarvestDate_', '').str.split().str[0])

# --- Melt CultivarID columns ---
df_cultivar = df.melt(id_vars=id_vars, value_vars=cultivar_cols,
                       var_name='Cultivar_col', value_name='CultivarFlag')
df_cultivar = df_cultivar[df_cultivar['CultivarFlag'] == True].copy()
df_cultivar['CultivarID'] = df_cultivar['Cultivar_col'].str.replace('CultivarID_', '')

# --- Merge the melted dataframes ---
df_melted = df_sowing.merge(df_harvest, on=id_vars, how='inner')
df_melted = df_melted.merge(df_cultivar, on=id_vars, how='inner')

# --- Calculate DaysToHarvest ---
df_melted['DaysToHarvest'] = (df_melted['HarvestDate'] - df_melted['SowingDate']).dt.days

# --- Optional: check results ---
print(df_melted[['SowingDate', 'HarvestDate', 'DaysToHarvest', 'CultivarID']].head())

# --- Save cleaned dataset ---
output_file = r"C:\Users\Arpon\Desktop\Uni\2025\DATA7901\obj2_logistic_predictions_unpivoted.csv"
df_melted.to_csv(output_file, index=False)
