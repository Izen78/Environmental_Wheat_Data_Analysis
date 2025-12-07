import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import zscore

df = pd.read_excel("Plot_Level/2015-2019 Wheat Main Plot Data (All traits)_anonymized.xlsx")

frost_damage_1 = df[df["Frost damage"] == 1]
frost_damage_2 = df[df["Frost damage"] == 2]
frost_damage_3 = df[df["Frost damage"] == 3]
frost_damage_4 = df[df["Frost damage"] == 4]

# Export to CSV
frost_damage_1.to_excel("frost_damage_value_1.xlsx", index=False)
frost_damage_2.to_excel("frost_damage_value_2.xlsx", index=False)
frost_damage_3.to_excel("frost_damage_value_3.xlsx", index=False)
frost_damage_4.to_excel("frost_damage_value_4.xlsx", index=False)