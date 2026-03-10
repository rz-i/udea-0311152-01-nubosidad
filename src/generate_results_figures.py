import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the figures directory exists
os.makedirs('paper/figures', exist_ok=True)

# 1. Load your data (Update the path to your actual CSV file)
validation_df = pd.read_csv('data/final_validation_dataset.csv')

# --- FIGURE 1: Histogram (Data Distribution) ---
plt.figure(figsize=(8, 5))
sns.histplot(validation_df['mean_index'], bins=30, kde=True, color='teal')
plt.title('Distribución del Sky Index')
plt.xlabel('Sky Index')
plt.ylabel('Frecuencia')
plt.savefig("paper/figures/hist_sky_index.png", dpi=300)
plt.close()

# --- FIGURE 2: Boxplot (Integrity/Outliers) ---
plt.figure(figsize=(8, 5))
sns.boxplot(y=validation_df['mean_index'], color='skyblue')
plt.title('Detección de Outliers (IQR)')
plt.ylabel('Sky Index')
plt.savefig("paper/figures/boxplot_outliers.png", dpi=300)
plt.close()

# --- FIGURE 3: Scatter Plot (The Final Correlation) ---
plt.figure(figsize=(8, 6))
sns.regplot(x='mean_index', y='sat_radiance', data=validation_df, 
            scatter_kws={'alpha':0.5, 'color': 'teal'}, 
            line_kws={'color':'red'})
plt.title('Correlación: Sky Index vs Irradiancia Satelital')
plt.xlabel('Sky Index (Smartphone)')
plt.ylabel('Irradiancia Satelital (W/m²)')
plt.savefig("paper/figures/scatter_correlation.png", dpi=300)
plt.close()

# 1. Asegurar que el timestamp sea formato fecha
validation_df['timestamp'] = pd.to_datetime(validation_df['timestamp'])

# 2. Extraer la hora
validation_df['hora'] = validation_df['timestamp'].dt.hour

# 3. Crear el Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='hora', y='mean_index', data=validation_df, palette='viridis')

plt.title('Variabilidad Diurna del Sky Index')
plt.xlabel('Hora del día')
plt.ylabel('Sky Index')
plt.savefig("paper/figures/diurnal_variability.png", dpi=300)

print("Figures successfully generated in the paper/figures folder.")