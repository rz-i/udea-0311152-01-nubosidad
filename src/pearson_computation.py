import pandas as pd
from scipy.stats import pearsonr

# 1. Load the data
ground = pd.read_csv('data/segmented_metrics_cleaned.csv')
sat = pd.read_csv('data/satellite_data.csv')

# 2. Convert and sort (Crucial for merge_asof)
ground['timestamp'] = pd.to_datetime(ground['timestamp'])
sat['timestamp'] = pd.to_datetime(sat['timestamp'])

ground = ground.sort_values('timestamp')
sat = sat.sort_values('timestamp')

# 3. Perform the Merge
# We map each ground observation to the satellite record within a 30-min window
validation_df = pd.merge_asof(
    ground, 
    sat, 
    on='timestamp', 
    direction='nearest', 
    tolerance=pd.Timedelta('30min')
)

# 4. Remove mismatches (where no satellite data was found within the window)
validation_df = validation_df.dropna(subset=['sat_radiance'])

# 5. Calculate Pearson Correlation
corr, p_value = pearsonr(validation_df['mean_index'], validation_df['sat_radiance'])

print(f"\n--- Validation Results ---")
print(f"Total matched observations: {len(validation_df)}")
print(f"Pearson Correlation (r): {corr:.3f}")
print(f"P-value: {p_value:.4e}")

# Save the final dataset for the team
validation_df.to_csv('data/final_validation_dataset.csv', index=False)
print("\n[SUCCESS] Final dataset saved to 'data/final_validation_dataset.csv'")