import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def plot_iqr_outliers(data, column_name):
    plt.figure(figsize=(10, 6))
    
    # Calculate IQR
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]
    
    # Create the boxplot
    sns.boxplot(y=data[column_name], color='skyblue')
    
    # Overlay outliers as red dots
    plt.scatter(x=[0]*len(outliers), y=outliers[column_name], color='red', label='Outliers (Removed)')
    
    plt.title(f'Detección de outliers en {column_name} (Método IQR)')
    plt.legend()
    plt.savefig("paper/figures/iqr_outliers.png")
    plt.show()

def analyze_data(df, interval_min=10):
    """
    Performs a dry-run diagnostic of the data health without making changes.
    """
    print("--- Data Diagnostic Report (Pre-Cleaning) ---")

    # 1. Integrity Check
    print(f"\n[INFO] Total records found: {len(df)}")
    
    # 2. Duplicate Detection
    dupes = df[df.duplicated(subset=['timestamp', 'direction'], keep=False)]
    if not dupes.empty:
        print(f"\n[ALERT] {len(dupes)} records identified for merging (same timestamp & direction).")
    
    # 3. Outlier Identification (IQR Method)
    Q1 = df['mean_index'].quantile(0.25)
    Q3 = df['mean_index'].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df['mean_index'] < lower) | (df['mean_index'] > upper)]
    print(f"\n[INFO] Statistical outliers detected: {len(outliers)}")
    print(f"Logical Bounds: [{lower:.4f} to {upper:.4f}]")

    plot_iqr_outliers(df, 'mean_index')

    # 4. Temporal Gap Analysis
    df = df.sort_values('timestamp')
    df['diff'] = df['timestamp'].diff()
    gap_threshold = pd.Timedelta(minutes=interval_min * 2)
    significant_gaps = df[df['diff'] > gap_threshold]
    print(f"\n[INFO] Data gaps > {interval_min * 2} min: {len(significant_gaps)}")
    if not significant_gaps.empty:
        print("Largest identified gaps:")
        print(significant_gaps[['timestamp', 'diff']].sort_values('diff', ascending=False).head(3))

def clean_and_merge(df):
    """
    Processes the data: averages duplicates, filters outliers, and removes low-quality masks.
    """
    print("\n[EXECUTION] Starting data cleaning and consolidation...")
    
    # 1. Merge Duplicates by Averaging
    # Using mean() on metrics reduces random observer error for simultaneous captures.
    initial_count = len(df)
    df_merged = df.groupby(['timestamp', 'direction']).agg({
        'observation_id': 'first', # Keep the first ID as reference
        'mean_index': 'mean',
        'std_dev': 'mean',
        'mask_pixel_count': 'mean'
    }).reset_index()
    print(f"-> Consolidated duplicates: reduced from {initial_count} to {len(df_merged)} records.")

    # 2. Apply Statistical and Quality Filters
    Q1 = df_merged['mean_index'].quantile(0.25)
    Q3 = df_merged['mean_index'].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    
    # Keep only records within IQR bounds AND with enough sky area (> 500 pixels)
    df_clean = df_merged[
        (df_merged['mean_index'] >= lower) & 
        (df_merged['mean_index'] <= upper) & 
        (df_merged['mask_pixel_count'] > 500)
    ].copy()
    
    print(f"-> Removed {len(df_merged) - len(df_clean)} records (outliers or insufficient mask area).")
    return df_clean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and clean sky segmentation metrics.")
    parser.add_argument("--clean", action="store_true", help="Merge duplicates and save cleaned file.")
    args = parser.parse_args()

    # Data Loading
    try:
        data_path = 'data/segmented_metrics.csv'
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Ensure the segmentation script has finished.")
        exit(1)
    
    if args.clean:
        df_final = clean_and_merge(df)
        output_path = 'data/segmented_metrics_cleaned.csv'
        df_final.to_csv(output_path, index=False)
        print(f"\n[SUCCESS] Analysis-ready file saved: '{output_path}'")
        
        # Generate final distribution plot for the paper
        plt.figure(figsize=(8, 5))
        plt.hist(df_final['mean_index'], bins=20, color='royalblue', edgecolor='white', alpha=0.8)
        plt.title("Final Sky Index Distribution (Averaged & Cleaned)")
        plt.xlabel("Sky Index (Blue / [Green + Red])")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.savefig('data/final_distribution.png')
        print("-> Distribution histogram saved to 'data/final_distribution.png'")
    else:
        analyze_data(df)