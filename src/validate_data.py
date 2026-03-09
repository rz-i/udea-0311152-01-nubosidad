import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

def analyze_data(df, interval_min=10):
    print("--- Diagnostic Report ---")

    # 1. Integrity Check (Step 1)
    print("\nIntegrity Check:")
    print(f"Total records: {len(df)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    if df.isnull().values.any():
        print("ALERT: Missing values detected in the CSV.")
    
    # 2. Duplicates
    dupes = df[df.duplicated(subset=['timestamp', 'direction'], keep=False)]
    if not dupes.empty:
        print(f"\n[ALERT] {len(dupes)} duplicate records found:")
        print(dupes[['timestamp', 'direction']].head(10))
    
    # 3. Outliers (IQR Method)
    Q1 = df['mean_index'].quantile(0.25)
    Q3 = df['mean_index'].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df['mean_index'] < lower) | (df['mean_index'] > upper)]
    print(f"\n[INFO] Statistical Outliers (IQR method): {len(outliers)} entries")
    print(f"Bounds: [{lower:.4f}, {upper:.4f}]")
    print(outliers[['timestamp', 'direction']])

    # 4. Time Gaps
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df['diff'] = df['timestamp'].diff()
    threshold = pd.Timedelta(minutes=interval_min * 2)
    gaps = df[df['diff'] > threshold]
    print(f"\n[INFO] Detected {len(gaps)} gaps > {interval_min * 2} minutes.")
    if not gaps.empty:
        print("Largest gaps:")
        print(gaps[['timestamp', 'diff']].sort_values('diff', ascending=False).head(5))

    # 4. Summary Report
    print("\nSummary Report:")
    print(f"Mean Sky Index: {df['mean_index'].mean():.3f}")
    print(f"Index per direction:\n{df.groupby('direction')['mean_index'].mean()}")
    print(f"\nStats per direction:\n{df.groupby('direction')['mean_index'].describe()}")
    
    # 5. Distribution Check (Quick Plausibility)
    print("\nDistribution Summary:")
    print(df['mean_index'].describe())


    # 5. Histogram for Plausibility
    plt.figure(figsize=(8, 5))
    plt.hist(df['mean_index'].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of Mean Sky Index")
    plt.xlabel("Sky Index Value")
    plt.ylabel("Frequency")
    plt.savefig('data/index_distribution.png')
    plt.show()

def clean_data(df):
    print("\n[ACTION] Cleaning data...")
    # Remove duplicates, keeping the first occurrence
    df = df.drop_duplicates(subset=['timestamp', 'direction'])
    # Remove IQR outliers
    Q1, Q3 = df['mean_index'].quantile(0.25), df['mean_index'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['mean_index'] >= Q1 - 1.5 * IQR) & (df['mean_index'] <= Q3 + 1.5 * IQR)]
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Clean data and save as new file")
    args = parser.parse_args()

    df = pd.read_csv('data/segmented_metrics.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if args.clean:
        df_clean = clean_data(df)
        df_clean.to_csv('data/segmented_metrics_cleaned.csv', index=False)
        print("Done! Saved to data/segmented_metrics_cleaned.csv")
    else:
        analyze_data(df)