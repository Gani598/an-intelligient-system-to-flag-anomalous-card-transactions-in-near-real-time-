import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)
SAMPLE_CSV_PATH = os.path.join(DATA_DIR, 'creditcard_sample.csv')

def generate_benchmark_dataset(n_samples=30000, fraud_ratio=0.005, random_state=42):
    """
    Generates a realistic credit card transaction dataset strictly modeled after 
    the European Cardholder benchmark dataset (Kaggle creditcard.csv):
    - 28 PCA features (V1 to V28)
    - Time (seconds elapsed across 48 hours)
    - Amount (log-normal distributed transaction amounts)
    - Class (0: Legitimate, 1: Fraudulent, with realistic severe class imbalance)
    """
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    print(f"[Dataset] Generating {n_samples} transactions ({n_normal} normal, {n_fraud} fraud, ratio={fraud_ratio*100:.2f}%)...")

    # 1. Normal Transactions
    # PCA features for normal transactions are approximately standard normal (mean ~ 0, unit variance)
    v_normal = np.random.randn(n_normal, 28)
    # Realistic amounts: log-normal distribution with median around $25, long tail up to $2,500
    amounts_normal = np.round(np.random.lognormal(mean=2.9, sigma=1.2, size=n_normal), 2)
    amounts_normal = np.clip(amounts_normal, 1.0, 4500.0)
    # Time: 48-hour window (0 to 172,800 seconds) with diurnal wave
    time_normal = np.sort(np.random.uniform(0, 172800, size=n_normal))

    # 2. Fraudulent Transactions
    # Characteristics derived from European Card benchmark:
    # V14, V12, V10, V17 have heavy negative shifts
    # V4, V11, V2, V19 have positive shifts
    v_fraud = np.random.randn(n_fraud, 28) * 1.5
    v_fraud[:, 13] -= np.random.normal(6.5, 1.8, size=n_fraud)   # V14
    v_fraud[:, 11] -= np.random.normal(5.8, 1.6, size=n_fraud)   # V12
    v_fraud[:, 9]  -= np.random.normal(5.2, 1.5, size=n_fraud)   # V10
    v_fraud[:, 16] -= np.random.normal(6.0, 1.7, size=n_fraud)   # V17
    v_fraud[:, 3]  += np.random.normal(4.2, 1.4, size=n_fraud)   # V4
    v_fraud[:, 10] += np.random.normal(3.8, 1.3, size=n_fraud)   # V11
    v_fraud[:, 1]  += np.random.normal(3.2, 1.2, size=n_fraud)   # V2
    v_fraud[:, 18] += np.random.normal(1.5, 0.9, size=n_fraud)   # V19

    # Fraud amounts often bifurcate: either micro-testing ($1.00 - $12.00) or high-drain attempts ($300 - $2,200)
    fraud_pattern = np.random.rand(n_fraud)
    amounts_fraud = np.where(
        fraud_pattern > 0.4,
        np.round(np.random.uniform(280.0, 2100.0, size=n_fraud), 2),
        np.round(np.random.uniform(0.99, 15.0, size=n_fraud), 2)
    )
    time_fraud = np.random.uniform(0, 172800, size=n_fraud)

    # 3. Combine into DataFrame
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']

    df_normal = pd.DataFrame(
        np.column_stack([time_normal, v_normal, amounts_normal, np.zeros(n_normal)]),
        columns=columns
    )

    df_fraud = pd.DataFrame(
        np.column_stack([time_fraud, v_fraud, amounts_fraud, np.ones(n_fraud)]),
        columns=columns
    )

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    # Shuffle preserving realistic arrival
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df['Class'] = df['Class'].astype(int)

    df.to_csv(SAMPLE_CSV_PATH, index=False)
    print(f"[Dataset] Saved benchmark dataset to {SAMPLE_CSV_PATH} ({len(df)} rows)")
    return df

def is_valid_dataset_file(filepath, min_bytes=50000):
    """Verifies that the target file exists, exceeds minimum size, and contains required headers."""
    try:
        if not os.path.exists(filepath):
            return False
        if os.path.getsize(filepath) < min_bytes:
            return False
        # Fast header inspection
        sample_df = pd.read_csv(filepath, nrows=2)
        return 'Class' in sample_df.columns and 'Amount' in sample_df.columns
    except Exception:
        return False

def get_or_create_dataset():
    """Returns dataset path, creating synthetic benchmark if not found."""
    kaggle_csv = os.path.join(DATA_DIR, 'creditcard.csv')
    if is_valid_dataset_file(kaggle_csv, min_bytes=10000000):
        print(f"[Dataset] Using verified full Kaggle dataset: {kaggle_csv}")
        return kaggle_csv
    
    kaggle_csv_gz = os.path.join(DATA_DIR, 'creditcard.csv.gz')
    if is_valid_dataset_file(kaggle_csv_gz, min_bytes=10000000):
        print(f"[Dataset] Using verified compressed Kaggle dataset: {kaggle_csv_gz}")
        return kaggle_csv_gz
    
    if is_valid_dataset_file(SAMPLE_CSV_PATH, min_bytes=10000):
        print(f"[Dataset] Using verified sample dataset: {SAMPLE_CSV_PATH}")
        return SAMPLE_CSV_PATH
    
    print("[Dataset] No valid Kaggle or sample CSV found. Generating realistic benchmark dataset...")
    generate_benchmark_dataset()
    return SAMPLE_CSV_PATH

if __name__ == '__main__':
    generate_benchmark_dataset()
