import os
import pandas as pd
from sklearn.datasets import fetch_openml

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)
REAL_CSV_PATH = os.path.join(DATA_DIR, 'creditcard.csv')

def fetch_and_save_real_dataset():
    print("[Real Data] Fetching official Kaggle European Cardholders dataset via OpenML...")
    dataset = fetch_openml('CreditCardFraudDetection', version=1, as_frame=True, parser='auto')
    df = dataset.frame

    # Ensure standard column naming and types
    df['Class'] = df['Class'].astype(int)
    
    print(f"[Real Data] Total Real Transactions: {len(df):,}")
    print(f"[Real Data] Total Real Fraud Cases: {df['Class'].sum():,} ({(df['Class'].sum()/len(df))*100:.3f}%)")
    print(f"[Real Data] Columns: {list(df.columns)}")

    print(f"[Real Data] Saving genuine dataset to: {REAL_CSV_PATH} ...")
    df.to_csv(REAL_CSV_PATH, index=False)
    print(f"[Real Data] Successfully saved {os.path.getsize(REAL_CSV_PATH)/(1024*1024):.1f} MB real CSV!")
    return REAL_CSV_PATH

if __name__ == '__main__':
    fetch_and_save_real_dataset()
