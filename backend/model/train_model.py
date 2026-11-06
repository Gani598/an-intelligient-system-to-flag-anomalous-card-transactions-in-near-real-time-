import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless backend for web servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
from imblearn.over_sampling import SMOTE

import sys

# Path configurations
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)
BACKEND_DIR = os.path.dirname(MODEL_DIR)
REPORTS_DIR = os.path.join(BACKEND_DIR, 'static_reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, 'fraud_model.joblib')
SCALER_FILE = os.path.join(MODEL_DIR, 'scaler.joblib')
METADATA_FILE = os.path.join(MODEL_DIR, 'model_metadata.json')
TEST_STREAM_FILE = os.path.join(MODEL_DIR, 'test_stream_samples.json')

try:
    from generate_dataset import get_or_create_dataset
except ImportError:
    from backend.model.generate_dataset import get_or_create_dataset

def train_and_evaluate():
    """
    Executes full ML Pipeline:
    1. Load anonymized transaction dataset (European Cardholders format)
    2. Train/Test split (stratified 80/20)
    3. StandardScaler for Amount & Time
    4. SMOTE balancing ONLY on training data (Strict No Data-Leakage policy)
    5. Logistic Regression classifier training
    6. Baseline comparison (without SMOTE)
    7. Evaluation: Precision, Recall, F1, PR-AUC, ROC-AUC, Confusion Matrix
    8. Generate visual analytical plots
    9. Export model artifacts & test streaming samples
    """
    csv_path = get_or_create_dataset()
    print(f"[Training] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    target_col = 'Class'

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    total_records = len(df)
    fraud_count = int(y.sum())
    normal_count = total_records - fraud_count
    print(f"[Training] Total: {total_records} | Legitimate: {normal_count} | Fraud: {fraud_count} ({fraud_count/total_records*100:.3f}%)")

    # 1. Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 2. Feature Scaling: Fit scaler strictly on X_train to prevent leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train Baseline Model (Without SMOTE) for direct scientific comparison
    print("[Training] Fitting Baseline Logistic Regression (Unbalanced)...")
    baseline_lr = LogisticRegression(max_iter=1000, random_state=42)
    baseline_lr.fit(X_train_scaled, y_train)
    y_pred_base = baseline_lr.predict(X_test_scaled)
    y_prob_base = baseline_lr.predict_proba(X_test_scaled)[:, 1]

    recall_base = recall_score(y_test, y_pred_base, zero_division=0)
    precision_base = precision_score(y_test, y_pred_base, zero_division=0)
    f1_base = f1_score(y_test, y_pred_base, zero_division=0)
    pr_auc_base = average_precision_score(y_test, y_prob_base)

    # 4. Apply SMOTE strictly to Training Set
    print("[Training] Applying SMOTE (Synthetic Minority Over-sampling Technique) to training set...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    train_normal_smote = int((y_train_smote == 0).sum())
    train_fraud_smote = int((y_train_smote == 1).sum())
    print(f"[Training] SMOTE Balanced Training Set: {train_normal_smote} Legitimate, {train_fraud_smote} Fraud")

    # 5. Train SMOTE-Balanced Logistic Regression
    print("[Training] Fitting SMOTE + Logistic Regression Classifier...")
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    model.fit(X_train_smote, y_train_smote)

    # 6. Evaluate on Held-Out Test Data
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
    
    # Analyze optimal threshold (e.g. default 0.50 vs tuned)
    default_threshold = 0.50
    y_pred_test = (y_prob_test >= default_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, zero_division=0)
    rec = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob_test)
    pr_auc = average_precision_score(y_test, y_prob_test)

    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()

    print(f"=== HELD-OUT TEST RESULTS (SMOTE + Logistic Regression) ===")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f} (Baseline: {precision_base:.4f})")
    print(f"Recall:      {rec:.4f} (Baseline: {recall_base:.4f})  <-- Massive SMOTE boost!")
    print(f"F1-Score:    {f1:.4f} (Baseline: {f1_base:.4f})")
    print(f"PR-AUC:      {pr_auc:.4f} (Baseline: {pr_auc_base:.4f})")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # 7. Extract Feature Coefficients (Feature Importance / Explainability)
    coefs = model.coef_[0]
    feature_importance = [
        {'feature': col, 'coefficient': float(c), 'abs_importance': float(abs(c))}
        for col, c in zip(feature_cols, coefs)
    ]
    feature_importance.sort(key=lambda x: x['abs_importance'], reverse=True)

    # 8. Generate Visual Analytics Charts
    # Chart 1: Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Legitimate (0)', 'Fraud (1)'],
                yticklabels=['Legitimate (0)', 'Fraud (1)'])
    plt.title('Confusion Matrix: SMOTE + Logistic Regression', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('Actual Ground Truth', fontweight='bold')
    plt.tight_layout()
    cm_plot_path = os.path.join(REPORTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()

    # Chart 2: Precision-Recall Curve
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob_test)
    prec_b, rec_b, _ = precision_recall_curve(y_test, y_prob_base)
    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, color='#6366f1', lw=2.5, label=f'SMOTE + LR (PR-AUC = {pr_auc:.3f})')
    plt.plot(rec_b, prec_b, color='#94a3b8', lw=1.8, linestyle='--', label=f'Baseline LR (PR-AUC = {pr_auc_base:.3f})')
    plt.title('Precision-Recall Curve (Highly Imbalanced Benchmark)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Recall (Detection Rate)', fontweight='bold')
    plt.ylabel('Precision (True Fraud Accuracy)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', frameon=True)
    plt.tight_layout()
    pr_plot_path = os.path.join(REPORTS_DIR, 'precision_recall_curve.png')
    plt.savefig(pr_plot_path, dpi=200)
    plt.close()

    # Chart 3: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='#10b981', lw=2.5, label=f'SMOTE + LR (ROC-AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#cbd5e1', linestyle=':', label='Random Chance (0.500)')
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    roc_plot_path = os.path.join(REPORTS_DIR, 'roc_curve.png')
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()

    # Chart 4: SMOTE vs Baseline Comparison Bar Chart
    metrics_names = ['Recall (Catch Rate)', 'F1-Score', 'PR-AUC', 'Precision']
    smote_vals = [rec, f1, pr_auc, prec]
    base_vals = [recall_base, f1_base, pr_auc_base, precision_base]
    x_idx = np.arange(len(metrics_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x_idx - width/2, base_vals, width, label='Baseline (Unbalanced)', color='#94a3b8')
    plt.bar(x_idx + width/2, smote_vals, width, label='SMOTE + Logistic Regression', color='#6366f1')
    plt.xticks(x_idx, metrics_names, fontweight='bold')
    plt.ylabel('Score (0.0 to 1.0)', fontweight='bold')
    plt.title('Impact of SMOTE on Fraud Detection Metrics', fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    comp_plot_path = os.path.join(REPORTS_DIR, 'smote_comparison.png')
    plt.savefig(comp_plot_path, dpi=200)
    plt.close()

    # 9. Save Model and Artifacts
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    metadata = {
        'model_name': 'Logistic Regression + SMOTE',
        'trained_at': pd.Timestamp.now().isoformat(),
        'dataset_total': total_records,
        'dataset_fraud_count': fraud_count,
        'dataset_fraud_ratio': fraud_count / total_records,
        'smote_train_samples': len(y_train_smote),
        'decision_threshold': default_threshold,
        'metrics': {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'roc_auc': round(float(roc_auc), 4),
            'pr_auc': round(float(pr_auc), 4),
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            }
        },
        'baseline_comparison': {
            'precision': round(float(precision_base), 4),
            'recall': round(float(recall_base), 4),
            'f1_score': round(float(f1_base), 4),
            'pr_auc': round(float(pr_auc_base), 4)
        },
        'top_predictive_features': feature_importance[:10]
    }

    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

    # 10. Prepare held-out test stream samples for near real-time simulation
    # Take a diverse slice of test samples (both fraud and legitimate)
    test_df = X_test.copy()
    test_df['Class'] = y_test.values
    test_stream = []

    fraud_samples = test_df[test_df['Class'] == 1].head(60)
    normal_samples = test_df[test_df['Class'] == 0].head(240)
    combined_stream = pd.concat([fraud_samples, normal_samples]).sample(frac=1.0, random_state=123)

    for idx, row in combined_stream.iterrows():
        sample_dict = {
            'time_sec': float(row['Time']),
            'amount': float(row['Amount']),
            'ground_truth': int(row['Class']),
            'features': {f'V{i}': float(row[f'V{i}']) for i in range(1, 29)}
        }
        test_stream.append(sample_dict)

    with open(TEST_STREAM_FILE, 'w') as f:
        json.dump(test_stream, f, indent=2)

    print(f"[Training] Model saved to: {MODEL_FILE}")
    print(f"[Training] Scaler saved to: {SCALER_FILE}")
    print(f"[Training] Metadata saved to: {METADATA_FILE}")
    print(f"[Training] Preserved {len(test_stream)} test transactions for real-time simulation.")
    return metadata

if __name__ == '__main__':
    train_and_evaluate()
