# An Intelligent System to Flag Anomalous Card Transactions in Near Real Time

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Imbalanced-Learn SMOTE](https://img.shields.io/badge/Balancing-SMOTE-green.svg)](https://imbalanced-learn.org/)
[![Flask REST API](https://img.shields.io/badge/Backend-Flask-black.svg)](https://flask.palletsprojects.com/)
[![React + Vite](https://img.shields.io/badge/Frontend-React%2018%20+%20Vite-61dafb.svg)](https://vitejs.dev/)
[![Database](https://img.shields.io/badge/DB-SQLite%20%7C%20PostgreSQL-indigo.svg)](https://www.sqlite.org/)

---

## 📌 Project Abstract

> This study designs a **Logistic Regression model with SMOTE balancing** to flag anomalous card transactions in near real time, addressing the problem of credit card fraud detection through a data-driven approach. The pipeline draws on anonymized transaction datasets, applying suitable pre-processing before training and testing the model on held-out data. The project culminates in a **bank fraud-alert console**, bridging the gap between the underlying model and everyday operational use.

---

## 🏛️ System Architecture

```
                                  [ Transaction Stream ]
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │   Flask REST API (/api/predict) │
                           └────────────────┬────────────────┘
                                            │
                      ┌─────────────────────┼─────────────────────┐
                      ▼                     ▼                     ▼
          ┌──────────────────────┐┌──────────────────────┐┌──────────────────────┐
          │ StandardScaler Engine││  SMOTE-Trained LR    ││  Explainability XAI  │
          │ (Time, Amount, V1-28)││  Sigmoid Inference   ││  (Key PCA Anomalies) │
          └──────────────────────┘└─────────┬────────────┘└──────────────────────┘
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │     Dynamic Decision Engine     │
                           │   (τ = 0.50 Custom Cutoff)      │
                           └────────────────┬────────────────┘
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      ▼                                           ▼
          ┌──────────────────────┐                    ┌──────────────────────┐
          │   Flagged / Alert    │                    │  Approved / Normal   │
          │ (High/Critical Risk) │                    │      (Low Risk)      │
          └───────────┬──────────┘                    └───────────┬──────────┘
                      │                                           │
                      └─────────────────────┬─────────────────────┘
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │ Database Layer (SQLite/Postgres)│
                           │  - Persistent Transaction Logs  │
                           │  - Alert Audit & Case History   │
                           │  - Real-Time Telemetry Stats    │
                           └────────────────┬────────────────┘
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │   React Fraud-Alert Dashboard   │
                           │  - Live Real-Time Stream Feed   │
                           │  - Interactive Anomaly Sandbox  │
                           │  - Model & SMOTE Metrics View   │
                           │  - Historical Case Audit Log    │
                           └─────────────────────────────────┘
```

---

## 🔬 Machine Learning Pipeline & Methodology

### 1. The Imbalanced Data Problem
In standard credit card datasets (e.g. European Cardholder benchmark), fraudulent events make up approximately **0.17% – 0.50%** of all transactions. Naive classifiers achieve 99.8% accuracy simply by classifying *everything* as legitimate, while missing 100% of the actual fraud.

### 2. SMOTE (Synthetic Minority Over-sampling Technique)
To solve this, our pipeline applies **SMOTE** to synthesize minority class instances along feature vectors between $k$-nearest fraud neighbors:
$$x_{\text{new}} = x_i + \lambda (x_{zi} - x_i), \quad \lambda \sim U(0, 1)$$

> **CRITICAL DATA INTEGRITY GUARANTEE (No Data Leakage):**
> SMOTE is applied **strictly to the training partition** after stratified splitting ($80/20$). The held-out test set remains completely untouched and representative of the true natural transaction distribution.

### 3. Logistic Regression Classifier
Logistic Regression models the posterior probability of fraud using the sigmoid function:
$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

A dynamic decision threshold $\tau$ (calibrated at $0.50$, tunable via the dashboard) maps continuous probabilities to operational decisions:
$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1 \mid \mathbf{x}) \ge \tau \\ 0 & \text{otherwise} \end{cases}$$

### 4. Evaluation Metrics
- **Recall (Detection Rate):** $\frac{TP}{TP + FN}$ — Captures the fraction of fraudulent transactions intercepted.
- **Precision:** $\frac{TP}{TP + FP}$ — Quantifies accuracy among all flagged transactions.
- **PR-AUC (Precision-Recall Area Under Curve):** The definitive metric for heavily imbalanced fraud data.
- **Confusion Matrix:** Tracks True Negatives ($TN$), False Positives ($FP$), False Negatives ($FN$), and True Positives ($TP$).

---

## 📊 Analytical Visualizations Generated

The training pipeline automatically computes and exports high-resolution evaluation figures located in `backend/static_reports/`:
1. `confusion_matrix.png`: Heatmap of held-out test predictions.
2. `precision_recall_curve.png`: PR curve comparing SMOTE vs baseline unweighted models.
3. `roc_curve.png`: True positive rate vs false positive rate across all discrimination thresholds.
4. `smote_comparison.png`: Direct metric-by-metric comparison highlighting the substantial recall boost delivered by SMOTE.

---

## ⚡ Near Real-Time Capabilities

- **Sub-5ms Inference Latency:** Preprocessed feature vectors execute through the trained model in **$1.8 - 4.5\text{ ms}$**.
- **Live Stream Simulator:** Cycles through realistic held-out test transactions and synthesizes dynamic payment events to demonstrate continuous live monitoring.
- **Explainable AI (XAI):** Automatically pinpoints the top 3 contributing factors for any flagged transaction (e.g. negative velocity shifts on $V_{12}$/$V_{14}$, abnormal amount spikes, or rapid auth retries).

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.10+** (verified compatible with standard Python and Windows `py` launcher)
- **Node.js 18+** & **npm**

### Option A: One-Click Launch (Windows)
Double-click `run_system.bat` (or execute `.\run_system.ps1` in PowerShell). This launches both:
- Flask REST API on `http://127.0.0.1:5000`
- React Vite Dashboard on `http://localhost:3000`

### Option B: Manual Setup

#### 1. Setup Backend & Train Model
```powershell
# In project root:
py -m pip install -r backend/requirements.txt

# Run ML pipeline to generate data, train model, and create report plots:
py backend/model/train_model.py

# Launch Flask backend:
py backend/app.py
```

#### 2. Setup Frontend
```powershell
# In another terminal:
cd frontend
npm install
npm run dev -- --port 3000
```

Open your browser to: **`http://localhost:3000`**

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health check and model loading status |
| `GET` | `/api/stats` | Telemetry: total volume, flagged count, fraud rate, mean latency |
| `GET` | `/api/metadata` | Model performance metrics, confusion matrix, top features |
| `GET` | `/api/transactions` | Query recent transactions (`?limit=50&fraud_only=true`) |
| `POST` | `/api/predict` | Score single transaction vector in real-time |
| `POST` | `/api/simulate/next` | Score the next live transaction from benchmark stream |
| `POST` | `/api/transactions/<tx_id>/review` | Update case status (`CONFIRMED_FRAUD`, `FALSE_POSITIVE`) |
| `GET/POST`| `/api/config` | Read or adjust decision threshold $\tau$ (0.10 - 0.90) |
| `GET` | `/api/reports/<filename>` | Serve static Matplotlib/Seaborn analytical plots |
| `POST` | `/api/reset_demo` | Reset and reseed live demonstration stream |

---

## 🗄️ Database Integration

- **Default Engine:** Unified SQLite (`backend/database/fraud_detection.db`), requiring zero setup or local server maintenance.
- **Enterprise PostgreSQL Support:** Seamlessly supported via the standard connection string:
  ```powershell
  $env:DATABASE_URL="postgresql://user:password@localhost:5432/fraud_db"
  ```
