@echo off
title Bank Fraud-Alert Console Runner
echo ======================================================================
echo  An Intelligent System to Flag Anomalous Card Transactions in Near Real Time
echo  Logistic Regression with SMOTE Balancing Pipeline
echo ======================================================================
echo.

echo [1/2] Starting Flask Backend API on http://127.0.0.1:5000 ...
start "Backend Flask API" cmd /k "py backend\app.py"

echo [2/2] Starting React Vite Frontend on http://localhost:3000 ...
cd frontend
start "Frontend React App" cmd /k "npm run dev -- --port 3000"

echo.
echo ======================================================================
echo  System is running!
echo  Open your browser at: http://localhost:3000
echo ======================================================================
pause
