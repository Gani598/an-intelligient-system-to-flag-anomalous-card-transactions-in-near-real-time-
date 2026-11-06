Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " An Intelligent System to Flag Anomalous Card Transactions in Near Real Time" -ForegroundColor Green
Write-Host " Logistic Regression with SMOTE Balancing Pipeline" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan

# Start Backend
Write-Host "`n[1/2] Starting Flask Backend API on http://127.0.0.1:5000 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "py backend\app.py"

# Start Frontend
Write-Host "[2/2] Starting React Vite Frontend on http://localhost:3000 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev -- --port 3000"

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " System running! Access dashboard at: http://localhost:3000" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
