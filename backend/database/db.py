import os
import json
import sqlite3
from datetime import datetime, timezone

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fraud_detection.db')
DATABASE_URL = os.environ.get('DATABASE_URL')

def get_connection():
    """Returns database connection (SQLite by default, or PostgreSQL if configured)."""
    if DATABASE_URL and DATABASE_URL.startswith('postgres'):
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        except Exception as e:
            print(f"[DB Warning] Could not connect to PostgreSQL ({e}). Falling back to SQLite.")
    
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes tables for transactions, alerts, and system configuration."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT UNIQUE NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        amount REAL NOT NULL,
        time_sec REAL NOT NULL,
        features_json TEXT,
        risk_score REAL NOT NULL,
        is_fraud INTEGER NOT NULL,
        ground_truth INTEGER,
        severity TEXT NOT NULL,
        status TEXT DEFAULT 'FLAGGED',
        latency_ms REAL DEFAULT 0,
        top_factors_json TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        severity TEXT NOT NULL,
        risk_score REAL NOT NULL,
        status TEXT DEFAULT 'PENDING',
        notes TEXT,
        resolved_at DATETIME,
        FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS system_config (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Default settings
    default_configs = [
        ('decision_threshold', '0.50'),
        ('auto_block_critical', 'true'),
        ('stream_speed_ms', '1500')
    ]
    for key, val in default_configs:
        cursor.execute('''
        INSERT OR IGNORE INTO system_config (key, value) VALUES (?, ?)
        ''', (key, val))

    conn.commit()
    conn.close()

def log_transaction(data):
    """Logs a scored transaction into the database and generates an alert if anomalous."""
    conn = get_connection()
    cursor = conn.cursor()
    
    features_json = json.dumps(data.get('features', {}))
    top_factors_json = json.dumps(data.get('top_factors', []))
    
    cursor.execute('''
    INSERT OR REPLACE INTO transactions 
    (transaction_id, timestamp, amount, time_sec, features_json, risk_score, is_fraud, ground_truth, severity, status, latency_ms, top_factors_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['transaction_id'],
        data.get('timestamp', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')),
        data['amount'],
        data.get('time_sec', 0.0),
        features_json,
        data['risk_score'],
        1 if data['is_fraud'] else 0,
        data.get('ground_truth'),
        data['severity'],
        data.get('status', 'FLAGGED' if data['is_fraud'] else 'APPROVED'),
        data.get('latency_ms', 0.0),
        top_factors_json
    ))

    # If flagged, automatically register an alert record
    if data['is_fraud']:
        cursor.execute('''
        INSERT INTO alerts (transaction_id, created_at, severity, risk_score, status, notes)
        VALUES (?, CURRENT_TIMESTAMP, ?, ?, 'PENDING', ?)
        ''', (
            data['transaction_id'],
            data['severity'],
            data['risk_score'],
            f"Auto-flagged with risk score {data['risk_score']*100:.1f}% ({data['severity']})"
        ))

    conn.commit()
    conn.close()

def get_recent_transactions(limit=100, status_filter=None, fraud_only=False):
    """Retrieves recent transactions with optional filters."""
    conn = get_connection()
    cursor = conn.cursor()

    query = 'SELECT * FROM transactions WHERE 1=1'
    params = []

    if status_filter:
        query += ' AND status = ?'
        params.append(status_filter)

    if fraud_only:
        query += ' AND is_fraud = 1'

    query += ' ORDER BY id DESC LIMIT ?'
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        d = dict(row)
        if d.get('features_json'):
            try:
                d['features'] = json.loads(d['features_json'])
            except:
                d['features'] = {}
        if d.get('top_factors_json'):
            try:
                d['top_factors'] = json.loads(d['top_factors_json'])
            except:
                d['top_factors'] = []
        results.append(d)
    return results

def get_stats():
    """Computes real-time telemetry stats for the dashboard."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM transactions')
    total_tx = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM transactions WHERE is_fraud = 1')
    flagged_tx = cursor.fetchone()[0]

    cursor.execute('SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE is_fraud = 1')
    fraud_amount = cursor.fetchone()[0]

    cursor.execute('SELECT COALESCE(SUM(amount), 0) FROM transactions')
    total_amount = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM alerts WHERE status = "PENDING"')
    pending_alerts = cursor.fetchone()[0]

    cursor.execute('SELECT AVG(latency_ms) FROM transactions')
    avg_latency = cursor.fetchone()[0] or 0.0

    conn.close()

    fraud_rate = (flagged_tx / total_tx * 100) if total_tx > 0 else 0.0

    return {
        'total_transactions': total_tx,
        'flagged_anomalies': flagged_tx,
        'fraud_amount_at_risk': round(fraud_amount, 2),
        'total_volume': round(total_amount, 2),
        'pending_alerts': pending_alerts,
        'fraud_rate_pct': round(fraud_rate, 2),
        'avg_latency_ms': round(avg_latency, 2)
    }

def update_transaction_status(transaction_id, new_status, notes=None):
    """Updates status for review (e.g. 'CONFIRMED_FRAUD', 'FALSE_POSITIVE', 'APPROVED')."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
    UPDATE transactions SET status = ? WHERE transaction_id = ?
    ''', (new_status, transaction_id))

    cursor.execute('''
    UPDATE alerts SET status = ?, notes = COALESCE(?, notes), resolved_at = CURRENT_TIMESTAMP
    WHERE transaction_id = ?
    ''', (new_status, notes, transaction_id))

    conn.commit()
    conn.close()
    return True

def get_config(key, default=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM system_config WHERE key = ?', (key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else default

def set_config(key, value):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO system_config (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
    ''', (key, str(value)))
    conn.commit()
    conn.close()

# Auto-initialize database on import
init_db()
