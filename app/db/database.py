import sqlite3
from app.core.config import DB_DIR, DB_PATH

def get_connection() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS MRRecord (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,

            record_uuid TEXT UNIQUE NOT NULL,
            user_uuid TEXT NOT NULL DEFAULT 'local',

            analysis_status TEXT NOT NULL DEFAULT 'PENDING',

            cough_detected INTEGER,
            cough_confidence REAL,
            cough_threshold REAL NOT NULL DEFAULT 0.6,

            quality_score REAL,
            quality_fail_reason TEXT,
            final_score REAL,
            final_label TEXT,

            healthy_distance REAL,
            reference_dataset TEXT DEFAULT 'COUGHVID_HEALTHY',
            distance_metric TEXT DEFAULT 'MAHALANOBIS',
            reference_version TEXT DEFAULT 'healthy_ref_final.json',
            risk_level TEXT,

            report_text TEXT,

            model_version TEXT DEFAULT 'ANIMA_BEST_1',

            audio_duration_sec REAL,
            local_audio_path TEXT,

            sync_status TEXT NOT NULL DEFAULT 'LOCAL_ONLY',
            sync_retry_count INTEGER NOT NULL DEFAULT 0,
            last_sync_attempt_at TEXT,
            server_received_at TEXT,
            synced_at TEXT,

            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()

def test_db_connection() -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print("DB 연결 성공:", dict(result))