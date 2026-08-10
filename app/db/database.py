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
        CREATE TABLE IF NOT EXISTS User (
            user_uuid TEXT PRIMARY KEY,

            device_os TEXT,

            user_id TEXT UNIQUE,
            password_hash TEXT,
            recovery_email TEXT,
            account_linked_at TEXT,

            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS UserProfile (
            user_uuid TEXT PRIMARY KEY NOT NULL,
        
            age INTEGER,
            gender TEXT,
            country TEXT,
            region TEXT,
            language TEXT,
            visit_purpose TEXT,
            first_visit INTEGER,
        
            smoking TEXT,
            allergy INTEGER,
            allergy_detail TEXT,
            dust_exposure INTEGER,
            has_pet INTEGER,
            living_environment TEXT,
        
            symptoms TEXT,
            symptom_duration TEXT,
            respiratory_conditions TEXT,
            recent_infection INTEGER,
            medication TEXT,
            note TEXT,
        
            consented_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        
            FOREIGN KEY (user_uuid) REFERENCES User(user_uuid)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS MRRecord (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,

            record_uuid TEXT UNIQUE NOT NULL,
            user_uuid TEXT NOT NULL DEFAULT 'local',
            source_device_type TEXT,

            measured_at TEXT,
            record_date TEXT,
            client_app_version TEXT,

            analysis_status TEXT NOT NULL DEFAULT 'PENDING',

            cough_detected INTEGER,
            cough_confidence REAL,
            cough_threshold REAL NOT NULL DEFAULT 0.6,

            quality_score REAL,
            quality_fail_reason TEXT,
            final_score REAL,
            final_label TEXT,   

            healthy_distance REAL,
            embedding_vector TEXT,
            reference_dataset TEXT DEFAULT 'COUGHVID_HEALTHY',
            distance_metric TEXT DEFAULT 'MAHALANOBIS',
            reference_version TEXT DEFAULT 'healthy_ref_final.json',
            risk_level TEXT,

            report_text TEXT,

            model_version TEXT DEFAULT 'ANIMA_BEST_1',

            audio_duration_sec REAL,
            local_audio_path TEXT,
                       
            audio_storage_namespace TEXT,
            audio_storage_bucket TEXT,
            audio_storage_key TEXT,
            audio_storage_uploaded_at TEXT,
            audio_storage_error TEXT,
            consent_to_store INTEGER NOT NULL DEFAULT 1,

            sync_status TEXT NOT NULL DEFAULT 'LOCAL_ONLY',
            sync_retry_count INTEGER NOT NULL DEFAULT 0,
            last_sync_attempt_at TEXT,
            server_received_at TEXT,
            synced_at TEXT,

            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("PRAGMA table_info(MRRecord)")
        columns = {row["name"] for row in cursor.fetchall()}

        if "source_device_type" not in columns:
            cursor.execute("""
                ALTER TABLE MRRecord
                ADD COLUMN source_device_type TEXT
            """)

        if "measured_at" not in columns:
            cursor.execute("""
                ALTER TABLE MRRecord
                ADD COLUMN measured_at TEXT
            """)
        
        if "record_date" not in columns:
            cursor.execute("""
                ALTER TABLE MRRecord
                ADD COLUMN record_date TEXT
            """)
        
        if "client_app_version" not in columns:
            cursor.execute("""
                ALTER TABLE MRRecord
                ADD COLUMN client_app_version TEXT
            """)

        conn.commit()

def test_db_connection() -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print("DB 연결 성공:", dict(result))