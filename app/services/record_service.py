import uuid
from datetime import datetime
from typing import Optional

from app.db.database import get_connection

def _to_int_bool(value) -> Optional[int]:
    if value is None:
        return None
    return 1 if bool(value) else 0

def _decide_analysis_status(mr_record: dict) -> str:
    confidence = mr_record.get("cough_confidence")
    detected = mr_record.get("cough_detected")

    if detected is False:
        return "NON_COUGH"
    
    if detected is True:
        return "ANALYZED"
    
    if confidence is None:
        return "FAILED"
    
    return "PENDING"

def save_mr_record(
    mr_record: dict,
    user_uuid: Optional[str] = None,
    local_audio_path: Optional[str] = None,
) -> dict:
    
    record_uuid = str(uuid.uuid4())
    server_received_at = datetime.now().isoformat(timespec="seconds")

    analysis_status = _decide_analysis_status(mr_record)

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO MRRecord (
                record_uuid,
                user_uuid,
                analysis_status,
                cough_detected,
                cough_confidence,
                cough_threshold,
                quality_score,
                quality_fail_reason,
                baseline_similarity_score,
                baseline_deviation_score,
                report_text,
                model_version,
                embedding_version,
                audio_duration_sec,
                local_audio_path,
                server_received_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_uuid,
                user_uuid,
                analysis_status,
                _to_int_bool(mr_record.get("cough_detected")),
                mr_record.get("cough_confidence"),
                mr_record.get("cough_threshold"),
                mr_record.get("quality_score"),
                mr_record.get("quality_fail_reason"),
                mr_record.get("baseline_similarity_score"),
                mr_record.get("baseline_deviation_score"),
                mr_record.get("report_text"),
                mr_record.get("model_version"),
                mr_record.get("embedding_version"),
                mr_record.get("audio_duration_sec"),
                local_audio_path,
                server_received_at,
            ),
        )

        conn.commit()

        saved_id = cursor.lastrowid

        return {
            "record_id": saved_id,
            "record_uuid": record_uuid,
            "analysis_status": analysis_status,
            "server_received_at": server_received_at,
        }