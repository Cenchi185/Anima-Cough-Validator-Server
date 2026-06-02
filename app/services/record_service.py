import uuid
from datetime import datetime
from typing import Optional

from app.db.database import get_connection


def _to_int_bool(value) -> Optional[int]:
    if value is None:
        return None
    return 1 if bool(value) else 0


def _decide_analysis_status(mr_record: dict) -> str:
    detected = mr_record.get("cough_detected")
    fail_reason = mr_record.get("quality_fail_reason")

    if detected is False or detected == 0:
        return "NON_COUGH"
    if detected is True or detected == 1:
        return "ANALYZED"
    if fail_reason:
        return "FAILED"

    return "PENDING"

def save_mr_record(
    mr_record: dict,
    user_uuid: Optional[str] = None,
    # local_audio_path: Optional[str] = None,
) -> dict:
    '''mr_record 를 DB 에 저장하기'''
    record_uuid = mr_record.get("record_uuid") or str(uuid.uuid4()) # 기록 uuid
    now = datetime.now().isoformat(timespec="seconds") # 분석 기록 일자
    analysis_status = _decide_analysis_status(mr_record) # 기침 분석 결과 저장(분석 되었는지 실패했는지)

    with get_connection() as conn: # db 연결 (임시)
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
                final_score,
                final_label,
                healthy_distance,
                reference_dataset,
                distance_metric,
                reference_version,
                risk_level,
                report_text,
                model_version,
                audio_duration_sec,
                local_audio_path,
                audio_storage_namespace,
                audio_storage_bucket,
                audio_storage_key,
                audio_storage_uploaded_at,
                audio_storage_error,
                consent_to_store,
                server_received_at,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_uuid,
                user_uuid or "local",
                analysis_status,
                _to_int_bool(mr_record.get("cough_detected")),
                mr_record.get("cough_confidence"),
                mr_record.get("cough_threshold"),
                mr_record.get("quality_score"),
                mr_record.get("quality_fail_reason"),
                mr_record.get("final_score"),
                mr_record.get("final_label"),
                mr_record.get("healthy_distance"),
                mr_record.get("reference_dataset", "COUGHVID_HEALTHY"),
                mr_record.get("distance_metric", "MAHALANOBIS"),
                mr_record.get("reference_version", "healthy_ref_final.json"),
                mr_record.get("risk_level"),
                mr_record.get("report_text"),
                mr_record.get("model_version", "ANIMA_BEST_1"),
                mr_record.get("audio_duration_sec"),
                None,
                mr_record.get("audio_storage_namespace"),
                mr_record.get("audio_storage_bucket"),
                mr_record.get("audio_storage_key"),
                mr_record.get("audio_storage_uploaded_at"),
                mr_record.get("audio_storage_error"),
                1,
                now,
                now,
                now,
            ),
        )

        conn.commit()
        saved_id = cursor.lastrowid

    return {
        "record_id": saved_id,
        "record_uuid": record_uuid,
        "analysis_status": analysis_status,
        "server_received_at": now,
    }