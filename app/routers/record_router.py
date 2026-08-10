from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime, date

from app.db.database import get_connection

import json

router = APIRouter(prefix="/records", tags=["records"])

class SyncRecord(BaseModel):
    record_uuid: str
    user_uuid: Optional[str] = "local"
    source_device_type: Optional[Literal["APP","WEB","WEARABLE"]] = None

    measured_at: Optional[datetime] = None
    record_date: Optional[date] = None
    client_app_version: Optional[str] = None

    analysis_status: Optional[str] = None

    cough_detected: Optional[bool] = None
    cough_confidence: Optional[float] = None
    cough_threshold: Optional[float] = None

    quality_score: Optional[float] = None
    final_score: Optional[float] = None
    final_label: Optional[str] = None
    quality_fail_reason: Optional[str] = None

    healthy_distance: Optional[float] = None
    embedding_vector: Optional[List[float]] = None

    reference_dataset: Optional[str] = "COUGHVID_HEALTHY"
    distance_metric: Optional[str] = "MAHALANOBIS"
    reference_version: Optional[str] = "healthy_ref_final.json"
    risk_level: Optional[str] = None

    report_text: Optional[str] = None
    model_version: Optional[str] = "ANIMA_BEST_1"

    audio_duration_sec: Optional[float] = None
    local_audio_path: Optional[str] = None

    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SyncRequest(BaseModel):
    user_uuid: str
    records: List[SyncRecord] = Field(default_factory=list)


def _to_int_bool(value):
    if value is None:
        return None
    return 1 if bool(value) else 0


def _now():
    return datetime.now().isoformat(timespec="seconds")


@router.post("/sync")
def sync_records(payload: SyncRequest):
    """
    앱 로컬 DB의 분석 기록을 서버 DB로 동기화.
    record_uuid 기준으로 이미 있으면 UPDATE, 없으면 INSERT.
    """
    if not payload.records:
        return {
            "ok": True,
            "synced_count": 0,
            "failed": [],
        }

    synced_count = 0
    failed = []
    now = _now()

    with get_connection() as conn:
        cursor = conn.cursor()

        for record in payload.records:
            try:
                measured_at = (
                    record.measured_at.isoformat()
                    if record.measured_at is not None
                    else None
                )
            
                record_date = (
                    record.record_date.isoformat()
                    if record.record_date is not None
                    else None
                )

                created_at = record.created_at or now
                updated_at = record.updated_at or now

                embedding_vector = None

                if record.embedding_vector is not None:
                    embedding_vector = json.dumps(record.embedding_vector)

                cursor.execute(
                    """
                    INSERT INTO MRRecord (
                        record_uuid,
                        user_uuid,
                        source_device_type,
                        measured_at,
                        record_date,
                        client_app_version,
                        analysis_status,
                        cough_detected,
                        cough_confidence,
                        cough_threshold,
                        quality_score,
                        final_score,
                        final_label,
                        quality_fail_reason,
                        healthy_distance,
                        embedding_vector,
                        reference_dataset,
                        distance_metric,
                        reference_version,
                        risk_level,
                        report_text,
                        model_version,
                        audio_duration_sec,
                        local_audio_path,
                        sync_status,
                        server_received_at,
                        synced_at,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(record_uuid) DO UPDATE SET
                        user_uuid = excluded.user_uuid,
                        source_device_type = excluded.source_device_type,
                        measured_at = excluded.measured_at,
                        record_date = excluded.record_date,
                        client_app_version = excluded.client_app_version,
                        analysis_status = excluded.analysis_status,
                        cough_detected = excluded.cough_detected,
                        cough_confidence = excluded.cough_confidence,
                        cough_threshold = excluded.cough_threshold,
                        quality_score = excluded.quality_score,
                        final_score = excluded.final_score,
                        final_label = excluded.final_label,
                        quality_fail_reason = excluded.quality_fail_reason,
                        healthy_distance = excluded.healthy_distance,
                        embedding_vector = excluded.embedding_vector,
                        reference_dataset = excluded.reference_dataset,
                        distance_metric = excluded.distance_metric,
                        reference_version = excluded.reference_version,
                        risk_level = excluded.risk_level,
                        report_text = excluded.report_text,
                        model_version = excluded.model_version,
                        audio_duration_sec = excluded.audio_duration_sec,
                        local_audio_path = excluded.local_audio_path,
                        sync_status = 'SYNCED',
                        synced_at = excluded.synced_at,
                        updated_at = excluded.updated_at
                    """,
                    (
                        record.record_uuid,
                        record.user_uuid or payload.user_uuid,
                        record.source_device_type,
                        measured_at,
                        record_date,
                        record.client_app_version,
                        record.analysis_status or "ANALYZED",
                        _to_int_bool(record.cough_detected),
                        record.cough_confidence,
                        record.cough_threshold,
                        record.quality_score,
                        record.final_score,
                        record.final_label,
                        record.quality_fail_reason,
                        record.healthy_distance,
                        embedding_vector,
                        record.reference_dataset,
                        record.distance_metric,
                        record.reference_version,
                        record.risk_level,
                        record.report_text,
                        record.model_version,
                        record.audio_duration_sec,
                        None,
                        "SYNCED",
                        now,
                        now,
                        created_at,
                        updated_at,
                    ),
                )

                synced_count += 1

            except Exception as e:
                failed.append({
                    "record_uuid": record.record_uuid,
                    "reason": str(e),
                })

        conn.commit()

    return {
        "ok": True,
        "synced_count": synced_count,
        "failed": failed,
    }

@router.get("")
def get_records(
    user_uuid: str,
    updated_after: Optional[str] = None,
):
    """
    서버 DB에 저장된 사용자 분석 기록 조회.
    앱 재설치/기기 변경 시 내려받기 용도.
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            if updated_after:
                cursor.execute(
                    """
                    SELECT *
                    FROM MRRecord
                    WHERE user_uuid = ?
                      AND updated_at > ?
                    ORDER BY created_at DESC
                    """,
                    (user_uuid, updated_after),
                )
            else:
                cursor.execute(
                    """
                    SELECT *
                    FROM MRRecord
                    WHERE user_uuid = ?
                    ORDER BY created_at DESC
                    """,
                    (user_uuid,),
                )

            rows = cursor.fetchall()

        return {
            "ok": True,
            "count": len(rows),
            "records": [dict(row) for row in rows],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))