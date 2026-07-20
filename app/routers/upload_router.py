from pathlib import Path
import uuid
import librosa

from app.core.config import UPLOAD_DIR, CONVERTED_DIR, MIN_DURATION, COUGH_THRESHOLD, TARGET_SR
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.services.audio_service import (
    probe_audio,
    convert_audio_to_wav,
    validate_audio,
    preprocess_for_model,
)
from app.services.prediction_service import predict
from app.services.record_service import save_mr_record
from app.services.score_service import (
    calculate_quality_score,
    calculate_final_score,
    get_final_prediction,
)

from datetime import datetime
from app.services.object_storage_service import (
    build_audio_object_name,
    upload_file_to_object_storage,
)

router = APIRouter()

# 음성파일 저장 폴더 생성
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

def safe_unlink(path:Path) -> None:
    ''' 파일 지우기 '''
    if path.exists():
        path.unlink()

def map_risk_level(stage2_level: str | None) -> str | None:
    '''이상도에 따른 기침 위험도 표시'''
    if stage2_level == "normal_range":
        return "GOOD"
    if stage2_level == "borderline":
        return "CAUTION"
    if stage2_level in ("warning"):
        return "WARNING"
    return None

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_uuid: str = Form(default="local"),
):
    ''' 파일 업로드 시작 '''
    file_id = str(uuid.uuid4()) # 고유 파일명(업로드한 음성파일) 생성
    record_uuid = str(uuid.uuid4())
    storage_result = None
    storage_error = None
    quality_fail_reason = None # 분석 실패 이유

    original_filename = file.filename   # 원래 파일 이름
    saved_path = UPLOAD_DIR / f"{file_id}_{original_filename}"       # 원본 음성 저장 경로
    wav_path = CONVERTED_DIR / f"{file_id}.wav"   # 전처리 음성 저장 경로

    try:
        # 원본 음성파일 저장 
        with open(saved_path, "wb") as f: # 'wb' = 바이너리 파일 쓰기 모드 (음성은 텍스트가 아니므로)
            content = await file.read() # 업로드 파일 내용 읽기
            f.write(content) # 서버 폴더에 저장

        # 업로드 파일 분석
        try:
            probe_info = probe_audio(str(saved_path)) # 업로드 파일 오디오 형식 검증
            # print("probe_info =", probe_info) # 분석 결과 확인용
        except Exception as e:
            print("probe_audio error =", repr(e))
            safe_unlink(saved_path) # safe_unlink 는 검증에 실패하면 파일을 삭제함
            # quality_fail_reason = "not_audio"
            raise HTTPException(status_code=400, detail="오디오가 아니거나 분석할 수 없음")
    
        # 업로드 음성 전처리(변환)
        try:
            convert_audio_to_wav(str(saved_path), str(wav_path)) # 업로드된 파일을 wav 로 변환
        except Exception:
            # 변환 실패 시 원본, 변환 음성 파일 삭제
            safe_unlink(saved_path)
            safe_unlink(wav_path)
            # quality_fail_reason = "to WAV convert Failed"
            raise HTTPException(status_code=400, detail=f"WAV 변환 실패")

        # 전처리(변환) 후 변환 결과 정상인지 확인
        try:
            wav_audio_info = probe_audio(str(wav_path)) # 변환된 파일 다시 검사
        except Exception:
            safe_unlink(saved_path)
            safe_unlink(wav_path)
            # quality_fail_reason = "WAV Analyse Failed"
            raise HTTPException(status_code=400, detail="변환된 WAV 분석 실패")

        # 변환된 wav 파일이 전처리에 적합한 형태인지 검사
        validation = validate_audio(wav_audio_info, min_duration=MIN_DURATION)

        # 변환된 파일 검증 실패 시
        if not validation["ok"]:
            safe_unlink(saved_path)
            safe_unlink(wav_path)
            # quality_fail_reason = validation['reason']
            raise HTTPException(status_code=400,
                                detail={"message":"변환된 wav 파일이 요구 조건 충족하지 않음",
                                        "reason":validation["reason"],
                                        },
                                )

        # quality score(녹음 품질 점수) 계산
        try:
            audio, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True) # 변환된 wav 파일 읽어오기

            quality_result = calculate_quality_score(
                audio=audio,
                sample_rate=sr,
                duration_sec=validation.get("duration"),
            )

            x = preprocess_for_model(audio=audio, sr=sr)

        except Exception as e:
            print("calculate_quality_score error =", repr(e))
            quality_result = {
                "quality_score": 0.0,
                "quality_flag": "LOW",
                "message": "녹음 품질을 평가하는 중 문제가 발생했습니다.",
                "details": {},
            }
            # quality_fail_reason = "quality_score_calculation_failed"

        quality_score = quality_result["quality_score"]
        quality_flag = quality_result["quality_flag"]

        prediction = predict(x) 

        # 최종 점수
        final_score = calculate_final_score(
            cough_confidence=prediction.get("prob"),
            quality_score=quality_score,
        )
        final_prediction = get_final_prediction(final_score)

        stage2 = prediction.get("stage2", {}) # 2단계 이상도 판정
        healthy_distance = None # 정상 기침 기준 데이터
        embedding_vector = None
        risk_level = None # 이상도 요약(사용자용)

        model_version = prediction.get("model") # 분석에 사용한 모델
        report_text = final_prediction["final_message"] # 보고서 요약 문장 (사용자용)

        if not stage2.get("skipped", True):
            distance = stage2.get("distance")
            if distance is not None:
                healthy_distance = float(distance)

            embedding = stage2.get("embedding")
            if embedding is not None:
                embedding_vector = embedding
                
            interpretation = stage2.get("interpretation", {})
            risk_level = map_risk_level(interpretation.get("level")) # 이상도 요약(사용자용)

        try:
            object_name = build_audio_object_name(
                user_uuid=user_uuid,
                record_uuid=record_uuid,
                suffix=".wav",
            )

            storage_result = upload_file_to_object_storage(
                file_path=wav_path,
                object_name=object_name,
                content_type="audio/wav",
            )

        except Exception as e:
            print("object storage upload error =", repr(e))
            storage_error = str(e)

        # DB 에 저장할 데이터
        mr_record = {
                "record_uuid": record_uuid,
                "cough_detected": prediction.get("is_cough"),
                "cough_confidence": prediction.get("prob"),
                "cough_threshold": COUGH_THRESHOLD,
                "quality_score": quality_score,
                "quality_flag": quality_flag,
                "quality_fail_reason": quality_fail_reason,
                "report_text": report_text,
                "healthy_distance": healthy_distance,
                "embedding_vector": embedding_vector,
                "model_version": model_version,
                "audio_duration_sec": validation.get("duration"),
                "reference_dataset": "COUGHVID_HEALTHY",
                "distance_metric": "MAHALANOBIS",
                "reference_version": "healthy_thresholds.json",
                "risk_level": risk_level,
                "final_score": final_score,
                "final_label": final_prediction["final_label"],
                "audio_storage_namespace": storage_result["namespace"] if storage_result else None,
                "audio_storage_bucket": storage_result["bucket"] if storage_result else None,
                "audio_storage_key": storage_result["object_name"] if storage_result else None,
                "audio_storage_uploaded_at": datetime.now().isoformat(timespec="seconds") if storage_result else None,
                "audio_storage_error": storage_error,
                "consent_to_store": 1,
            }
        
        saved_result = save_mr_record(
            mr_record=mr_record,
            user_uuid=None,
            # local_audio_path=str(saved_path), 
        )

        # 클라이언트에게 반환할 json
        return {
            "ok":True,
            "filename":original_filename,
            "validation":validation,
            "prediction":prediction,
            "quality": quality_result,
            "final_prediction": {
                "final_score": final_score,
                "final_detected": final_prediction["final_detected"],
                "final_label": final_prediction["final_label"],
                "final_message": final_prediction["final_message"],
            },
            "mr_record": mr_record,
            "db_saved": saved_result,
            "stored_audio": storage_result is not None,
            "storage": storage_result,
            "storage_error": storage_error,
        }
    
    finally:
        await file.close()
        safe_unlink(saved_path)
        safe_unlink(wav_path)