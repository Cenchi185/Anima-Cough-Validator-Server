from pathlib import Path
import uuid
import librosa

from app.core.config import UPLOAD_DIR, CONVERTED_DIR, MIN_DURATION, COUGH_THRESHOLD
from fastapi import APIRouter, UploadFile, File, HTTPException
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

router = APIRouter()

# 음성파일 저장 폴더 생성
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

# 파일 지우기
def safe_unlink(path:Path) -> None:
    if path.exists():
        path.unlink()

def map_risk_level(stage2_level: str | None) -> str | None:
    if stage2_level == "normal_range":
        return "GOOD"
    if stage2_level == "borderline":
        return "CAUTION"
    if stage2_level in ("abnormal_high", "abnormal_very_high"):
        return "RISK"
    return None

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4()) # 고유 파일명(업로드한 음성파일) 생성
    quality_fail_reason = None # 분석 실패 이유

    original_filename = file.filename   # 원래 파일 이름
    saved_path = UPLOAD_DIR / f"{file_id}_{original_filename}"       # 원본 파일 저장 경로
    wav_path = CONVERTED_DIR / f"{file_id}.wav"   # 변환 파일 저장 경로

    try:
        # 원본 음성파일 저장
        with open(saved_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 업로드 파일 오디오 검증
        try:
            probe_info = probe_audio(str(saved_path))
            print("probe_info =", probe_info)
        except Exception as e:
            print("probe_audio error =", repr(e))
            safe_unlink(saved_path) # safe_unlink 는 검증에 실패하면 파일을 삭제함
            # quality_fail_reason = "not_audio"
            raise HTTPException(status_code=400, detail="오디오가 아니거나 분석할 수 없음")
    
        # 변환 검증
        try:
            convert_audio_to_wav(str(saved_path), str(wav_path))
        except Exception:
            safe_unlink(saved_path)
            safe_unlink(wav_path)
            # quality_fail_reason = "to WAV convert Failed"
            raise HTTPException(status_code=400, detail=f"WAV 변환 실패")

        # 변환 후 파일 검증
        try:
            wav_audio_info = probe_audio(str(wav_path))
        except Exception:
            safe_unlink(saved_path)
            safe_unlink(wav_path)
            # quality_fail_reason = "WAV Analyse Failed"
            raise HTTPException(status_code=400, detail="변환된 WAV 분석 실패")

        # 변환된 wav 파일로 기침 검증
        validation = validate_audio(wav_audio_info, min_duration=MIN_DURATION)

        if not validation["ok"]:
            safe_unlink(saved_path)
            safe_unlink(wav_path)
            # quality_fail_reason = validation['reason']
            raise HTTPException(status_code=400,
                                detail={"message":"변환된 wav 파일이 요구 조건 충족하지 않음",
                                        "reason":validation["reason"],
                                        },
                                )

        # quality score 계산
        try:
            audio, sr = librosa.load(str(wav_path), sr=None, mono=True)

            quality_result = calculate_quality_score(
                audio=audio,
                sample_rate=sr,
                duration_sec=validation.get("duration"),
            )
        except Exception as e:
            print("calculate_quality_score error =", repr(e))
            quality_result = {
                "quality_score": 0.0,
                "quality_flag": "LOW",
                "message": "녹음 품질을 평가하는 중 문제가 발생했습니다.",
                "details": {},
            }
            quality_fail_reason = "quality_score_calculation_failed"

        quality_score = quality_result["quality_score"]
        quality_flag = quality_result["quality_flag"]

        x = preprocess_for_model(str(wav_path)) # 모델 입력 파라미터 전처리
        prediction = predict(x) 

        # 최종 점수
        final_score = calculate_final_score(
            cough_confidence=prediction.get("prob"),
            quality_score=quality_score,
        )
        final_prediction = get_final_prediction(final_score)

        stage2 = prediction.get("stage2", {})
        healthy_distance = None
        risk_level = None

        model_version = prediction.get("model")
        report_text = final_prediction["final_message"]

        if not stage2.get("skipped", True):
            distance = stage2.get("distance")
            if distance is not None:
                healthy_distance = float(distance)
                
            interpretation = stage2.get("interpretation", {})
            risk_level = map_risk_level(interpretation.get("level"))

        mr_record = {
                "cough_detected": prediction.get("is_cough"),
                "cough_confidence": prediction.get("prob"),
                "cough_threshold": COUGH_THRESHOLD,
                "quality_score": quality_score,
                "quality_flag": quality_flag,
                "quality_fail_reason": quality_fail_reason,
                "report_text": report_text,
                "healthy_distance": healthy_distance,
                "model_version": model_version,
                "audio_duration_sec": validation.get("duration"),
                "reference_dataset": "COUGHVID_HEALTHY",
                "distance_metric": "MAHALANOBIS",
                "reference_version": "healthy_ref_final.json",
                "risk_level": risk_level,
                "final_score": final_score,
                "final_label": final_prediction["final_label"],
            }
        
        saved_result = save_mr_record(
            mr_record=mr_record,
            user_uuid=None,
            local_audio_path=str(saved_path),
        )

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
        }
    
    finally:
        await file.close()