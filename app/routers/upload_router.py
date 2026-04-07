from pathlib import Path
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

import uuid

router = APIRouter()

# 음성파일 저장 폴더 생성
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

# 파일 지우기
def safe_unlink(path:Path) -> None:
    if path.exists():
        path.unlink()

def build_report_text(prediction: dict) -> str:
    stage2 = prediction.get("stage2", {})

    # 2단계 결과가 있으면 그 해석 문구를 우선 사용
    if not stage2.get("skipped", True):
        interpretation = stage2.get("interpretation", {})
        message = interpretation.get("message")
        if message:
            return message

    # 2단계가 없으면 1단계 결과 기준 기본 문구
    if prediction.get("is_cough"):
        return "기침으로 판단되었습니다."
    return "기침이 아닌 것으로 판단되었습니다."

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
                                        "reason":validation["reason"],},
                                )
    
        x = preprocess_for_model(str(wav_path)) # 모델 입력 파라미터 전처리
        prediction = predict(x) 

        stage2 = prediction.get("stage2", {})
        change_score = None

        model_version = prediction.get("model")
        report_text = build_report_text(prediction)

        if not stage2.get("skipped", True):
            distance = stage2.get("distance")
            if distance is not None:
                change_score = float(distance)

        mr_record = {
                "cough_detected": prediction.get("is_cough"),
                "cough_confidence": prediction.get("prob"),
                "cough_threshold": COUGH_THRESHOLD,
                "quality_score": prediction.get("prob"),
                "quality_fail_reason": quality_fail_reason,
                "report_text": report_text,
                "change_score": change_score,
                "embedding_version": model_version,
                "audio_duration_sec": validation.get("duration"),
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
            "mr_record":mr_record,
            "db_saved": saved_result,
        }
    
    finally:
        await file.close()