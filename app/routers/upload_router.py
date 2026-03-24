from pathlib import Path
from app.core.config import UPLOAD_DIR, CONVERTED_DIR, MIN_DURATION
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.audio_service import (
    probe_audio,
    convert_audio_to_wav,
    validate_audio,
    preprocess_for_model,
)
from app.services.prediction_service import predict
import uuid

router = APIRouter()

# 음성파일 저장 폴더 생성
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

# 파일 지우기
def safe_unlink(path:Path) -> None:
    if path.exists():
        path.unlink()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4()) # 고유 파일명(업로드한 음성파일) 생성

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
            probe_audio(str(saved_path))
        except Exception:
            safe_unlink(saved_path) # safe_unlink 는 검증에 실패하면 파일을 삭제함
            raise HTTPException(status_code=400, detail=f"오디오가 아니거나 분석할 수 없음")

        # 변환 검증
        try:
            convert_audio_to_wav(str(saved_path), str(wav_path))
        except Exception:
            safe_unlink(saved_path)
            safe_unlink(wav_path)
            raise HTTPException(status_code=400, detail=f"WAV 변환 실패")

        # 변환 후 파일 검증
        try:
            wav_audio_info = probe_audio(str(wav_path))
        except Exception:
            safe_unlink(saved_path)
            safe_unlink(wav_path)
            raise HTTPException(status_code=400, detail="변환된 WAV 분석 실패")

        # 변환된 wav 파일로 기침 검증
        validation = validate_audio(wav_audio_info, min_duration=MIN_DURATION)

        if not validation["ok"]:
            safe_unlink(saved_path)
            safe_unlink(wav_path)
            raise HTTPException(status_code=400,
                                detail={"message":"변환된 wav 파일이 요구 조건 충족하지 않음",
                                        "reason":validation["reason"],},
                                )
    
        x = preprocess_for_model(str(wav_path)) # 모델 입력 파라미터 전처리
        prediction = predict(x)  # 가짜 모델

        return {
            "ok":True,
            "filename":original_filename,
            "validation":validation,
            "prediction":prediction,
        }
    
    finally:
        await file.close()