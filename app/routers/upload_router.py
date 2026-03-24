from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.audio_service import (
    probe_audio,
    convert_audio_to_wav,
    validate_audio
)
import os, uuid

router = APIRouter()

UPLOAD_DIR = "uploads"      # 업로드 음성 파일 경로
CONVERTED_DIR = "converted" # 변환된 음성 파일 경로

# 업로드 폴더 생성
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CONVERTED_DIR,exist_ok=True)

@router.get("/test")
def test_router():
    return {"message":"업로드 라우터 연결됨"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4()) # 고유 파일명(업로드한 음성파일) 생성

    original_filename = file.filename   # 원래 파일 이름
    saved_name = f"{file_id}_{original_filename}"       # 저장될 음성파일 명
    saved_path = os.path.join(UPLOAD_DIR, saved_name)   # 원본 음성파일 저장 경로

    # 원본 음성파일 저장
    with open(saved_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 음성파일 분석중 예외처리
    try:
        original_audio_info = probe_audio(saved_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"오디오 파일 분석 실패: {str(e)}")

    # 변환된 파일 저장 경로
    wav_filename = f"{file_id}.wav"
    wav_path = os.path.join(CONVERTED_DIR, wav_filename)

    # 변환이 제대로 되었는지 검사
    try:
        convert_audio_to_wav(saved_path, wav_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"WAV 변환 실패: {str(e)}")
    try:
        wav_audio_info = probe_audio(wav_path)
    except Exception as e :
        raise HTTPException(status_code=400, detail=f"변환된 WAV 분석 실패:{str(e)}")

    # 변환된 wav 파일로 검증
    validation = validate_audio(wav_audio_info, min_duration=1.0)

    # 디버깅용
    return {
        "ok":True,
        "filename":original_filename,
        "saved_path":saved_path,
        "wav_path":wav_path,
        "origianl_audio_info":original_audio_info,
        "wav_audio_info":wav_audio_info,
        "validation":validation,
    }