from fastapi import APIRouter, UploadFile, File
from app.services.audio_service import probe_audio
import os, uuid

router = APIRouter()

UPLOAD_DIR = "uploads"

# 업로드 폴더 생성
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/test")
def test_router():
    return {"message":"업로드 라우터 연결됨"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4()) # 고유 파일명 생성
    file_path = os.path.join(UPLOAD_DIR, file_id + "_" + file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    audio_info = probe_audio(file_path)

    return {
        "ok":True,
        "filename":file.filename,
        "saved_path":file_path,
        "audio_info":audio_info
    }