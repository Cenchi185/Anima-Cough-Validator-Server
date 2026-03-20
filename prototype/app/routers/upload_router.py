'''
UploadFile, File = 파일 업로드
JSONResponse = 에러를 상태코드 포함한 JSON 으로 보여줌
subprocess = 터미널 명령어 실행
tempfile = 업로드 파일을 임시파일로 저장
os = 그 임시파일을 나중에 삭제
'''

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.services.anima_service import analyze_anima1, analyze_anima2
from app.services.audio_service import probe_wav, convert_to_wav, SR, WIN_SEC, HOP_SEC
from app.services import model_loader
from app.core.config import UPLOAD_DIR

import subprocess
import tempfile
import os
import uuid

from typing import Any, Dict, Optional

router = APIRouter()

@router.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model_loader.ANIMA1_INFER is not None,
        "model_version": model_loader.ANIMA1_MODEL_VERSION,
        "threshold_default": float(model_loader.ANIMA1_THRESHOLD),
        "sr": SR,
        "win_sec": WIN_SEC,
        "hop_sec": HOP_SEC,
    }

# /upload 로 post 요청 수신 (나중에 웹(?)에서 수정하기)
@router.post("/upload",
        summary="Upload cough audio",
        description="1.5~6초 길이의 음성 파일을 업로드하면 기침 분석 결과를 반환"
        )
#async 는 비동기 함수 (이 함수를 중간에 멈추고 다른 작업을 할 수도 있음)
async def upload_audio(
    file: UploadFile = File(...),
    mr_audio_path: str = Form(""),  # 앱이 S3 업로드 후 경로를 같이 보내는 걸 권장 :contentReference[oaicite:16]{index=16}
    cough_count: Optional[int] = Form(None),
    pain_score: Optional[int] = Form(None)
):
    mr_audio_path = (mr_audio_path or "").strip()
    if not mr_audio_path:
        mr_audio_path = f"local://uploads/{uuid.uuid4().hex}_{file.filename}"

    # with 은 현재 코드 블록이 끝나는 순간 .close() 를 바로 호출하는 파이썬 문법
    # 업로드한 파일을 임시 파일에 붙여넣기
    # 임시 파일이지만, 검증이 끝난뒤 처리 전까지는 지워지면 안되므로 delete=False 조건을 넣음
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    wav_path = tmp_path + ".wav"

    try:
        # wav(16k mono)로 변환
        convert_to_wav(tmp_path, wav_path)

        meta = probe_wav(wav_path)
        
        # ffprobe 결과에 streams 가 없거나 빈 리스트인지 검증
        if "streams" not in meta or not meta["streams"]:
            return JSONResponse(status_code=400, content={"ok": False, "reason": "오디오가 비었음"})

        # streams[0] 은 오디오 스트림 정보를 담고 있음
        stream = meta["streams"][0]
        duration = float(meta["format"]["duration"]) # 전체 길이

        if stream["sample_rate"] != "16000": # 레이트가 16kHz 인지 검증
            return JSONResponse(status_code=500, content={"ok": False, "reason": "samplerate 가 16kHz 가 아님"})
        if stream["channels"] != 1: # 채널이 1채널인지 검증
            return JSONResponse(status_code=500, content={"ok": False, "reason": "채널이 mono 가 아님"})
        if not (1.5 <= duration <= 6.0): # 파일 길이 제한
            return JSONResponse(status_code=500, content={"ok": False, "reason": "파일 길이가 너무 길거나 짧음", "duration": duration})
        
        a1 = analyze_anima1(wav_path)
        if not a1.get("ok", False):
            return JSONResponse(status_code=500, content={"ok": False, "reason": "anima1_failed", "detail": a1})

        # anima1이 기침이라고 판단할 때만 anima2 실행
        a2 = None
        if a1["is_cough"]:
            a2 = analyze_anima2(wav_path, include_raw=False)
            if not a2.get("ok", False):
                # anima2 실패해도 a1 결과는 유효하므로 null로 둠(원하면 여기서 500 처리로 바꿔도 됨)
                a2 = None

         # DB(MRRecord)로 바로 매핑 가능한 통합 구조 :contentReference[oaicite:17]{index=17}
        mr_record = {
            "is_cough": 1 if a1["is_cough"] else 0,                    # :contentReference[oaicite:18]{index=18}
            "cough_probability": float(a1["cough_probability"]),       # :contentReference[oaicite:19]{index=19}
            "pd_threshold": float(a1["pd_threshold"]),                 # :contentReference[oaicite:20]{index=20}
            "cough_count": cough_count,
            "pain_score": pain_score,
            "quality_score": None,                                     # :contentReference[oaicite:21]{index=21} (추후 계산 로직 추가 권장)
            "similarity": (float(a2["similarity"]) if a2 else 0.0),
            "change_score": None,                                      # :contentReference[oaicite:22]{index=22}
            "raw_embeddings": (a2["embedding_mean"] if a2 else None),  # :contentReference[oaicite:23]{index=23}
            "mr_duration": float(duration),                            # :contentReference[oaicite:24]{index=24}
            "mr_audio_path": mr_audio_path,                            # :contentReference[oaicite:25]{index=25}
            "model_version": str(a1["model_version"]),                 # :contentReference[oaicite:26]{index=26}
        }

        return {
            "ok": True,
            "mr_record": mr_record,
            "debug": {  # 앱 저장엔 안 써도 되는 참고값
                "duration": duration,
                "anima1": a1,
                "anima2": a2,
            }
        }

    # .wav 변환 에러 발생시
    except subprocess.CalledProcessError as e:
        return JSONResponse(status_code=400, content={
        "ok": False,
        "reason": "ffmpeg_convert_failed",
        "detail": (e.stderr.decode() if isinstance(e.stderr, bytes) else str(e))
    })

    except Exception as e:
        # 예상치 못한 예외도 JSON으로 반환(디버깅 도움)
        return JSONResponse(status_code=500, content={"ok": False, "reason": "서버 에러", "detail": str(e)})
    
    # 만들어진 임시파일은 무조건 삭제되도록
    finally:
        for p in [tmp_path, wav_path]:
            if p and os.path.exists(p):
                os.remove(p)

# 음성파일 업로드 테스트용
@router.post("/upload_test")
async def upload_test(file: UploadFile = File(...)):
    # 1) 파일명 충돌 방지
    ext = os.path.splitext(file.filename or "")[1]
    save_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)

    # 2) 저장
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    return JSONResponse({
        "ok": True,
        "original_filename": file.filename,
        "content_type": file.content_type,
        "bytes": len(content),
        "saved_path": save_path,
    })