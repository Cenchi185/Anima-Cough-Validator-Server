from pathlib import Path

# app 폴더 기준 경로
APP_DIR = Path(__file__).resolve().parent.parent

# 기존 main.py 에 있던 폴더들
MODEL_DIR = APP_DIR / "models"
UPLOAD_DIR = APP_DIR / "uploads"

# 업로드 폴더 없으면 생성
UPLOAD_DIR.mkdir(exist_ok=True)