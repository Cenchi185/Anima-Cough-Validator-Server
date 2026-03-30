from pathlib import Path

# 저장 파일 위치 기준점 (프로젝트 루트 폴더)
BASE_DIR = Path(__file__).resolve().parent.parent.parent 
DATA_DIR = BASE_DIR / "app" / "data"

UPLOAD_DIR = DATA_DIR / "uploads"  # 원본 파일 저장 위치
CONVERTED_DIR = DATA_DIR / "converted"  # 변환 파일 저장 위치

MODEL_DIR = BASE_DIR / "models"
# MODEL_PATH = MODEL_DIR / "savedmodel_best_1to1_embed64"
MODEL_PATH = MODEL_DIR / "best_1to1_embed64.keras"

TARGET_SR = 16000
TARGET_CHANNELS = 1
MIN_DURATION = 1.0

TARGET_SEC = 1.5
TARGET_HW = (90, 513)

N_FFT = 1024
HOP_LENGTH = 128
N_MELS = 128

COUGH_THRESHOLD = 0.6