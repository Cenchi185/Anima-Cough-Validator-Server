from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent # 저장 파일 위치 기준점
DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR = DATA_DIR / "uploads"  # 원본 파일 저장 위치
CONVERTED_DIR = DATA_DIR / "converted"  # 변환 파일 저장 위치

USE_FAKE_MODEL = True
MODEL_DIR = DATA_DIR / "models"
MODEL_PATH = MODEL_DIR / "model" # 실제 모델명 나중에 집어넣기

TARGET_SR = 16000
TARGET_CHANNELS = 1
MIN_DURATION = 1.0

TARGET_SEC = 1.5
TARGET_HW = (128, 188)

N_FFT = 1024
HOP_LENGTH = 128
N_MELS = 128