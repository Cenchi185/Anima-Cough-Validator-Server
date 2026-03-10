import os, json
from typing import Optional

import numpy as np
import tensorflow as tf

from app.core.config import MODEL_DIR

# 전역변수 (서버 시작 시 1회 로드, 요청마다 로드하면 느려짐)
ANIMA1_LOADED = None
ANIMA1_INFER = None  # serving_default signature
ANIMA1_THRESHOLD: float = 0.6
ANIMA1_MODEL_VERSION: str = "unknown"
ANIMA1_SAVEDMODEL_DIR = os.path.join(MODEL_DIR, "savedmodel_best_1to1")
ANIMA1_SPEC_PATH = os.path.join(ANIMA1_SAVEDMODEL_DIR, "inference_spec.json")

ANIMA2_MODEL: Optional[tf.keras.Model] = None
ANIMA2_CENTROID: Optional[np.ndarray] = None # centroid (정상 기침 중심점)
ANIMA2_DISTANCE_METRIC = "cosine"  # 거리 계산 방식


def _load_anima1_inference_spec() -> None:
    global ANIMA1_THRESHOLD, ANIMA1_MODEL_VERSION

    if os.path.exists(ANIMA1_SPEC_PATH): # anima1 설정값 불러오기
        with open(ANIMA1_SPEC_PATH, "r", encoding="utf-8") as f:
            spec = json.load(f)
        ANIMA1_THRESHOLD = float(spec.get("threshold_default", 0.6))    # threshold_default 값 불러오기, 없으면 0.6
        ANIMA1_MODEL_VERSION = str(spec.get("model_version", "unknown"))# 모델 버전 정보
    else: # 설정 파일이 없을 경우
        ANIMA1_THRESHOLD = 0.6
        ANIMA1_MODEL_VERSION = "unknown"

# 분석에 필요한 변수 로드
def load_assets() -> None:
    global ANIMA1_LOADED, ANIMA1_INFER
    global ANIMA2_MODEL, ANIMA2_CENTROID

    # anima1 로드
    if not os.path.exists(ANIMA1_SAVEDMODEL_DIR):
        raise FileNotFoundError(f"SavedModel dir not found: {ANIMA1_SAVEDMODEL_DIR}")

    _load_anima1_inference_spec()

    try:
        ANIMA1_LOADED = tf.saved_model.load(ANIMA1_SAVEDMODEL_DIR)
        ANIMA1_INFER = ANIMA1_LOADED.signatures["serving_default"]
    except Exception as e:
        raise RuntimeError(f"Failed to load anima1 SavedModel: {e}")
    
    # 로드 검증용
    # print("ANIMA1 signature inputs:", ANIMA1_INFER.structured_input_signature)
    # print("ANIMA1 signature outputs:", ANIMA1_INFER.structured_outputs)

    # anima2 로드
    anima2_path = os.path.join(MODEL_DIR, "embedding_model.keras")
    if not os.path.exists(anima2_path):
        raise FileNotFoundError(f"embedding_model.keras not found at: {anima2_path}")
    ANIMA2_MODEL = tf.keras.models.load_model(anima2_path, compile=False)

    # centroid 로드 (models/centroid.npy)
    centroid_path = os.path.join(MODEL_DIR, "centroid.npy")
    if not os.path.exists(centroid_path):
        raise FileNotFoundError(f"centroid.npy not found at: {centroid_path}")
    ANIMA2_CENTROID = np.load(centroid_path).astype(np.float32)

    # (64,)인지 확인
    if ANIMA2_CENTROID.shape != (64,):
        raise ValueError(f"centroid shape must be (64,), got {ANIMA2_CENTROID.shape}")