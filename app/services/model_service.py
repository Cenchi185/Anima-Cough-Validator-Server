from functools import lru_cache
from app.core.config import MODEL_PATH, USE_FAKE_MODEL

@lru_cache(maxsize=1)
def get_model():
    # fake 모드일 때
    if USE_FAKE_MODEL:
        return None

    # 모델 파일 없으면 에러
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델을 찾을수 없음: {MODEL_PATH}")

    # 👉 여기 나중에 실제 모델 로딩 코드 들어감
    # 예: TensorFlow
    # import tensorflow as tf
    # model = tf.keras.models.load_model(MODEL_PATH)
    # return model

    raise NotImplementedError("모델이 준비되지 않음")