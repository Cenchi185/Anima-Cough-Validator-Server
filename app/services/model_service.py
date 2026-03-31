import tensorflow  as tf
from functools import lru_cache
from app.core.config import MODEL_PATH

@lru_cache(maxsize=1)
def get_model():
    print(f"MODEL_PATH = {MODEL_PATH}")
    print(f"exists = {MODEL_PATH.exists()}")
    print(f"is_file = {MODEL_PATH.is_file()}")

    if not MODEL_PATH.exists():
        print("model load failed: path does not exist")
        return None

    if not MODEL_PATH.is_file():
        print("model load failed: path is not a file")
        return None

    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
        print("모델 로드 성공")
        print("모델 입력형 =", model.input_shape)
        print("모델 출력형 =", model.output_shape)
        return model
    except Exception as e:
        print(f"모델 로드 실패: {type(e).__name__}: {e}")
        return None