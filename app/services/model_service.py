import tensorflow  as tf
from functools import lru_cache
from app.core.config import MODEL_PATH

_model = None

@lru_cache(maxsize=1)
def get_model():
    global _model

    if _model is not None:
        return _model

    print(f"MODEL_PATH = {MODEL_PATH}")
    print(f"exists = {MODEL_PATH.exists()}")
    print(f"is_file = {MODEL_PATH.is_file()}")

    try:
        _model = tf.keras.models.load_model(str(MODEL_PATH))
        print("model load success")
        return _model
    except Exception as e:
        print(f"model load failed: {e}")
        return None