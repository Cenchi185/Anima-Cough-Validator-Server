import numpy as np
from app.core.config import COUGH_THRESHOLD
from app.services.model_service import get_model

# 서버 입출력 처리 테스트용 페이크 모델
def fake_model_predict(x:np.ndarray) -> dict:
    prob = 0.85
    return {
        "is_cough":prob >= 0.7,
        "prob":prob,
        "input_shape":list(x.shape),
        "model":"fake_model"
    }

def predict(x: np.ndarray) -> dict:
    model = get_model()

    if model is None:
        return fake_model_predict(x)

    return real_model_predict(model, x)

def real_model_predict(model, x: np.ndarray) -> dict:
    y = model.predict(x)
    prob = float(y[0][0]) if hasattr(y[0], "__len__") else float(y[0])

    return {
        "is_cough": prob >= 0.7,
        "prob": prob,
        "input_shape": list(x.shape),
        "model": "best_1to1_embed64.keras"
    }