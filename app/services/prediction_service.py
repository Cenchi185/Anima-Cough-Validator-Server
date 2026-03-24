import numpy as np
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

def predict(x):
    model = get_model()

    if model is None:   # 모델이 없을때는 페이크 모델로 연결
        return fake_model_predict(x)
    
    return real_model_predict(model, x)

def real_model_predict(model, x):
    return NotImplementedError("실제 예측 모델이 아직 준비되지 않음")