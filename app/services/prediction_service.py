import numpy as np
from app.core.config import COUGH_THRESHOLD, DEBUG
from app.services.model_service import (
    get_model,
    get_joint_model,
    get_healthy_reference,
)

def debug_print(*args):
    if DEBUG:
        print(*args)

def fake_model_predict(x: np.ndarray) -> dict:
    ''' 모델이 없을때 서버가 죽지 않도록 가짜 모델 사용 '''
    prob = 0.85
    debug_print("[PREDICT] fallback to fake model")
    
    return {
        "is_cough": prob >= COUGH_THRESHOLD,
        "prob": prob,
        "input_shape": list(x.shape),
        "model": "fake_model",
        "stage2": {
            "skipped": True,
            "reason": "fake_model"
        }
    }

def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> float:
    '''
    마할라노비스 거리 계산
    정상 기침 데이터 분포에서 얼마나 벗어났는지 확인하는 거리
    정상 데이터가 퍼진 방향과 분산까지 고려함
    '''
    diff = x - mean
    dist_sq = float(diff @ inv_cov @ diff.T)
    dist_sq = max(dist_sq, 0.0)
    return float(np.sqrt(dist_sq))

def interpret_distance(dist: float, q: dict) -> dict:
    '''
    p80 이하 → 정상 기침 기준에 가까움
    p95 이하 → 약간 벗어남
    p99 이하 → 이상 가능성 높음
    p99 초과 → 많이 벗어남
    '''
    if dist <= q["p80"]:
        return {
            "level": "normal_range",
            "message": "정상 기침 기준에 비교적 가깝습니다."
        }
    elif dist <= q["p95"]:
        return {
            "level": "borderline",
            "message": "정상 범위에서 다소 벗어날 수 있습니다."
        }
    elif dist <= q["p99"]:
        return {
            "level": "abnormal_high",
            "message": "이상 가능성이 높습니다."
        }
    else:
        return {
            "level": "abnormal_very_high",
            "message": "정상 기침 기준에서 많이 벗어났습니다."
        }

def predict(x: np.ndarray) -> dict:
    '''전처리된 데이터(x)로 모델 예측 수행'''
    model = get_model()

    if model is None: # 모델이 없을땐 페이크 모델로 대체 (실제 분석 X)
        return fake_model_predict(x)

    return real_model_predict(model, x) # 진짜 모델 예측 결과 반환

def real_model_predict(model, x: np.ndarray) -> dict:
    '''
    진짜 모델로 예측 수행
    1단계 - 음성이 기침이 맞는지 검증
    2단계 - 기침의 이상도 측정
    '''
    joint_model = get_joint_model()

    if joint_model is None:
        y = model.predict(x, verbose=0)
        emb = None
    else :
        y, emb = joint_model.predict(x, verbose=0)

    y = np.asarray(y)
    prob = float(np.squeeze(y))

    # 기침 검증 결과, COUFH_THRESHOLD(0.6) 기준으로 기침인지 검증
    result = {
        "is_cough": prob >= COUGH_THRESHOLD,
        "prob": prob,
        "input_shape": list(x.shape),
        "model": "best_1to1_embed64.keras"
    }
    debug_print(f"[STAGE1] prob={prob:.6f}, threshold={COUGH_THRESHOLD}")
    
    # 기침이 아니면 2단계 분석 생략
    if prob < COUGH_THRESHOLD:
        result["stage2"] = {
            "skipped": True,
            "reason": f"prob < {COUGH_THRESHOLD}"
        }
        return result
    
    ref = get_healthy_reference()   # 정상 기침 기준 데이터 
    
    if emb is None:
        debug_print("[STAGE2] skipped: embedding not available")
        result["stage2"] = {
            "skipped": True,
            "reason": "embedding_not_available"
        }
        return result
    
    if ref is None: # 정상 기침 기준 데이터가 없음
        debug_print("[STAGE2] skipped: healthy reference not available")
        result["stage2"] = {
            "skipped": True,
            "reason": "healthy_reference_not_available"
        }
        return result
    
    # embedding 추출
    emb = np.squeeze(emb).astype(np.float32)
    
    # shape 확인
    if emb.ndim != 1:
        debug_print(f"[STAGE2] skipped: unexpected embedding shape={list(emb.shape)}")
        result["stage2"] = {
            "skipped": True,
            "reason": f"unexpected_embedding_shape: {list(emb.shape)}"
        }
        return result

    # 마할라노비스 거리 계산
    dist = mahalanobis_distance(
        x=emb,
        mean=ref["mean"],
        inv_cov=ref["inv_cov"],
    )

    quantiles = ref["meta"]["distance_quantiles"]
    interpretation = interpret_distance(dist, quantiles) # 마할라노비스 거리 해석

    debug_print(
        f"[STAGE2] distance={dist:.6f}, "
        f"level={interpretation['level']}"
    )

    result["stage2"] = {
        "skipped": False,
        "embedding_dim": int(ref["meta"]["embedding_dim"]),
        "distance": dist,
        "quantiles": quantiles,
        "interpretation": interpretation,
    }

    return result