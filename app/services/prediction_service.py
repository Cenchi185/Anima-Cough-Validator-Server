import numpy as np
from app.core.config import COUGH_THRESHOLD
from app.services.model_service import (
    get_model,
    get_embedding_model,
    get_healthy_reference,
)

# 서버 입출력 처리 테스트용 페이크 모델
def fake_model_predict(x: np.ndarray) -> dict:
    prob = 0.85
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
    diff = x - mean
    dist_sq = float(diff @ inv_cov @ diff.T)
    dist_sq = max(dist_sq, 0.0)
    return float(np.sqrt(dist_sq))

def interpret_distance(dist: float, q: dict) -> dict:
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
    model = get_model()

    if model is None:
        return fake_model_predict(x)

    return real_model_predict(model, x)

def real_model_predict(model, x: np.ndarray) -> dict:
    y = model.predict(x, verbose=0)
    prob = float(y[0][0]) if hasattr(y[0], "__len__") else float(y[0])

    # 기침 검증 결과
    result = {
        "is_cough": prob >= COUGH_THRESHOLD,
        "prob": prob,
        "input_shape": list(x.shape),
        "model": "best_1to1_embed64.keras"
    }
    
    # 기침이 아니면 2단계 분석 생략
    if prob < COUGH_THRESHOLD:
        result["stage2"] = {
            "skipped": True,
            "reason": f"prob < {COUGH_THRESHOLD}"
        }
        return result
    
    emb_model = get_embedding_model()
    ref = get_healthy_reference()
    
    print("=== healthy ref meta ===")
    print(ref["meta"])
    
    if emb_model is None:
        result["stage2"] = {
            "skipped": True,
            "reason": "embedding_model_not_available"
        }
        return result

    if ref is None:
        result["stage2"] = {
            "skipped": True,
            "reason": "healthy_reference_not_available"
        }
        return result

    # embedding 추출
    emb = emb_model.predict(x, verbose=0)
    emb = np.squeeze(emb).astype(np.float32)
    
    emb_norm = np.linalg.norm(emb)
    print("emb l2 norm =", emb_norm)

    emb_act = np.maximum(emb, 0)
    print("emb_act min =", np.min(emb_act))
    print("emb_act max =", np.max(emb_act))
    print("emb_act mean =", np.mean(emb_act))
    print("emb_act first5 =", emb_act[:5])
    print("emb_act l2 norm =", np.linalg.norm(emb_act))

    # shape 확인
    if emb.ndim != 1:
        result["stage2"] = {
            "skipped": True,
            "reason": f"unexpected_embedding_shape: {list(emb.shape)}"
        }
        return result

    print("ref mean shape =", ref["mean"].shape)
    print("ref mean min =", np.min(ref["mean"]))
    print("ref mean max =", np.max(ref["mean"]))
    print("ref mean first5 =", ref["mean"][:5])

    dist = mahalanobis_distance(
        x=emb,
        mean=ref["mean"],
        inv_cov=ref["inv_cov"],
    )

    print("distance =", dist)
    print("ref quantiles =", ref["meta"]["distance_quantiles"])

    quantiles = ref["meta"]["distance_quantiles"]
    interpretation = interpret_distance(dist, quantiles)

    result["stage2"] = {
        "skipped": False,
        "embedding_dim": int(ref["meta"]["embedding_dim"]),
        "distance": dist,
        "quantiles": quantiles,
        "interpretation": interpretation,
    }

    return result