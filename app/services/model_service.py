import json
from functools import lru_cache

import numpy as np
import tensorflow  as tf

from app.core.config import (
    MODEL_PATH,
    HEALTHY_REF_JSON_PATH,
    HEALTHY_REF_NPZ_PATH
)

@lru_cache(maxsize=1)
def get_model():
    '''
    모델 로드 테스트용
    print(f"MODEL_PATH = {MODEL_PATH}")
    print(f"exists = {MODEL_PATH.exists()}")
    '''

    if not MODEL_PATH.exists():
        print("model load failed: path does not exist")
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
    
@lru_cache(maxsize=1)
def get_embedding_model():
    model = get_model()
    if model is None:
        return None

    print("=== 모델 레이어 목록 ===")
    for i, layer in enumerate(model.layers):
        try:
            print(f"[{i}] name={layer.name}, output_shape={layer.output.shape}")
        except Exception as e:
            print(f"[{i}] name={layer.name}, output_shape=확인불가 ({e})")
            
    candidate_names = ["embedding", "activation", "embed", "embedding_1"]
    
    for name in candidate_names:
        try :
            emb_layer = model.get_layer(name)
            emb_model = tf.keras.Model(inputs=model.input, outputs=emb_layer.output)
            print(f"embedding model 생성 성공 : layer={name}")
            return emb_model
        except Exception:
            pass
        
    print("embedding layer를 찾지 못했습니다. model.summary() / layer 이름 확인 필요")
    return None

@lru_cache(maxsize=1)
def get_healthy_reference():
    if not HEALTHY_REF_JSON_PATH.exists():
        print("healthy ref json not found")
        return None

    if not HEALTHY_REF_NPZ_PATH.exists():
        print("healthy ref npz not found")
        return None

    try:
        with open(HEALTHY_REF_JSON_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        npz = np.load(HEALTHY_REF_NPZ_PATH, allow_pickle=True)
        ref = {
            "mean": npz["mean"].astype(np.float32),
            "inv_cov": npz["inv_cov"].astype(np.float32),
            "cov_reg": npz["cov_reg"].astype(np.float32),
            "ref_distances": npz["ref_distances"].astype(np.float32),
            "meta": meta,
        }

        print("healthy reference 로드 성공")
        print("mean shape =", ref["mean"].shape)
        print("inv_cov shape =", ref["inv_cov"].shape)
        print("ref_distances shape =", ref["ref_distances"].shape)
        return ref

    except Exception as e:
        print(f"healthy reference 로드 실패: {type(e).__name__}: {e}")
        return None