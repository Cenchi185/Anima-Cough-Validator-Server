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
    ''' 모델 로드 '''
    # 모델 로드 테스트용
    # print(f"MODEL_PATH = {MODEL_PATH}")
    # print(f"exists = {MODEL_PATH.exists()}")

    if not MODEL_PATH.exists(): # 모델이 경로에 없음
        print("model load failed: path does not exist")
        return None
    
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
        # print("모델 로드 성공")
        # print("모델 입력형 =", model.input_shape)
        # print("모델 출력형 =", model.output_shape)
        return model
    except Exception as e:
        print(f"모델 로드 실패: {type(e).__name__}: {e}")
        return None
    
@lru_cache(maxsize=1)
def get_joint_model():
    model = get_model()
    if model is None:
        return None

    try:
        emb_layer = model.get_layer("embedding")
        joint_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.output, emb_layer.output]
        )
        print("joint model 생성 성공: output + embedding")
        return joint_model
    except Exception as e:
        print(f"joint model 생성 실패: {type(e).__name__}: {e}")
        return None

@lru_cache(maxsize=1)
def get_healthy_reference():
    '''정상 기침 기준 파일 불러오기'''
    if not HEALTHY_REF_NPZ_PATH.exists(): # 기준 파일이 없으면 에러
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

        #print("healthy reference 로드 성공")
        #print("mean shape =", ref["mean"].shape)
        #print("inv_cov shape =", ref["inv_cov"].shape)
        #print("ref_distances shape =", ref["ref_distances"].shape)
        return ref

    except Exception as e:
        print(f"healthy reference 로드 실패: {type(e).__name__}: {e}")
        return None