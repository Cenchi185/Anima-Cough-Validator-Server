from typing import Any, Dict, Callable

import numpy as np
import tensorflow as tf

from app.services.audio_service import (
    SR,
    HOP_SEC,
    _read_wav_mono_16k,
    _frame_audio,
    _prepare_input_for_model,
)

from app.services import model_loader

# ANIMA1: 피치(고음 에너지) peak 기반 6윈도우 추론 설정
ANIMA1_USE_PITCH_PEAKS = True     # 피치 기반 추론 ON/OFF
ANIMA1_NUM_PEAKS = 2              # peak 2개
ANIMA1_WINDOWS_PER_PEAK = 3       # pre/center/post
ANIMA1_TOTAL_WINDOWS = ANIMA1_NUM_PEAKS * ANIMA1_WINDOWS_PER_PEAK  # 6

ANIMA1_PEAK_MIN_SEP_SEC = 0.8     # peak끼리 최소 간격(초)
ANIMA1_PEAK_MIN_RMS = 0.01        # 너무 조용하면 peak탐지 의미없음 -> fallback
ANIMA1_FALLBACK_TOPK = 6          # fallback 시 A 방식으로 뽑더라도 top 6개만 사용
ANIMA1_AGGREGATION = "mean"       # 6개 평균

# 1차 기침 검증 모델 전용 입력 규격 상수
ANIMA1_N_SAMPLES = 24000  # 1.5s * 16k
ANIMA1_N_FFT = 1024
ANIMA1_HOP = 256
ANIMA1_FRAMES = 90
ANIMA1_BINS = 513
ANIMA1_CH = 1

# 1차 기침 검증 모델 전용 전처리 함수
def _wav_to_anima1_spec_batch(audio_batch: np.ndarray) -> np.ndarray:
    """
    audio_batch: (B, 24000)
    return: (B, 90, 513, 1) float32
    """
    x = tf.convert_to_tensor(audio_batch, dtype=tf.float32)  # (B, N)

    stft = tf.signal.stft(
        x,
        frame_length=ANIMA1_N_FFT,          # 1024
        frame_step=ANIMA1_HOP,              # 256
        fft_length=ANIMA1_N_FFT,            # 1024
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )  # (B, 90, 513)

    feat = tf.math.log1p(tf.square(tf.abs(stft)))  # power + log1p
    feat = feat[..., tf.newaxis]                   # (B, 90, 513, 1)

    return feat.numpy().astype(np.float32)

# 숫자를 0~1 사이의 값으로 바꾸는 sigmoid 함수를 적용
def _sigmoid_safe(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))

# 모델의 예측값을 돌림
def _predict_prob(predict_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> Dict[str, Any]:
    """
    predict_fn(x) -> raw ndarray (B,1) or (B,)
    여러 윈도우면 max prob를 최종 prob로 사용.
    """
    raw = np.asarray(predict_fn(x))

    # shape 정리: (B,1) 또는 (B,)
    if raw.ndim == 2 and raw.shape[1] == 1:
        raw1 = raw[:, 0]
    else:
        raw1 = raw.reshape(-1)

    # 이미 prob(0~1)이면 그대로
    if np.all((raw1 >= 0.0) & (raw1 <= 1.0)):
        probs = raw1
        mode = "prob"
    else:
        probs = _sigmoid_safe(raw1)
        mode = "logit->sigmoid"

    prob_max = float(np.max(probs))

    return {
        "mode": mode,
        "window_probs": probs.tolist(),
        "prob": prob_max,
        "num_windows": int(len(probs)),
        "raw_shape": list(raw.shape),
        "input_shape_used": list(x.shape),
    }

# 기침 측정을 위한 거리계산 용 함수 1
def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = v.astype(np.float32)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n

# 기침 측정을 위한 거리계산 용 함수 2
def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2_normalize(a)
    b = _l2_normalize(b)
    sim = float(np.dot(a, b))        
    return 1.0 - sim

# 기침 측정을 위한 거리계산 용 함수 3
def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

# 구간을 3구역으로 나누는 함수
def _slice_window_1p5s(audio: np.ndarray, center_sec: float, sr: int = SR) -> np.ndarray:
    """
    center_sec를 중심으로 [-0.75s, +0.75s] 구간(총 1.5s)을 잘라 반환.
    범위를 벗어나면 0으로 패딩.
    return: (24000,) float32
    """
    win = int(1.5 * sr)             # 잘라낼 크기
    half = win // 2  # 12000        # 앞 뒤 0.75초씩 잘라내야 하므로
    center = int(center_sec * sr)

    # 잘라낼 구간
    start = center - half
    end = start + win

    # 결과 배열
    out = np.zeros((win,), dtype=np.float32)

    # 겹치는 구간 계산
    src_s = max(0, start)           # 오디오가 너무 앞에서 시작할 경우 부족한 부분을 빈 패딩(0)을 만들어 채움
    src_e = min(audio.shape[0], end)# 오디오가 너무 뒤에서 시작할 경우 부족한 부분을 빈 패딩(0)을 만들어 채움
    dst_s = src_s - start           # 복사해서 채워넣을 배열 위치계산(앞쪽)
    dst_e = dst_s + (src_e - src_s) # 복사해서 채워넣을 배열 위치계산(뒤쪽)

    if src_e > src_s:
        out[dst_s:dst_e] = audio[src_s:src_e].astype(np.float32)

    return out

# 피치가 튀는 구간을 찾는 함수
def _highfreq_energy_peaks(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    '피치가 높게 튀는 구간'을 직접 pitch 추정 없이 근사:
    STFT에서 고주파 대역 에너지(예: 2kHz 이상)의 시간별 합으로 score를 만들고 peak를 찾는다.

    return: peak_times_sec (shape (K,))  (K는 상황에 따라 0~ANIMA1_NUM_PEAKS)
    """
    # 1) 너무 조용한 경우 조기 종료
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32))) + 1e-12)) # 전체 음성의 크기
    if rms < ANIMA1_PEAK_MIN_RMS: # ANIMA1_PEAK_MIN_RMS 이거 보다 작으면 계산 안함
        return np.array([], dtype=np.float32)

    x = tf.convert_to_tensor(audio[None, :], dtype=tf.float32)  # (1, N)
    stft = tf.signal.stft(
        x,
        frame_length=ANIMA1_N_FFT,     # 1024
        frame_step=ANIMA1_HOP,         # 256
        fft_length=ANIMA1_N_FFT,       # 1024
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )  # (1, T(시간 프레임 갯수?, 513) stft 계산

    mag = tf.abs(stft)[0].numpy().astype(np.float32)  # (T, 513)

    # 시간 * 주파수 배열의 표의 에너지를 계산해서 2000Hz 이상의 주파수만 집어냄
    freqs = np.arange(mag.shape[1], dtype=np.float32) * (sr / float(ANIMA1_N_FFT))
    hi_mask = freqs >= 2000.0
    if not np.any(hi_mask):
        return np.array([], dtype=np.float32)

    # 고주파 에너지 합산(?)
    # 한 프레임 내에 있는 주파수의 에너지를 합해서 점수로 나타냄
    score = mag[:, hi_mask].sum(axis=1)  

    # 노이즈 때문에 갑자기 크게 튀는 값의 편차를 조금 줄여줌
    '''
    그냥 peak를 찾으면:
    아주 짧은 노이즈도 peak로 잡힘
    기침 하나에서 여러 개 peak가 생김

    갑자기 크게 튀는 값을 줄여서
    일정 프레임동안 높은 값이 나타나는 구간을 peak 로 잡기 위함
    보통 기침이 아주 짧게 노이즈처럼 한프레임만 높은 값이 나오지는 않기 때문
    '''
    if score.shape[0] >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        score = np.convolve(score, kernel, mode="same")

    # 4) peak 후보를 큰 값 순으로 고르고, 최소 간격 조건으로 2개 선택
    T = score.shape[0]
    if T == 0:
        return np.array([], dtype=np.float32)

    # 시간 변환: stft frame i의 대략적인 시간(초)
    # frame i 시작 샘플 = i * hop
    times = (np.arange(T, dtype=np.float32) * ANIMA1_HOP) / float(sr)

    # 큰 값 인덱스 순으로 정렬
    idx_sorted = np.argsort(-score)  # 내림차순

    picked = []
    min_sep = ANIMA1_PEAK_MIN_SEP_SEC

    for idx in idx_sorted:
        t = float(times[idx])
        # 파일 길이 밖이나 너무 앞/뒤 극단은 허용하되 패딩으로 처리 가능
        ok = True
        for pt in picked:
            if abs(t - pt) < min_sep:
                ok = False
                break
        if ok:
            picked.append(t)
            if len(picked) >= ANIMA1_NUM_PEAKS:
                break

    return np.array(picked, dtype=np.float32)

# 기침 확률 분석 모델(anima1) 로드
def analyze_anima1(wav_path: str) -> Dict[str, Any]:
    if model_loader.ANIMA1_INFER is None:
        return {"ok": False, "reason": "anima1 SavedModel not loaded"}

    audio = _read_wav_mono_16k(wav_path)

    # =========================
    # 팀장 방식: peak 2개 × (pre/center/post) 3개 = 총 6개 윈도우만 추론
    # =========================
    used_mode = "pitch_peaks_6w"
    frames_6 = None  # (6, 24000)

    if ANIMA1_USE_PITCH_PEAKS:
        peak_times = _highfreq_energy_peaks(audio, sr=SR)  # (0~2,)
        windows = []

        # peak마다 3구간:
        # center: t
        # pre:    t - 1.5
        # post:   t + 1.5
        # (주의) "앞0.75/뒤0.75 중심 1.5초"는 center window가 이미 만족.
        #        "앞 0.75 이전의 1.5초" == center보다 1.5초 앞
        #        "뒤 0.75 이후의 1.5초" == center보다 1.5초 뒤
        for t in peak_times.tolist():
            windows.append(_slice_window_1p5s(audio, center_sec=t, sr=SR))         # center
            windows.append(_slice_window_1p5s(audio, center_sec=t - 1.5, sr=SR))   # pre
            windows.append(_slice_window_1p5s(audio, center_sec=t + 1.5, sr=SR))   # post

        # peak이 2개 미만이면 부족분을 fallback로 채우거나 전체 fallback로 전환
        if len(windows) >= ANIMA1_TOTAL_WINDOWS:
            frames_6 = np.stack(windows[:ANIMA1_TOTAL_WINDOWS], axis=0).astype(np.float32)
        else:
            frames_6 = None

    # =========================
    # fallback: 축소 A 방식 (전체를 슬라이딩하되 top 6개만)
    # =========================
    if frames_6 is None:
        used_mode = "fallback_slide_top6"
        win_samples = ANIMA1_N_SAMPLES
        hop_samples = int(HOP_SEC * SR)
        all_frames = _frame_audio(audio, win_samples=win_samples, hop_samples=hop_samples)  # (B,24000)

        # 모든 프레임을 다 추론하면 의미가 없으니,
        # 빠른 "고음 에너지 proxy"로 프레임별 점수를 만들어 상위 6개만 추려냄
        # (간단히 frame RMS로도 되지만 팀장 의도(고음/피치)와 맞추기 위해 highfreq proxy 사용)
        # 여기서는 간단하게 프레임별 RMS + 약간의 고주파 성향을 섞은 점수
        # (너무 복잡하게 하면 다시 비용 증가)
        rms_scores = np.sqrt(np.mean(np.square(all_frames), axis=1) + 1e-12)  # (B,)
        top_idx = np.argsort(-rms_scores)[:ANIMA1_FALLBACK_TOPK]
        frames_6 = all_frames[top_idx].astype(np.float32)

        # 만약 B < 6인 상황(짧은 파일)에서도 1개 이상은 나오므로 안전

    # =========================
    # 6개 프레임 -> spec -> 추론
    # =========================
    x = _wav_to_anima1_spec_batch(frames_6)  # (6, 90, 513, 1)

    def predict_fn(inp: np.ndarray) -> np.ndarray:
        out = model_loader.ANIMA1_INFER(spec=tf.convert_to_tensor(inp, dtype=tf.float32))
        return out["cough_probability"].numpy()

    outsum = _predict_prob(predict_fn, x)
    probs = np.array(outsum["window_probs"], dtype=np.float32)

    # 팀장 요청: 6개 평균
    if ANIMA1_AGGREGATION == "mean":
        cough_probability = float(np.mean(probs))
        agg = "mean"
    elif ANIMA1_AGGREGATION == "topk3":
        k = min(3, probs.shape[0])
        cough_probability = float(np.mean(np.sort(probs)[-k:]))
        agg = "topk3_mean"
    else:
        cough_probability = float(np.max(probs))
        agg = "max"

    pd_threshold = float(model_loader.ANIMA1_THRESHOLD)
    is_cough = bool(cough_probability >= pd_threshold)

    # detail에 mode/agg를 남겨서 디버깅 가능하게
    outsum["aggregation"] = agg
    outsum["final_prob_used"] = cough_probability 
    outsum["mode_used"] = used_mode

    return {
        "ok": True,
        "is_cough": is_cough,
        "cough_probability": cough_probability,
        "pd_threshold": pd_threshold,
        "model_version": model_loader.ANIMA1_MODEL_VERSION,
        "detail": outsum,
    }

# 기침 분석 모델 (anima2) 로드
def analyze_anima2(wav_path: str, include_raw: bool = False) -> Dict[str, Any]:
    if model_loader.ANIMA2_MODEL is None:
        return {"ok": False, "reason": "anima2 모델 없음"}
    if model_loader.ANIMA2_CENTROID is None:
        return {"ok": False, "reason": "centroid 없음"}

    x = _prepare_input_for_model(wav_path)
    emb_batch = np.asarray(model_loader.ANIMA2_MODEL.predict(x, verbose=0), dtype=np.float32)  # (B, 64)

    # 윈도우 여러 개면 하나로 요약 (mean 추천)
    emb_mean = emb_batch.mean(axis=0)  # (64,)

    # 거리 계산
    if model_loader.ANIMA2_DISTANCE_METRIC == "euclidean":
        abnormality = _euclidean_distance(emb_mean, model_loader.ANIMA2_CENTROID)
        similarity = 1.0 / (1.0 + float(abnormality))                         # (0,1]
    else:
        abnormality = _cosine_distance(emb_mean, model_loader.ANIMA2_CENTROID)
        similarity = 1.0 - float(abnormality)                                 # cosine_similarity

    # 혹시 수치가 살짝 튀는 걸 방지(특히 cosine)
    similarity = float(np.clip(similarity, -1.0, 1.0))

    # dist 값이 클수록 centroid에서 멀다 = 비정상 가능성↑
    result = {
        "ok": True,
        "distance_metric": model_loader.ANIMA2_DISTANCE_METRIC,
        "abnormality": float(abnormality),
        "similarity": similarity,
        "num_windows": int(emb_batch.shape[0]),
        "embedding_mean": emb_mean.tolist(),     # ✅ DB raw_embeddings 저장용 :contentReference[oaicite:13]{index=13}
    }

    if include_raw:
        result["raw_embedding_by_window"] = emb_batch.tolist()

    return result