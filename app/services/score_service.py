# app/services/score_service.py

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))


def calculate_snr_db(audio: np.ndarray, silence_threshold: float = 0.01) -> float:
    """
    전체 신호와 무음 구간의 RMS를 비교해 SNR(dB)을 계산.
    silence_threshold 이하의 절대 진폭을 무음 후보로 간주.
    """
    if audio is None or len(audio) == 0:
        return 0.0

    audio = np.asarray(audio, dtype=np.float32).flatten()

    signal_rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) > 0 else 0.0

    silence_mask = np.abs(audio) < silence_threshold
    silence_samples = audio[silence_mask]

    if len(silence_samples) == 0:
        # 완전한 무음 구간이 없으면 전체 RMS를 기준으로 아주 작은 noise floor 부여
        noise_rms = max(signal_rms * 0.1, 1e-8)
    else:
        noise_rms = float(np.sqrt(np.mean(np.square(silence_samples))))
        noise_rms = max(noise_rms, 1e-8)

    signal_rms = max(signal_rms, 1e-8)

    snr_db = 20.0 * math.log10(signal_rms / noise_rms)
    return round(snr_db, 4)


def score_snr(snr_db: float) -> float:
    """
    SNR 점수화
    """
    if snr_db >= 20:
        return 1.0
    elif snr_db >= 10:
        return 0.7
    elif snr_db >= 5:
        return 0.4
    else:
        return 0.1


def score_duration(duration_sec: float) -> float:
    """
    녹음 길이 점수화
    """
    duration_sec = max(duration_sec, 0.0)

    if 1.0 <= duration_sec <= 10.0:
        return 1.0
    elif duration_sec < 1.0:
        return round(_clamp(duration_sec / 1.0), 4)
    else:
        return 0.7


def calculate_clipping_ratio(audio: np.ndarray, clipping_threshold: float = 0.99) -> float:
    """
    클리핑 비율 계산
    - 오디오가 -1.0 ~ 1.0 정규화 상태라고 가정
    """
    if audio is None or len(audio) == 0:
        return 1.0

    audio = np.asarray(audio, dtype=np.float32).flatten()
    clipped = np.abs(audio) >= clipping_threshold
    clipping_ratio = float(np.mean(clipped)) if len(audio) > 0 else 1.0
    return round(clipping_ratio, 6)


def score_clipping(clipping_ratio: float) -> float:
    """
    클리핑 비율 점수화
    """
    if clipping_ratio < 0.01:
        return 1.0
    elif clipping_ratio < 0.05:
        return 0.7
    else:
        return 0.3


def calculate_energy_ratio(
    audio: np.ndarray,
    frame_length: int = 1024,
    hop_length: int = 512,
) -> float:
    """
    프레임 단위 RMS 에너지에서
    peak / mean 비율을 계산해 순간적 에너지 변화 패턴을 측정
    """
    if audio is None or len(audio) == 0:
        return 0.0

    audio = np.asarray(audio, dtype=np.float32).flatten()

    if len(audio) < frame_length:
        rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) > 0 else 0.0
        mean_energy = max(rms, 1e-8)
        peak_energy = rms
        return round(peak_energy / mean_energy, 4)

    frame_energies = []
    for start in range(0, len(audio) - frame_length + 1, hop_length):
        frame = audio[start:start + frame_length]
        rms = float(np.sqrt(np.mean(np.square(frame))))
        frame_energies.append(rms)

    if not frame_energies:
        return 0.0

    peak_energy = max(frame_energies)
    mean_energy = max(float(np.mean(frame_energies)), 1e-8)

    energy_ratio = peak_energy / mean_energy
    return round(energy_ratio, 4)


def score_energy_pattern(energy_ratio: float) -> float:
    """
    기침 특유의 순간적인 에너지 변화 패턴 점수화
    """
    if energy_ratio >= 5:
        return 1.0
    elif energy_ratio >= 3:
        return 0.7
    else:
        return 0.3


def calculate_quality_score(
    audio: np.ndarray,
    sample_rate: int,
    duration_sec: float | None = None,
) -> dict[str, Any]:
    """
    quality_score 계산 메인 함수
    반환:
    {
        "quality_score": 0.43,
        "quality_flag": "LOW",
        "message": "...",
        "details": {
            "snr_db": ...,
            "snr_score": ...,
            "duration_score": ...,
            "clipping_ratio": ...,
            "clipping_score": ...,
            "energy_ratio": ...,
            "energy_pattern_score": ...
        }
    }
    """
    if audio is None or len(audio) == 0:
        return {
            "quality_score": 0.0,
            "quality_flag": "LOW",
            "message": "오디오 데이터가 비어 있어 품질을 평가할 수 없습니다.",
            "details": {
                "snr_db": 0.0,
                "snr_score": 0.0,
                "duration_score": 0.0,
                "clipping_ratio": 1.0,
                "clipping_score": 0.0,
                "energy_ratio": 0.0,
                "energy_pattern_score": 0.0,
            },
        }

    audio = np.asarray(audio, dtype=np.float32).flatten()

    if duration_sec is None:
        duration_sec = len(audio) / sample_rate if sample_rate > 0 else 0.0

    snr_db = calculate_snr_db(audio)
    snr_score = score_snr(snr_db)

    duration_score = score_duration(duration_sec)

    clipping_ratio = calculate_clipping_ratio(audio)
    clipping_score = score_clipping(clipping_ratio)

    energy_ratio = calculate_energy_ratio(audio)
    energy_pattern_score = score_energy_pattern(energy_ratio)

    quality_score = (
        0.4 * snr_score
        + 0.2 * duration_score
        + 0.2 * clipping_score
        + 0.2 * energy_pattern_score
    )
    quality_score = round(_clamp(quality_score), 4)

    if quality_score >= 0.6:
        quality_flag = "OK"
        message = "녹음 품질이 양호합니다."
    else:
        quality_flag = "LOW"
        message = "녹음 품질이 낮아 정확도가 떨어질 수 있습니다."

    return {
        "quality_score": quality_score,
        "quality_flag": quality_flag,
        "message": message,
        "details": {
            "snr_db": round(snr_db, 4),
            "snr_score": round(snr_score, 4),
            "duration_score": round(duration_score, 4),
            "clipping_ratio": round(clipping_ratio, 6),
            "clipping_score": round(clipping_score, 4),
            "energy_ratio": round(energy_ratio, 4),
            "energy_pattern_score": round(energy_pattern_score, 4),
        },
    }


def calculate_final_score(cough_confidence: float, quality_score: float | None) -> float:
    """
    최종 점수 계산
    final_score = (0.7 * cough_confidence) + (0.3 * quality_score)
    """
    cough_confidence = _clamp(_safe_float(cough_confidence))
    quality_score = _clamp(_safe_float(quality_score))

    final_score = (0.7 * cough_confidence) + (0.3 * quality_score)
    return round(_clamp(final_score), 4)


def get_final_prediction(final_score: float) -> dict[str, Any]:
    """
    최종 점수에 따른 판정
    """
    final_score = _clamp(_safe_float(final_score))

    if final_score >= 0.6:
        return {
            "final_detected": True,
            "final_label": "COUGH_HIGH",
            "final_message": "기침 (신뢰도 높음)",
        }
    elif final_score >= 0.3:
        return {
            "final_detected": True,
            "final_label": "COUGH_LOW",
            "final_message": "기침 가능성 있음 (주의)",
        }
    else:
        return {
            "final_detected": False,
            "final_label": "NO_COUGH",
            "final_message": "기침 아님",
        }