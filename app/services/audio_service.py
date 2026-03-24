from pathlib import Path
from typing import Tuple
import json, subprocess, librosa
import numpy as np

# 음성파일의 형식을 ffmpeg로 분석
def probe_audio(file_path : str) -> dict:
    cmd = [
        "ffprobe",
        "-v","error",
        "-select_streams","a:0",
        "-show_entries","stream=codec_name,sample_rate,channels",
        "-show_entries","format=duration",
        "-of","json",
        file_path
    ]

    # cmd 명령어(위에 내용) 실행 결과 저장
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0: # 실패시 예외 출력
        raise RuntimeError(f"ffprobe Failed: {result.stderr}")
    
    return json.loads(result.stdout)

# 업로드 음성 전처리
def convert_audio_to_wav(input_path: str, output_path: str) -> str:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:{result.stderr}")
    return output_path

# 변환된 음성 파일 검증
def validate_audio(audio_info:dict, min_duration: float = 1.0) -> dict:
    try:
        streams = audio_info["streams"]
        if not streams:
            return {"ok":False, "reason":"audio_stream_not_found"}

        stream = streams[0]
        duration = float(audio_info["format"]["duration"])
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        codec_name = stream["codec_name"]

    except Exception:   # 음성파일 여부 검증
        return {"ok":False, "reason":"unknown_audio_format"}
    if codec_name != "pcm_s16le":    # ?
        return {"ok":False, "reason":"invalid_codec"}
    if sample_rate != 16000:    # 샘플레이트 검증
        return {"ok": False, "reason": "invalid_sample_rate"}
    if channels != 1:   # 채널 검증
        return {"ok": False, "reason": "invalid_channels"}
    if duration < min_duration: # 음성 파일 최소 길이 검증
        return {"ok": False, "reason": "too_short"}
    
    # 검증 결과 출력
    return {
        "ok":True,
        "duration":duration,
        "sample_rate":sample_rate,
        "channels":channels,
        "codec_name":codec_name
    }

# 업로드 음성을 모델에 맞도록 길이 조정
def pad_or_crop_1d(x: np.ndarray, target_len:int) -> np.ndarray:
    if len(x) < target_len: # 목표 길이보다 짧으면
        return np.pad(x, (0, target_len - len(x))) # 패딩을 넣어서 길이를 맞춤
    return x[:target_len] # 목표 길이보다 길면 앞에서부터 짜름

# 음성파일을 모델이 쓸 수 있는 형태로 변환
# 여기에 쓰인 매개변수들은 나중에 빼서 따로 관리
def wav_to_logmel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 128,
    n_mels: int = 128,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel.astype(np.float32)

# spectrogram 크기를 모델에 맞게 조절
# pad_or_crop_1d 에서 이미 하고 있지만, 예외상황을 대비하기 위한 함수
def resize_spectrogram(spec: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    spec_resized = librosa.util.fix_length(spec, size=target_w, axis=1)

    if spec_resized.shape[0] < target_h:
        pad_h = target_h - spec_resized.shape[0]
        spec_resized = np.pad(spec_resized, ((0, pad_h), (0, 0)))
    else:
        spec_resized = spec_resized[:target_h, :]

    return spec_resized.astype(np.float32)

# (H, W) 형태의 spectrogram을 모델이 요구하는 (1, H, W, 1) 형태로 바꾸는 함수
def make_model_input(spec: np.ndarray) -> np.ndarray:
    x = np.expand_dims(spec, axis=-1)   # (H, W, 1)
    x = np.expand_dims(x, axis=0)       # (1, H, W, 1)
    return x.astype(np.float32)         # 

# 전처리 함수를 하나로 묶어서 실행하고, 입력값을 반환하는 함수
def preprocess_for_model(
    wav_path: str,
    target_sec: float = 1.5,
    sr: int = 16000,
    target_hw: Tuple[int, int] = (128, 188),
) -> np.ndarray:
    # librosa.load 함수가 2개의 audio_array, sample_rate 2개의 튜플을 반환함
    # 샘플레이트는 이미 변환을 통해 보정되어 있으므로 검증이 필요 없어서 _ 변수로 무시
    audio, _ = librosa.load(wav_path, sr=sr, mono=True) 

    target_len = int(sr * target_sec)
    audio = pad_or_crop_1d(audio, target_len)

    spec = wav_to_logmel_spectrogram(audio, sr=sr)
    spec = resize_spectrogram(spec, target_hw)

    x = make_model_input(spec)
    return x

# 서버 입출력 처리 테스트용 페이크 모델
def fake_model_predict(x:np.ndarray) -> dict:
    prob = 0.85
    return {
        "is_cough":prob >= 0.7,
        "prob":prob,
        "input_shape":list(x.shape),
        "model":"fake_model"
    }