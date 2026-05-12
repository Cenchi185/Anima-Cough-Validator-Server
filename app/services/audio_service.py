from typing import Tuple,List
import json, subprocess, librosa
import numpy as np

from app.core.config import (
    DEBUG,
    TARGET_SR,
    TARGET_SEC,
    TARGET_HW,
    N_FFT,
    HOP_LENGTH,
)

# 디버그 출력 함수
def debug_print(*args):
    if DEBUG:
        print(*args)

def probe_audio(file_path : str) -> dict:
    ''' 음성파일의 형식을 ffmpeg로 분석 '''
    # ffprobe 를 사용해 파일을 분석
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

    if result.returncode != 0: # 분석 실패, 에러 리턴
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout) # ffprobe 결과를 저장
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe json parse failed: {e}")

    streams = data.get("streams", []) # 분석 내역에 stream 이 있는지 확인
    if not streams:
        raise RuntimeError("오디오 스트림이 없음")

    return data

def convert_audio_to_wav(input_path: str, output_path: str) -> str:
    ''' 업로드 음성 전처리 '''
    # ffmpeg 로 1채널, 샘플레이트 16000Hz, pcm_s16le 코덱, output_path 에 wav 형식으로 변환 후 저장
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

def validate_audio(audio_info:dict, min_duration: float = 1.0) -> dict:
    ''' 변환된 음성 파일 검증 '''
    try:
        streams = audio_info["streams"]
        if not streams:
            return {"ok":False, "reason":"오디오 스트림이 없음"}

        stream = streams[0]
        duration = float(audio_info["format"]["duration"])
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        codec_name = stream["codec_name"]

    except Exception:   # 음성파일 여부 검증
        return {"ok":False, "reason":"unknown_audio_format"}
    if codec_name != "pcm_s16le":    # 코덱 pcm_s16le 인지 검증
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

# 레퍼런스 맞춤용 전처리 코드
EVENT_FRAME_MS = 20
EVENT_HOP_MS = 10
EVENT_FRAME_LEN = int(TARGET_SR * (EVENT_FRAME_MS / 1000.0))
EVENT_HOP_LEN = int(TARGET_SR * (EVENT_HOP_MS / 1000.0))

K_STD = 1.0
MIN_EVENT_SEC = 0.05
MIN_EVENT_FRAMES = int(MIN_EVENT_SEC / (EVENT_HOP_MS / 1000.0))
MERGE_GAP_SEC = 0.15
MERGE_GAP_FRAMES = int(MERGE_GAP_SEC / (EVENT_HOP_MS / 1000.0))
MAX_EVENTS_PER_FILE = 3

def fix_length_1p5s(y: np.ndarray, target_len: int) -> np.ndarray:
    """
    길면 가운데 기준으로 1.5초 crop
    짧으면 뒤에 0 padding
    """
    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        return y[start:start + target_len]
    pad = target_len - len(y)
    return np.pad(y, (0, pad), mode="constant")

def detect_cough_events_rms(
    y: np.ndarray,
    k_std: float = K_STD,
    max_events_per_file: int = MAX_EVENTS_PER_FILE,
) -> List[Tuple[int, int]]:
    ''' 
    RMS(현재 구간의 평균적인 소리 세기) 기반 기침 이벤트 구간 탐지
    오디오 전체중, 에너지가 큰 구간(기침)을 찾음
    '''
    if y is None or len(y) < EVENT_FRAME_LEN:
        return []

    rms = librosa.feature.rms(
        y=y,
        frame_length=EVENT_FRAME_LEN,
        hop_length=EVENT_HOP_LEN
    )[0]

    mu = float(np.mean(rms))
    sigma = float(np.std(rms)) + 1e-10
    thr = mu + k_std * sigma
    active = rms > thr

    events = []
    i, n = 0, len(active)

    while i < n:
        if not active[i]:
            i += 1
            continue

        j = i
        while j < n and active[j]:
            j += 1

        if (j - i) >= MIN_EVENT_FRAMES:
            events.append((i, j - 1))
        i = j

    if not events:
        return []

    merged = []
    cur_s, cur_e = events[0]
    
    for s, e in events[1:]:
        if s - cur_e <= MERGE_GAP_FRAMES:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
            
    merged.append((cur_s, cur_e))

    sample_events = []
    for fs, fe in merged:
        start = fs * EVENT_HOP_LEN
        end = min(len(y), fe * EVENT_HOP_LEN + EVENT_FRAME_LEN)
        if end > start:
            sample_events.append((start, end))

    scored = []
    for start, end in sample_events:
        seg = y[start:end]
        score = float(np.sqrt(np.mean(seg * seg)) + 1e-12)
        scored.append((score, start, end))

    scored.sort(reverse=True, key=lambda x: x[0])
    
    return [(s, e) for _, s, e in scored[:max_events_per_file]]

def wav_to_log_spec_tensor(
    wav_1p5: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    target_hw: Tuple[int, int] = TARGET_HW,
) -> np.ndarray:
    """
    오디오 파형을 2차원 이미지 데이터로 변환
    stft -> abs -> log(x + 1e-6)
    최종 shape: (90, 513)
    """
    target_frames, target_bins = target_hw

    stft = librosa.stft(
        y=wav_1p5.astype(np.float32),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        center=False,
    )

    spec = np.abs(stft) # 주파수의 크기만 가져옴
    spec = np.log(spec + 1e-6).astype(np.float32) # 값의 차이(저음/고음)가 너무 크지 않도록 조절

    # librosa 결과는 (freq, time) 이므로 (time, freq)로 전치
    spec = spec.T
    # reference 코드처럼 앞에서부터 [:90, :513]
    spec = spec[:target_frames, :target_bins]

    # 혹시 모자라면 pad
    cur_t, cur_f = spec.shape

    if cur_t < target_frames:
        spec = np.pad(spec, ((0, target_frames - cur_t), (0, 0)), mode="constant")
    if cur_f < target_bins:
        spec = np.pad(spec, ((0, 0), (0, target_bins - cur_f)), mode="constant")

    return spec.astype(np.float32)

def make_model_input(spec: np.ndarray) -> np.ndarray:
    '''모델 입력이 가능한 형태로 변환'''
    x = np.expand_dims(spec, axis=-1)   # (90, 513, 1) 맨뒤에 차원 추가
    x = np.expand_dims(x, axis=0)       # (1, 90, 513, 1) 맨앞에 차원 추가
    return x.astype(np.float32)

def preprocess_for_model(
    wav_path: str | None = None,
    audio : np.ndarray | None =None,
    target_sec: float = TARGET_SEC,
    sr: int = TARGET_SR,
    target_hw: Tuple[int, int] = TARGET_HW,
) -> np.ndarray:
    ''' 모델 입력용 전처리 '''
    if audio is None :
        if wav_path is None :
            raise ValueError("wav_path 또는 audio 중 하나 필요")
        audio, _ = librosa.load(wav_path, sr=sr, mono=True) # wav 파일 읽기

    target_len = int(sr * target_sec)

    events = detect_cough_events_rms(audio) # 기침 이벤트 탐지

    # 이벤트가 있으면 가장 강한 이벤트 사용, 없으면 전체 오디오 사용
    if events:
        start, end = events[0]
        audio_seg = audio[start:end]
        print(f"event detected: start={start}, end={end}, len={end-start}")
    else:
        audio_seg = audio
        print("event detected: none -> fallback to full audio")

    audio_fixed = fix_length_1p5s(audio_seg, target_len) # 1.5초 길이 맞추기

    # STFT(abs) -> log(x + 1e-6)
    spec = wav_to_log_spec_tensor(
        wav_1p5=audio_fixed,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        target_hw=target_hw,
    )

    debug_print("[SPEC] shape =", spec.shape)
    x = make_model_input(spec)
    debug_print("[MODEL INPUT] shape =", x.shape)
    
    return x