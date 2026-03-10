import json
import subprocess
from typing import Any, Dict

import numpy as np
import soundfile as sf
import tensorflow as tf

# 모델에 사용할 변수
SR = 16000
# [0.0 ~ 1.5], [0.25 ~ 1.75], [0.5 ~ 2.0] ... 처럼 오디오 클립을 자름
WIN_SEC = 1.5   # 모델에 입력할 오디오 클립 단위
HOP_SEC = 0.25  # 다음 클립 추출 딜레이
# STFT는 소리를 잘개 쪼가서 각 조각을 '주파수 사진' 으로 찍어 분석 하는 방법
# 클립을 주파수 이미지로 변환할때 필요한 카메라 설정 값
FRAME_LENGTH = 256 # STFT 에서 한 프레임(장)에 쓸 샘플의 길이
FRAME_STEP = 128    # STFT 에서 프레임을 움직이는 간격
FFT_LENGTH = 256   # FFT 계산 길이(주파수 높낮이를 분석하는 크기)

'''
ffprobe 는 오디오,비디오 파일 분석 툴
cmd = [
    "사용할 툴"
    "에러만 출력",
    "스트림에서 뽑을 내용", "코덱(codec_name), 레이트(sample_rate), 채널(channels)"
    "스트림에서 뽑을 내용", "파일의 재생 시간(duration)"
    "JSON 으로 결과 출력"
]
의 내용
'''
def probe_wav(file_path : str) -> Dict[str, Any]:
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
    
    # JSON 형태로 반환
    return json.loads(result.stdout)

# 업로드 파일을 .wav 형식으로 변환하기
def convert_to_wav(input_path: str, output_path: str):
    """
    어떤 오디오 파일이든 16kHz mono wav로 변환
    """
    cmd = [
        "ffmpeg",
        "-y",                # 덮어쓰기
        "-i", input_path,    # 입력파일
        "-ac", "1",          # mono
        "-ar", "16000",      # 16kHz
        output_path
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# wav 파일을 float32 mono 파형으로 변환
# 모델은 wav 파일을 그대로 읽을수가 없기 때문에, 모델에 맞는 규격으로 변환하는 과정
def _read_wav_mono_16k(wav_path:str) -> np.ndarray:
    # soundfile로 wav 읽어서 float32 mono로 반환
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if sr != SR:
        raise ValueError(f"sample_rate_not_16k: {sr}")
    
    # 만약 audio 파일의 채널이 1채널(mono)이 아니라면 1채널로 변환
    # 1채널=소리만 있음, 2채널=소리가 어느쪽에서 나는지에 대한 정보도 있음
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)

# 모델의 학습기준길이(1.5초)에 맞춰서 오디오 클립을 자름 (모델에 입력될 값)
# 구버전(anima1) 분석용 기능
def _frame_audio(audio: np.ndarray, win_samples:int, hop_samples: int) -> np.ndarray:
    
    n = audio.shape[0]  # 전체 샘플 수
    # 오디오 클립이 너무 짧다면 무음을 이어붙여서 1.5초로 만들어줌
    if n < win_samples:
        pad = win_samples - n
        audio = np.pad(audio, (0,pad), mode="constant")
        n = audio.shape[0]

    # 1.5초 짜리 창을 0.25초씩 밀어서 프레임을 생성
    frames = []
    start = 0
    while start + win_samples <= n:
        frames.append(audio[start:start + win_samples])
        start += hop_samples

    # frames 가 비어있을 경우를 대비한 안전장치 (최소 1개는 생성되도록)
    if not frames:
        frames = [np.pad(audio, (0, max(0, win_samples - n)), mode="constant")[:win_samples]]

    # (프레임 갯수, 프레임 길이) 값을 반환함
    return np.stack(frames, axis=0).astype(np.float32)

# _frame_audio 에서 쪼개진 오디오 클립을 모델(Conv2D)이 분석할 수 있는 형태로 변환
def _wav_to_stft_magnitude_batch(audio_batch: np.ndarray) -> np.ndarray:
    # (B, N) tf.singnal.stft 가 받아들일 수 있는 입력 형태로 변환
    x = tf.convert_to_tensor(audio_batch, dtype=tf.float32)

    '''
    1.5초의 오디오 클립을
    frame_length마다 자르고
    frame_step씩 이동 (50%는 겹쳐짐)
    각 프레임 주파수는 FFT_LENGTH 만큼 분석
    마지막 조각에 프레임이 모자라면 그냥 버림 (pad_end=False)
    '''
    stft = tf.signal.stft(
        x,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )  # (B, T, F) complex64 << stft 에 저장되는 값 형태
    
    spec = tf.abs(stft)            
    spec = spec[..., tf.newaxis]   # (B, T, F, 1)
    return spec.numpy().astype(np.float32)

# _read_wav_mono_16k, _frame_audio, _wav_to_stft_magnitude_batch 를 실행
# 모델에 입력 할 수있는 값을 실제로 반환
def _prepare_input_for_model(wav_path: str) -> np.ndarray:
    # docs 기준: wav -> (윈도우 분할) -> STFT magnitude -> (B,T,F,1)
    audio = _read_wav_mono_16k(wav_path)

    win_samples = int(WIN_SEC * SR)
    hop_samples = int(HOP_SEC * SR)

    frames = _frame_audio(audio, win_samples=win_samples, hop_samples=hop_samples)  # (B, win)
    x = _wav_to_stft_magnitude_batch(frames)  # (B, T, F, 1)

    # 만약 모델이 고정된 input_shape을 강제한다면 여기서 체크 가능
    # (None, T, F, 1) 형태가 아닐 경우를 대비
    return x