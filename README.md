# ANIMA 서버 README
(마지막 수정: 2026-04 기준)

## 1. 개요

ANIMA 서버는 사용자가 업로드한 기침 음성을 분석하여

- 1단계: 기침 여부 판단
- 2단계: 건강 기준(healthy reference) 대비 이탈도 계산
- 분석 결과 저장
- 앱에 결과 반환

을 수행하는 FastAPI 기반 서버입니다.

현재 서버는 분석 결과를 SQLite DB에 저장하며, 앱은 이 응답을 받아 로컬 DB에 별도로 저장하여 사용자 기록 조회에 활용할 수 있습니다.

---

## 2. 현재 구현 범위

현재 구현된 기능은 다음과 같습니다.

- 음성 파일 업로드
- ffprobe 기반 오디오 확인
- ffmpeg 기반 WAV 변환
- 변환된 WAV 파일 validation
- 모델 입력용 전처리
- 1단계 기침 여부 판단
- 2단계 healthy reference 기반 distance 계산
- 분석 결과 DB 저장
- 분석 결과 JSON 응답 반환

현재 서버는 **기록 저장 기능까지 포함**하며,  
과거 기록 조회 API는 아직 구현 범위에 포함하지 않았습니다.

---

## 3. 기술 스택

- FastAPI
- SQLite
- TensorFlow / Keras
- librosa
- ffmpeg / ffprobe

---

## 4. 프로젝트 구조

```text
app/
├── main.py
├── core/
│   └── config.py
├── db/
│   └── database.py
├── routers/
│   └── upload_router.py
├── services/
│   ├── audio_service.py
│   ├── model_service.py
│   ├── prediction_service.py
│   └── record_service.py
└── data/
    ├── uploads/
    └── converted/
```

---

## 5. 실행 흐름
1. 클라이언트가 음성 파일을 /upload로 전송
2. 서버가 원본 파일을 저장
3. ffprobe로 오디오 여부 확인
4. ffmpeg로 WAV(16kHz, mono, pcm_s16le) 변환
5. 변환된 WAV 파일 validation
6. 전처리 후 모델 입력 생성
7. 1단계 기침 여부 예측
8. 기침으로 판단된 경우 2단계 distance 계산
9. 결과를 DB에 저장
10. JSON 응답 반환

---

## 6. API 명세

### POST /upload
기침 음성파일 업로드 및 분석결과 반환

### Request
Content-Type: multipart/form-data
필드: file : 업로드 음성 파일

### Response 예시
```
{
  "ok": true,
  "filename": "sample.wav",
  "validation": {
    "ok": true,
    "duration": 2.31,
    "sample_rate": 16000,
    "channels": 1,
    "codec_name": "pcm_s16le"
  },
  "prediction": {
    "is_cough": true,
    "prob": 0.93,
    "input_shape": [1, 90, 513, 1],
    "model": "best_1to1_embed64.keras",
    "stage2": {
      "skipped": false,
      "embedding_dim": 64,
      "distance": 6.84,
      "quantiles": {
        "p50": 5.457,
        "p80": 6.583,
        "p90": 7.532,
        "p95": 8.547,
        "p99": 11.115,
        "max": 13.912
      },
      "interpretation": {
        "level": "borderline",
        "message": "정상 범위에서 다소 벗어날 수 있습니다."
      }
    }
  },
  "mr_record": {
    "cough_detected": true,
    "cough_confidence": 0.93,
    "cough_threshold": 0.6,
    "quality_score": null,
    "quality_fail_reason": null,
    "report_text": "정상 범위에서 다소 벗어날 수 있습니다.",
    "healthy_distance": 6.84,
    "model_version": "best_1to1_embed64.keras",
    "audio_duration_sec": 2.31,
    "reference_dataset": "COUGHVID_HEALTHY",
    "distance_metric": "MAHALANOBIS",
    "reference_version": "healthy_ref_final.json",
    "risk_level": "CAUTION"
  },
  "db_saved": {
    "record_id": 1,
    "record_uuid": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "analysis_status": "ANALYZED",
    "server_received_at": "2026-04-14T12:00:00"
  }
}
```

---

## 7. 오디오 처리 정책

모든 입력 오디오는 아래 포맷으로 변환 후 검증합니다.
- format: wav
- codec: pcm_s16le
- sample_rate: 16000Hz
- channels: 1 (mono)

validation 기준
- 오디오 스트림 존재
- codec = pcm_s16le
- sample_rate = 16000
- channels = 1
- duration >= MIN_DURATION

---

## 8. 전처리 규격
- 입력 샘플레이트: 16000Hz
- 입력 길이: 1.5초
- 입력 shape: (90, 513, 1)
- dtype: float32

전처리 순서
1. RMS 기반 이벤트 탐지
2. 이벤트 구간 또는 전체 오디오 선택
3. 1.5초 길이 맞춤
4. STFT 수행
5. log spectrogram 생성
6. 모델 입력 tensor 변환

---

## 9. 모델 분석 구조
### 1단계: 기침 여부 판단
- 출력: 확률값 prob
- threshold: 0.6
- 결과:
  - is_cough = true → 2단계 진행
  - is_cough = false → 2단계 생략
### 2단계: healthy reference 기반 거리 계산
- embedding 추출
- Mahalanobis distance 계산
- healthy reference quantile 기준 해석
### 내부 stage2 interpretation level
- normal_range
- borderline
- abnormal_high
- abnormal_very_high
### DB 저장용 risk_level 매핑
- normal_range → GOOD
- borderline → CAUTION
- abnormal_high, abnormal_very_high → RISK

---

## 10. DB 구조

현재 서버는 SQLite를 사용하며, 주요 저장 테이블은 MRRecord입니다.

### MRRecord 주요 컬럼
- record_id
- record_uuid
- user_uuid
- analysis_status
- cough_detected
- cough_confidence
- cough_threshold
- quality_score
- quality_fail_reason
- healthy_distance
- reference_dataset
- distance_metric
- reference_version
- risk_level
- report_text
- model_version
- audio_duration_sec
- local_audio_path
- sync_status
- sync_retry_count
- last_sync_attempt_at
- server_received_at
- synced_at
- created_at
- updated_at
### analysis_status
- PENDING
- NON_COUGH
- ANALYZED
- FAILED

---

## 11. 파일 저장 정책
- 원본 파일 저장 경로: app/data/uploads
- 변환 파일 저장 경로: app/data/converted

다음 경우 파일은 삭제될 수 있습니다.
- 오디오 파일이 아닌 경우
- WAV 변환 실패
- validation 실패 (too_short, invalid_codec, invalid_sample_rate, invalid_channels 등)
