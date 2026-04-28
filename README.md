# ANIMA 서버 README

(마지막 수정: 2026-04 최신화)

---

## 1. 개요

ANIMA 서버는 사용자가 업로드한 기침 음성을 분석하여

* 1단계: 기침 여부 판단
* 2단계: 건강 기준(healthy reference) 대비 이탈도 계산
* 품질 평가 (quality score)
* 최종 점수 계산 (final score)
* 분석 결과 저장 및 동기화

를 수행하는 FastAPI 기반 서버입니다.

서버 DB는 분석 기록을 저장하며,
앱은 로컬 DB와 서버 DB를 동기화하여 사용자 기록을 관리합니다.

---

## 2. 현재 구현 범위

* 음성 파일 업로드
* ffprobe 기반 오디오 검증
* ffmpeg 기반 WAV 변환
* WAV validation
* 모델 입력 전처리
* 1단계 기침 분류
* 2단계 healthy reference distance 분석
* 녹음 품질 평가 (quality score)
* 최종 점수 계산 (final score)
* 분석 결과 DB 저장
* 앱과의 동기화 API 제공

---

## 3. 기술 스택

* FastAPI
* SQLite
* TensorFlow / Keras
* librosa
* ffmpeg / ffprobe

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
│   ├── upload_router.py
│   └── record_router.py
├── services/
│   ├── audio_service.py
│   ├── model_service.py
│   ├── prediction_service.py
│   ├── score_service.py
│   └── record_service.py
└── data/
```

---

## 5. 전체 분석 흐름

```text
음성 업로드
 → 오디오 검증
 → WAV 변환
 → validation
 → quality_score 계산
 → 모델 예측 (prob)
 → final_score 계산
 → 최종 판정 생성
 → DB 저장
 → JSON 응답
```

---

## 6. 점수 계산 구조

### quality_score

녹음 품질 평가 점수

구성 요소:

* SNR
* 녹음 길이
* 클리핑 비율
* 에너지 패턴

```text
quality_score = 0.4*SNR + 0.2*duration + 0.2*clipping + 0.2*energy
```

---

### final_score

```text
final_score = (0.7 * cough_confidence) + (0.3 * quality_score)
```

---

### 최종 판정 기준

```text
≥ 0.6 → COUGH_HIGH (기침, 신뢰도 높음)
≥ 0.3 → COUGH_LOW (기침 가능성 있음)
< 0.3 → NO_COUGH (기침 아님)
```

---

## 7. API 명세

### POST /upload

기침 분석 수행

### Response (최신)

```json
{
  "ok": true,
  "prediction": {...},
  "quality": {
    "quality_score": 0.82
  },
  "final_prediction": {
    "final_score": 0.88,
    "final_detected": true,
    "final_label": "COUGH_HIGH",
    "final_message": "기침 (신뢰도 높음)"
  }
}
```

---

### POST /records/sync

앱 → 서버 기록 동기화

* record_uuid 기준으로 INSERT or UPDATE

---

### GET /records

서버 → 앱 기록 조회

---

### GET /health

서버 상태 확인

---

## 8. DB 구조 (MRRecord)

### 주요 컬럼

* record_uuid
* user_uuid
* cough_detected
* cough_confidence
* quality_score
* final_score
* final_label
* report_text
* healthy_distance
* risk_level
* audio_duration_sec
* sync_status
* synced_at
* created_at
* updated_at

---

## 9. 동기화 구조

앱과 서버는 다음 기준으로 동기화됨:

```text
record_uuid (고유 ID)
user_uuid
updated_at
```

동작 방식:

1. 앱 로컬 저장 → sync_status = PENDING
2. 서버로 업로드 (/records/sync)
3. 서버 DB 저장
4. 앱 → sync_status = SYNCED
5. 필요 시 서버 → 앱 다운로드

---

## 10. 오디오 처리 정책

* format: wav
* codec: pcm_s16le
* sample_rate: 16000
* channels: 1

validation 조건:

* duration ≥ MIN_DURATION
* mono
* 16kHz

---

## 11. 전처리 규격

* 입력 길이: 1.5초
* shape: (90, 513, 1)

과정:

1. RMS 이벤트 탐지
2. 이벤트 선택
3. 길이 보정
4. STFT
5. log spectrogram

---

## 12. 모델 구조

### Stage 1

* 기침 여부 분류

### Stage 2

* embedding 추출
* Mahalanobis distance 계산
* 정상 기준 비교

---

## 13. 현재 상태 요약

```text
분석 기능: 완료
품질 평가: 완료
최종 판정: 완료
DB 저장: 완료
동기화 API: 완료
```