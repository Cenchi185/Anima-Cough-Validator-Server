# ANIMA Cough Validator Server

(Last Updated: 2026-04-28, Render 배포 기준)

---

## 1. 개요

ANIMA 서버는 사용자가 업로드한 음성 데이터를 분석하여
기침 여부를 판단하고, 건강 기준과 비교하여 결과를 제공하는 FastAPI 기반 서버입니다.

### 주요 기능

* 기침 여부 분석 (1단계 모델)
* 건강 기준 대비 거리 분석 (2단계)
* 녹음 품질 평가 (Quality Score)
* 최종 판단 점수 계산 (Final Score)
* 분석 기록 DB 저장
* 앱과의 데이터 동기화 지원

---

## 2. 배포 주소

```text
https://anima-cough-validator-server.onrender.com
```

### Swagger (API 테스트)

```text
https://anima-cough-validator-server.onrender.com/docs
```

---

## 3. 전체 처리 흐름

```text
음성 업로드
 → 오디오 검증 (ffprobe)
 → WAV 변환 (ffmpeg)
 → validation
 → quality_score 계산
 → 모델 예측 (prob)
 → final_score 계산
 → 최종 판정 생성
 → DB 저장
 → JSON 응답 반환
```

---

## 4. 점수 구조

### 4.1 Quality Score (녹음 품질)

구성 요소:

* SNR
* 녹음 길이
* 클리핑 비율
* 에너지 패턴

```text
quality_score = 0.4*SNR + 0.2*duration + 0.2*clipping + 0.2*energy
```

---

### 4.2 Final Score (최종 판단 점수)

```text
final_score = (0.7 * cough_confidence) + (0.3 * quality_score)
```

---

### 4.3 최종 판정 기준

```text
final_score ≥ 0.6 → COUGH_HIGH (기침, 신뢰도 높음)
final_score ≥ 0.3 → COUGH_LOW (기침 가능성 있음)
final_score < 0.3 → NO_COUGH (기침 아님)
```

---

## 5. API 명세

### 5.1 POST /upload

기침 음성 업로드 및 분석

#### Request

* Content-Type: multipart/form-data
* field name: `file`

#### Response 예시

```json
{
  "ok": true,
  "prediction": {...},
  "quality": {
    "quality_score": 0.94
  },
  "final_prediction": {
    "final_score": 0.982,
    "final_label": "COUGH_HIGH",
    "final_message": "기침 (신뢰도 높음)"
  }
}
```

---

### 5.2 GET /records

서버 DB에 저장된 분석 기록 조회

```text
GET /records?user_uuid=local
```

---

### 5.3 POST /records/sync

앱 로컬 DB → 서버 DB 동기화

* record_uuid 기준으로 INSERT 또는 UPDATE

---

### 5.4 GET /health

서버 상태 확인

```json
{
  "ok": true
}
```

---

## 6. DB 구조

### 테이블: MRRecord

#### 주요 컬럼

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

## 7. 동기화 구조

앱과 서버는 다음 기준으로 데이터 동기화 수행:

```text
record_uuid (고유 ID)
user_uuid
updated_at
```

### 동작 흐름

```text
앱 분석 결과 저장 (로컬 DB)
→ sync_status = PENDING
→ /records/sync 호출
→ 서버 DB 저장
→ sync_status = SYNCED
→ 필요 시 서버 → 앱 조회
```

---

## 8. 오디오 처리 정책

모든 입력 오디오는 아래 형식으로 변환 후 분석:

* format: wav
* codec: pcm_s16le
* sample_rate: 16000Hz
* channels: 1 (mono)

---

## 9. 전처리 규격

* 입력 길이: 1.5초
* 입력 shape: (90, 513, 1)

처리 과정:

1. RMS 기반 이벤트 탐지
2. 이벤트 구간 선택
3. 길이 맞춤
4. STFT
5. log spectrogram 생성

---

## 10. 모델 구조

### Stage 1

* 기침 여부 분류 (prob)

### Stage 2

* embedding 추출
* Mahalanobis distance 계산
* 정상 기준과 비교

---

## 11. 실행 방법 (로컬)

```bash
pip install -r requirements-final.txt
uvicorn app.main:app --reload
```

---

## 12. 배포 관련 설정

### Render 설정

* Python: 3.11
* Build Command:

  ```bash
  pip install -r requirements-final.txt
  ```
* Start Command:

  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port 10000
  ```

### 필수 파일

* `.python-version`
* `apt.txt` (ffmpeg)

---

## 13. 현재 상태

```text
분석 기능: 완료
품질 평가: 완료
최종 점수: 완료
DB 저장: 완료
동기화 API: 완료
외부 배포: 완료
```