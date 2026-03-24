# ANIMA 프로젝트 통합 설계 문서
(마지막 수정 2026/3/24)

## 개요

### 프로젝트 목적
- 사용자의 기침 음성을 분석하여 건강 상태를 평가하는 서비스
- AI 기반으로 기침 여부 및 이상 정도를 분석하고 결과를 시각화

### 주요 기능
- 기침 음성 녹음 및 업로드
- 기침 여부 판단
- 기침 이상도 분석 (정상 대비)
- 결과 리포트 제공
- 기록 저장 및 히스토리 조회

### 설계 원칙 (음성 처리)
본 서버는 다양한 사용자 입력 환경을 고려하여  
업로드된 원본 파일을 직접 검증하지 않고,  
ffmpeg를 통해 표준 포맷으로 변환한 뒤  
변환된 결과를 기준으로 validation(유효성 검사)을 수행한다.

이를 통해 입력 파일 형식에 대한 의존성을 줄이고,  
모델 입력의 일관성을 유지한다.

---

## 사용자 흐름

1. 사용자가 앱에서 기침 음성 녹음
2. 건강 정보 입력 (통증, 기침 횟수, 메모 등)
3. 서버로 데이터 전송
4. 서버에서 오디오 전처리
5. AI 모델 분석 수행
6. 결과 생성 (점수, 그래프, 텍스트)
7. DB 저장
8. 앱으로 결과 반환
9. 사용자는 히스토리 조회 가능

---

## 데이터 정의

### 입력 데이터 (앱 → 서버)

| 필드명 | 타입 | 설명 |
|--------|------|------|
| file | audio | 기침 음성 파일 |
| user_id | string | 사용자 ID |
| pain_score | int | 통증 정도 (0~10) |
| cough_count | int | 기침 횟수 |
| memo | string | 증상 메모 |
| medication_taken | bool | 복약 여부 |
| medication_time | datetime | 복약 시각 |

---

### 서버 내부 데이터

| 필드명 | 설명 |
|--------|------|
| audio_path | 저장된 파일 경로 |
| duration | 오디오 길이 |
| sample_rate | 샘플레이트 |

---

### 분석 결과 데이터 (서버 → 앱)

| 필드명 | 설명 |
|--------|------|
| ok | 처리 성공 여부 |
| is_cough | 기침 여부 |
| cough_confidence | 기침 확률 |
| quality_score | 음질 점수 |
| deviation_score | 정상 대비 이상도 |
| distance | 내부 거리값 |
| report_text | 결과 설명 문장 |
| graph_data | 그래프용 데이터 |
| created_at | 생성 시간 |

---

## API 명세

### 음성 분석

#### POST /analyze

**Request**
```json
{
  "file": "...",
  "user_id": "user123",
  "pain_score": 3,
  "cough_count": 5
}
```
**Response**

```json
{
  "ok": true,
  "is_cough": true,
  "cough_confidence": 0.92,
  "quality_score": 0.85,
  "deviation_score": 72,
  "report_text": "평균보다 기침 패턴이 다소 불안정합니다.",
  "graph_data": [0.1, 0.3, 0.8]
}
```

### 기록 조회

#### GET /records

### 상세 조회

#### GET /records/{id}

---

## 파일 처리 정책
- 업로드된 파일은 app/data/uploads 에 저장
- 변환된 파일은 app/data/converted 에 저장
- 다음의 경우 파일 자동 삭제
  - 오디오 파일이 아닌 경우
  - wav 변환 실패
  - validation 실패

---

## 디렉토리 구조
app/
 ├── main.py
 ├── routers/
 │    └── upload_router.py
 ├── services/
 │    └── audio_service.py
 └── data/
      ├── uploads/
      └── converted/

---
## DB 설계

### users
| 컬럼 | 타입 |
|------|------|
| id | varchar |
| created_at | datetime |

### cough_records
| 컬럼 | 타입 |
|------|------|
| id | int |
| user_id | varchar |
| audio_path | text |
| created_at | datetime |

### analysis_results
| 컬럼 | 타입 |
|------|------|
| record_id | int |
| is_cough | bool |
| confidence | float |
| deviation_score | float |
| report_text | text |

---

## 모델 입출력 규격

### 입력
- 입력 형태 : spectrogram (H,W,1)
- dtype : float32

### 모델 출력

#### 1단계 모델 (기침 여부 판단)
- 출력 : 확률값 (0~1)
- threshold : 0.7

#### 2단계 모델 (이상도 분석)
- 출력:
  - distance
  - deviation_score
  - percentile

---

## 전처리 흐름

### 오디오 표준 포맷

모든 입력 오디오는 아래 포맷으로 변환한다.
- format : wav
- codec : pcm_s16le
- sample_rate : 16000Hz
- channels : 1 (mono)

### 처리 단계 (파이프 라인)

1. 파일 저장 (임시 경로)
2. ffprobe로 원본 파일 검사
   - audio stream 존재 여부 확인

3. ffmpeg로 표준 포맷 변환
   - wav (pcm_s16le)
   - 16000Hz
   - mono

4. ffprobe로 변환 결과 검증
   - sample_rate
   - channels
   - duration

5. validation 수행
   - 최소 길이 검사
   - 손상 여부 확인
     
6. 길이 조정 (pad / crop)
7. spectrogram 생성
8. 모델 입력 변환

---

## 결과 리포트 규칙
### 점수 해석
| 점수 | 의미 |
|------|------|
| 0 ~ 30 | 정상 |
| 30 ~ 70 | 주의 |
| 70 ~ 100 | 위험 |

### 텍스트 생성 규칙
- deviation_score 기반으로 문구 선택
- 예시 :
  - 낮음 -> "안정적인 상태입니다."
  - 중간 -> "주의가 필요합니다."
  - 높음 -> "이상 패턴이 감지되었습니다."

---

## 에러처리
| 코드 | 상황 |
|------|------|
| invalid_file | 오디오 스트림 없음 |
| conversion_failed | ffmpeg 변환 실패 |
| invalid_audio | 변환 후 스펙 불일치 |
| too_short | 길이 부족 |
| not_cough | 기침 아님 |
| server_error | 서버 오류 |

---

## 향후 변경 가능성
- 실제 기침 검증 모델 연결
- /analyze 엔드포인트로 통합
- 분석 결과 DB 저장
- 사용자 히스토리 조회 기능
- AWS S3 업로드 및 관리
- 2차 기침 분석 모델 연결

---

## 메모
- 현재 모델은 테스트용
- 실제 환경에서는 S3 업로드
- DB는 MYSQL 사용
