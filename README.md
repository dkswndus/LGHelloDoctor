# LGHelloDoctor

AI 음성 문진 기반 지역 병원 연결 서비스

## 기술 스택

| 역할 | 기술 |
|------|------|
| STT | Whisper (Groq API) |
| LLM | Groq llama-3.3-70b |
| TTS | gTTS (Google Text-to-Speech) |
| 병원 검색 | 카카오맵 로컬 API |
| 백엔드 | FastAPI |
| 프론트엔드 | HTML / CSS / JavaScript |

---

## 폴더 구조

```
LGHelloDoctor/
├── backend/        # FastAPI 서버 (LLM, 병원 검색)
├── frontend/       # 웹 UI (HTML/CSS/JS)
├── deeplearning/   # STT(Whisper), TTS(gTTS)
└── .env            # API 키 설정
```

---

## 환경 설정

루트에 `.env` 파일 생성:

```
GROQ_API_KEY=your_groq_api_key
KAKAO_REST_API_KEY=your_kakao_rest_api_key
```

---

## 실행 방법

### 백엔드만 실행

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8001
```

API 문서 확인: `

---

### 프론트엔드만 확인

별도 서버 불필요. `frontend/index.html`을 브라우저에서 바로 열거나,
VSCode Live Server 확장을 사용하세요.

> 백엔드가 실행 중이어야 병원 검색 및 음성 분석이 동작합니다.

---

### 딥러닝(STT/TTS)만 테스트

```bash
# STT 테스트
cd deeplearning
python -c "
from stt_whisper import transcribe_audio_file
print(transcribe_audio_file('테스트파일.mp3', language='ko'))
"

# TTS 테스트
python -c "
from tts_kokoro import synthesize_speech
audio = synthesize_speech('안녕하세요, 가까운 정형외과를 안내해드립니다.')
with open('output.mp3', 'wb') as f:
    f.write(audio)
print('output.mp3 생성 완료')
"http://127.0.0.1:8001/docs`
```

---

### 전체 파이프라인 테스트

백엔드 실행 후 터미널에서:

```bash
python -c "
import os, sys
sys.path.insert(0, 'deeplearning')
sys.path.insert(0, 'backend')
from dotenv import load_dotenv
load_dotenv()
from stt_whisper import transcribe_audio_file
from backend.llm_gpt import generate_triage
from hospital_search import search_nearby_hospitals

text = transcribe_audio_file('테스트파일.mp3', language='ko')
print('STT:', text)
triage = generate_triage(text)
print('LLM:', triage)
hospitals = search_nearby_hospitals(triage['recommended_department'], 37.4947, 126.7118)
print('병원:', hospitals)
"
```

---

## 서비스 흐름

```
음성 입력
  ↓
[deeplearning] Whisper STT → 텍스트
  ↓
[backend] Groq LLM → 진료과 판단 + 안내 문장
  ↓
[backend] 카카오맵 API → 근처 병원 검색
  ↓
[deeplearning] gTTS → 음성 합성
  ↓
[frontend] 결과 표시 + 음성 재생
```

---

1. 서비스 개요

본 서비스는 LG HelloVision 주 이용층인 50대~80대 사용자를 대상으로,
사용자가 음성으로 증상을 말하면 이를 텍스트로 변환하고,
AI가 문진 내용을 이해 가능한 형태로 정리한 뒤,
적절한 진료과 및 인근 병원 연결을 지원하는 서비스이다.

본 서비스는 의료 진단을 제공하는 것이 아니라,
사용자의 병원 방문 결정을 돕는 1차 증상 안내 및 예약 요청 연결 서비스를 목적으로 한다.

2. 타겟 사용자

50대~80대 중장년/고령 사용자

병원 검색과 예약이 익숙하지 않은 사용자

음성으로 쉽게 증상을 설명하고 안내받고 싶은 사용자

지역 병원을 빠르게 연결받고 싶은 사용자

3. 핵심 사용자 시나리오
시나리오 1. 증상 입력

사용자가 서비스에 접속한 뒤 음성으로 증상을 말한다.
예:

“내가 오늘 무릎이 좀 쿡쿡 쑤셔.”

“며칠 전부터 허리가 너무 아파.”

“기침이 나고 목이 칼칼해.”

시나리오 2. 문진 내용 정리 및 안내

시스템은 사용자 발화를 STT로 텍스트화하고,
LLM을 이용해 이를 표준 문진 형식으로 구조화한다.

예:

증상 부위: 무릎

증상 표현: 쑤심

지속 여부: 오늘 발생

권장 진료과: 정형외과

이후 사용자에게 TTS로 안내한다.
예:

“무릎 통증 관련 진료가 필요할 수 있습니다.”

“가까운 정형외과를 안내해드릴까요?”

시나리오 3. 병원 연결

사용자가 병원 안내를 원하면,
시스템은 사용자 위치 또는 설정 지역 기준으로 가까운 병원을 추천한다.
Whisper
예:

“가장 가까운 정형외과는 ○○병원입니다. 예약 요청을 남길까요?”

시나리오 4. 예약 요청 접수

사용자가 예약 요청에 동의하면,
시스템은 병원에 예약 요청 정보를 전달한다.
병원은 해당 요청을 확인한 후 사용자에게 전화로 연락한다.

예:

“예약 요청이 접수되었습니다. 병원 확인 후 연락드릴 예정입니다.”

4. 전체 시스템 구성 개요

4-1. 음성 입력 및 STT (한 문장 단위 발화)

- STT 모델: Whisper large-v3 (자체 호스팅)
- 사용자는 리모컨/마이크 버튼을 눌러 한 문장 정도를 말한다.
- 버튼을 떼면 해당 구간이 하나의 음성 조각으로 서버에 전송된다.
- 서버에서는 이 짧은 음성(한 발화)을 Whisper STT 서버에서 텍스트로 변환한다.
- 결과: 한국어 텍스트(구어체)
- 예: "내가 오늘 무릎이 좀 쿡쿡 쑤셔."

4-2. Groq (llama-3.3-70b)를 이용한 문진 구조화 및 안내 생성

- LLM: Groq API (llama-3.3-70b-versatile)
- 입력: STT 텍스트 + 시스템 프롬프트(역할/출력 형식 정의)
- 출력(JSON 예):

{
  "structured_interview": {
    "chief_complaint": "무릎 통증",
    "body_part": "무릎",
    "symptom_description": "쿡쿡 쑤시는 통증",
    "onset": "오늘",
    "duration": "하루 미만",
    "severity": "중간",
    "other_symptoms": []
  },
  "recommended_department": "정형외과",
  "assistant_message": "무릎 통증으로 정형외과 진료가 필요할 수 있습니다."
}

4-3. 카카오맵 API를 이용한 병원 검색

- API: 카카오 로컬 키워드 검색 API
- 입력: recommended_department + 사용자 위도/경도
- 출력: 근처 병원명, 주소, 전화번호
- 실제 서비스: 셋탑박스에서 위치 자동 전달
- 테스트: API 요청 시 latitude, longitude 파라미터로 직접 전달

4-4. TTS 안내

- TTS: gTTS (Google Text-to-Speech)
- `assistant_message` + 병원 정보를 gTTS로 음성 합성
- 생성된 MP3를 Base64 인코딩하여 API 응답에 포함

5. 트러블슈팅(개발 중 자주 발생)

5-1. `uvicorn` 명령이 인식되지 않음

- 증상: `uvicorn : 'uvicorn' 용어가 ... 인식되지 않습니다.`
- 원인: 가상환경 미활성화 또는 `uvicorn` 미설치
- 해결:
  - 가상환경 활성화 후 설치
    - `pip install -r requirements.txt`
  - 실행은 아래 중 하나 사용
    - `uvicorn app:app --reload`
    - `python -m uvicorn app:app --reload`

5-2. 파일 업로드에서 `python-multipart` 필요 에러

- 증상: `Form data requires "python-multipart" to be installed.`
- 원인: FastAPI 파일 업로드(`UploadFile`)는 `python-multipart`가 필요
- 해결:
  - `pip install python-multipart`
  - `requirements.txt`에 `python-multipart` 추가

5-3. Whisper import/동작 오류 (`pip install whisper` 패키지 충돌)

- 증상: `import whisper` 단계에서 예외가 나거나, Whisper가 정상 동작하지 않음
- 원인: `pip install whisper`로 설치되는 다른 패키지와 이름이 충돌하는 경우가 있음
- 해결:
  - 잘못 설치된 패키지 제거 후 올바른 패키지 설치
    - `pip uninstall -y whisper`
    - `pip install -U openai-whisper`
  - `requirements.txt`에는 `whisper` 대신 `openai-whisper` 사용

5-4. mp3 업로드 시 Whisper에서 `WinError 2` 발생 (ffmpeg 없음)

- 증상: `FileNotFoundError: [WinError 2] 지정된 파일을 찾을 수 없습니다`
- 원인: Whisper가 mp3/wav 디코딩을 위해 내부적으로 `ffmpeg`를 호출하는데, Windows에서 `ffmpeg` 실행 파일을 찾지 못함
- 해결:
  - ffmpeg 설치 후 PATH 등록 (설치 후 새 터미널에서 확인)
    - `ffmpeg -version`
  - 예: `winget install Gyan.FFmpeg`