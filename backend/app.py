import base64
import os
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# 경로 설정
_dl_path = os.path.join(os.path.dirname(__file__), '..', 'deeplearning')
_backend_path = os.path.dirname(__file__)
sys.path.insert(0, _dl_path)
sys.path.insert(0, _backend_path)

# 모듈 임포트 (중복 제거 및 최적화)
from llm_gpt import generate_triage
from stt_whisper import transcribe_audio_file
from tts_kokoro import synthesize_speech
from hospital_search import format_hospital_message, search_nearby_hospitals

# .env 로드
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

app = FastAPI(title="LGHelloDoctor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-audio")
async def analyze_audio(
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
) -> JSONResponse:
    """
    1) 음성 파일을 Whisper로 STT
    2) Groq(llama-3.3-70b)로 문진 구조화 + 4단계 Cross-Encoder 적용 (generate_triage 내부)
    3) NMC API 비동기 병렬 호출로 병원 영업시간 검색 (6단계)
    4) TTS로 안내 문장 합성 (7단계)
    """
    # 임시 파일 생성
    suffix = os.path.splitext(file.filename or "")[1]
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 1. Whisper STT
        transcript = transcribe_audio_file(tmp_path, language="ko")

        # 2. Groq 문진 구조화 및 리랭킹 (llm_gpt 내부에서 vector_store 호출)
        triage: Dict[str, Any] = generate_triage(transcript)

        # 3. 6단계: 비동기 병원 검색 (NMC 영업시간 연동)
        hospital_message = ""
        hospitals = []
        if latitude is not None and longitude is not None:
            department = triage.get("recommended_department", "내과")
            try:
                # [핵심 수정] 비동기 함수이므로 반드시 await를 붙여야 함
                hospitals = await search_nearby_hospitals(department, latitude, longitude)
                hospital_message = format_hospital_message(hospitals)
            except Exception as e:
                triage["hospital_search_error"] = f"병원 검색 실패: {str(e)}"

        # 4. 최종 안내 문장 조립
        assistant_message = triage.get("assistant_message", "")
        if hospital_message:
            assistant_message = f"{assistant_message} {hospital_message}"

        # 프론트엔드 호환성을 위해 첫 번째 병원 정보를 triage에 삽입
        if hospitals:
            triage["hospital"] = hospitals[0]
            triage["all_hospitals"] = hospitals # 전체 리스트도 전달

        # 5. 응답 생성 (내부에서 TTS 호출)
        response_data = _build_response_payload(
            transcript=transcript,
            triage=triage,
            assistant_message=assistant_message,
        )
        
        return JSONResponse(content=response_data)

    finally:
        # 임시 파일 삭제
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def _build_response_payload(
    transcript: str, triage: Dict[str, Any], assistant_message: str
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"transcript": transcript, "triage": triage}
    try:
        # 7. TTS 음성 합성
        audio_bytes = synthesize_speech(assistant_message)
        payload["tts_status"] = "ok"
        payload["tts_audio_base64"] = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        payload["tts_status"] = "error"
        payload["tts_error"] = str(e)

    payload["assistant_message"] = assistant_message
    return payload

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}