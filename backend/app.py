import base64
import os
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# deeplearning 모듈 경로 추가 (로컬: ../deeplearning, Docker: /app/deeplearning)
_dl_path = os.path.join(os.path.dirname(__file__), '..', 'deeplearning')
_backend_path = os.path.dirname(__file__)
sys.path.insert(0, _dl_path)
sys.path.insert(0, _backend_path)

from llm_gpt import generate_triage
from stt_whisper import transcribe_audio_file
from tts_kokoro import synthesize_speech
from hospital_search import format_hospital_message, search_nearby_hospitals

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
    2) Groq(llama-3.3-70b)로 문진 구조화 + 안내 문장 생성
    3) 카카오 API로 근처 병원 검색 (위치 제공 시)
    4) gTTS로 안내 문장을 음성으로 합성
    """
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "")[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 1. Whisper STT
        transcript = transcribe_audio_file(tmp_path, language="ko")

        # 2. Groq 문진 구조화
        triage: Dict[str, Any] = generate_triage(transcript)

        # 3. 근처 병원 검색 (위치 정보가 있을 때만)
        hospital_message = ""
        hospitals = []
        if latitude is not None and longitude is not None:
            department = triage.get("recommended_department", "")
            try:
                hospitals = search_nearby_hospitals(department, latitude, longitude)
                hospital_message = format_hospital_message(hospitals)
            except RuntimeError as e:
                triage["hospital_search_error"] = str(e)

        # 4. 최종 안내 문장 = LLM 메시지 + 실제 병원 정보
        assistant_message = triage.get("assistant_message", "")
        if hospital_message:
            assistant_message = f"{assistant_message} {hospital_message}"

        if hospitals:
            triage["hospital"] = hospitals[0]

        return JSONResponse(
            _build_response_payload(
                transcript=transcript,
                triage=triage,
                assistant_message=assistant_message,
            )
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _build_response_payload(
    transcript: str, triage: Dict[str, Any], assistant_message: str
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"transcript": transcript, "triage": triage}
    try:
        audio_bytes = synthesize_speech(assistant_message)
    except Exception as e:
        payload["tts_status"] = "error"
        payload["tts_error"] = str(e)
        return payload

    payload["tts_status"] = "ok"
    payload["tts_audio_base64"] = base64.b64encode(audio_bytes).decode("utf-8")
    payload["assistant_message"] = assistant_message
    return payload


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
