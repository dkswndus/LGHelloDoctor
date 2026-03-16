from __future__ import annotations

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def transcribe_audio_file(path: str, language: str = "ko") -> str:
    """
    Groq의 Whisper API로 한국어 STT를 수행한다. (로컬 대비 10배 이상 빠름)
    """
    client = _get_client()
    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(os.path.basename(path), f),
            model="whisper-large-v3-turbo",
            language=language,
            response_format="text",
        )
    return result.strip() if isinstance(result, str) else result.text.strip()

