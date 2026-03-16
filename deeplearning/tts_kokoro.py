import io


def synthesize_speech(text: str) -> bytes:
    """
    gTTS로 텍스트를 한국어 음성(MP3)으로 합성한다.
    API 키 없이 무료로 사용 가능.

    Returns:
        MP3 형식의 오디오 바이너리
    """
    from gtts import gTTS

    if not text or not text.strip():
        raise ValueError("텍스트는 비어있을 수 없습니다.")

    tts = gTTS(text=text, lang="ko", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()
