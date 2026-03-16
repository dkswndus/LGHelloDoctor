from typing import Any, Dict


SYSTEM_PROMPT = """
너는 50~80대 한국 사용자 대상 음성 문진을 돕는 AI 어시스턴트이다.
사용자의 자연스러운 구어체 문장을 입력으로 받아, 아래 JSON 스키마에 맞게만 응답한다.

응답 형식(JSON):
{
  "structured_interview": {
    "chief_complaint": string,
    "body_part": string,
    "symptom_description": string,
    "onset": string,
    "duration": string,
    "severity": string,
    "other_symptoms": string[]
  },
  "recommended_department": string,
  "assistant_message": string,
  "followup_questions": string[]
}

- 의료 진단/확정적인 표현은 피하고, "필요할 수 있습니다", "의료진 상담이 필요합니다"와 같이 안내 위주로 표현한다.
- assistant_message는 증상 공감 + 권장 진료과 안내만 담는다. 병원 정보는 포함하지 않는다.
- 반드시 위 JSON 형태로만, 한국어 내용을 채워서 응답한다.
"""


def generate_triage(transcript: str) -> Dict[str, Any]:
    """
    사용자의 발화 텍스트를 받아 문진 구조화와 안내 메시지를 생성한다.
    Groq API (llama-3.3-70b)를 사용한다.
    """
    from groq import Groq
    from dotenv import load_dotenv
    import os
    import json

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment variables.")

    client = Groq(api_key=api_key)

    user_prompt = f"사용자 발화: \"{transcript.strip()}\"\n\n위 발화를 분석하여 지정된 JSON 스키마에 맞는 단일 JSON 객체만 반환하세요."

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or ""
    if not content:
        raise RuntimeError("Groq 응답이 비어 있습니다.")

    return json.loads(content)


