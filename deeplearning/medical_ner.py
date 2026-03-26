import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """당신은 의료 NER(개체명 인식) 전문가입니다.
입력된 구어체 텍스트에서 의료적으로 유의미한 정보만 추출하세요.

추출 대상:
- 신체 부위 (배, 머리, 가슴, 허리 등)
- 증상 양상 (쥐어짜듯, 찌릿찌릿, 뻐근한, 욱신욱신 등)
- 증상/상태 (아파, 메스꺼움, 어지러움, 구역질 등)
- 지속 시간 (사흘째, 아침부터, 일주일 전부터 등)
- 강도 (심하게, 약간, 조금 등)

제거 대상:
- 시간/날짜 표현 (오늘, 어제, 요즘)
- 일상 행동 (밥 먹었는데, 운동하다가, 걷다가)
- 감탄사 및 접속사 (그냥, 근데, 아, 음)
- 의료와 무관한 명사/동사

규칙:
- 추출한 키워드만 공백으로 구분하여 한 줄로 출력
- 설명, 부연, 문장 부호 없이 키워드만 출력
- 추출할 의료 정보가 없으면 빈 문자열 반환"""


def extract_medical_entities(transcript: str) -> str:
    """
    구어체 텍스트에서 의료 관련 엔티티만 추출합니다.

    예시:
        입력: "오늘 밥을 먹었는데 배가 쥐어짜듯 아프고 속이 메스꺼워요"
        출력: "배 쥐어짜듯 아파 메스꺼움"
    """
    if not transcript or not transcript.strip():
        return ""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        temperature=0.0,  # 결정적 출력을 위해 0으로 고정
        max_tokens=100,   # 키워드만 출력하므로 짧게 제한
    )
    return response.choices[0].message.content.strip()
