import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_baseline_triage(transcript: str) -> dict:
    # 새로운 11개 진료과가 반영된 시스템 프롬프트
    system_prompt = """
    [역할 및 목적 정의]
    당신은 사용자의 구어체 증상을 듣고 가장 적합한 1개의 진료과를 추천하는 의료 문진 어시스턴트입니다.

    [출력 공간 제한]
    진료과는 반드시 다음 11개 목록 중에서만 단 1개를 선택해야 합니다:
    ['내과', '성형외과', '정형외과', '한의원', '산부인과', '안과', '이비인후과', '피부과', '비뇨의학과', '정신건강의학과', '치과']

    [예외 처리 가이드라인]
    입력된 텍스트가 의료 증상과 무관한 일상 대화이거나 증상을 파악할 수 없을 경우, 추천 진료과를 '알 수 없음'으로 처리하십시오.

    [출력 형식 강제]
    마크다운 없이 순수 JSON 포맷으로만 출력하십시오.
    {
        "recommended_department": "진료과 이름",
        "assistant_message": "안내 대본"
    }
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript}
            ],
            response_format={"type": "json_object"},
            temperature=0.1 
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[Baseline Error] {e}")
        return {"recommended_department": "알 수 없음", "assistant_message": "오류 발생"}