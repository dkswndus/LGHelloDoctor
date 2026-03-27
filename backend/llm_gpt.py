import json
import os
import sys
from groq import Groq
from dotenv import load_dotenv
from backend.database.vector_store import SymptomVectorStore # 벡터 DB 불러오기

_dl_path = os.path.join(os.path.dirname(__file__), '..', 'deeplearning')
sys.path.insert(0, _dl_path)
from medical_ner import extract_medical_entities

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 서버 시작 시 벡터 DB를 한 번만 초기화합니다.
v_store = SymptomVectorStore()
v_store.initialize_mapping_data()

BODY_PARTS = [
    "머리", "눈", "귀", "코", "입", "목", "어깨", "팔", "팔꿈치", "손목", "손", "손가락",
    "가슴", "등", "허리", "배", "복부", "옆구리", "골반", "엉덩이", "다리", "무릎",
    "발목", "발", "발가락", "허벅지", "종아리", "관절", "뼈", "근육", "피부", "두피"
]

EMERGENCY_KEYWORDS = [
    "숨", "호흡", "의식", "마비", "쓰러", "말이 안", "눈앞", "갑자기", "응급", "119"
]

def _has_body_part(text: str) -> bool:
    return any(part in text for part in BODY_PARTS)

def _has_emergency_keyword(text: str) -> bool:
    return any(kw in text for kw in EMERGENCY_KEYWORDS)

def generate_triage(transcript: str):
    # 1단계: NER - 구어체 텍스트에서 의료 키워드만 추출합니다.
    symptom_keywords = extract_medical_entities(transcript)
    # NER이 설명문/긴 텍스트를 반환하면 원본 사용 (50자 초과 또는 줄바꿈 포함 시)
    if symptom_keywords and (len(symptom_keywords) > 50 or "\n" in symptom_keywords or "->" in symptom_keywords):
        symptom_keywords = ""
    search_query = symptom_keywords if symptom_keywords else transcript

    # 2단계: 벡터 DB 코사인 유사도 검색 (Top-5)
    db_result = v_store.search_similar_symptom(search_query)
    status = db_result.get("status", "rejected")

    # 신체 부위 없어도 응급 키워드 있으면 LLM으로 바로 전달
    if _has_emergency_keyword(transcript):
        pass
    # 신체 부위 없으면 → 판단
    elif not _has_body_part(transcript):
        if status == "clear":
            pass  # vectorDB가 확실히 의료 질문으로 판단 → LLM 호출로 이어짐
        elif status == "rejected":
            return {
                "structured_interview": {},
                "recommended_department": "알 수 없음",
                "assistant_message": "의료 관련 질문에만 답변 가능합니다."
            }
        else:  # ambiguous + 신체 부위 없음
            return {
                "structured_interview": {},
                "recommended_department": "알 수 없음",
                "assistant_message": "구체적으로 어디가 아프신지 말씀해주시겠어요?"
            }

    # clear/ambiguous/rejected(신체부위 있음) → LLM으로 판단 (vectorDB는 참조 정보로 활용)
    top_candidates = db_result.get("top_candidates", [])
    mapping_context = "\n".join([
        f"{i+1}. 유사 증상: {c['document']} → 의학용어: {c['medical_term']}, 권장 진료과: {c['recommended_department']} (거리: {c['distance']:.3f})"
        for i, c in enumerate(top_candidates)
    ]) if top_candidates else "참조 정보 없음"

    system_prompt = f"""
    당신은 고령층 대상 음성 문진 어시스턴트입니다.
    반드시 JSON 형식으로만 응답하세요.
    assistant_message는 반드시 한국어로만 작성하세요. 영어, 일본어, 중국어 등 다른 언어는 절대 사용하지 마세요.

    [참조 정보 - 유사 증상 Top-5]
    {mapping_context}

    규칙:
    1. 응급 상황이면 is_emergency=true, assistant_message="지금 즉시 119에 신고하세요!"
       응급 상황 기준: 심근경색, 뇌졸중, 가슴이 쥐어짜듯 아픔, 갑자기 마비, 의식 잃음,
       숨쉬기 힘듦/숨이 안 쉬어짐/호흡 곤란, 갑자기 말이 안 나옴, 눈앞이 갑자기 안 보임,
       극심한 두통, 온몸에 힘이 갑자기 빠짐, 쓰러질 것 같음
    2. "아파요", "몸이 안좋아요", "불편해요" 처럼 신체부위도 없고 증상도 막연하면: recommended_department="알 수 없음", assistant_message="구체적으로 어디가 어떻게 아프신지 말씀해주시겠어요?"
    3. "잠을 못 자요", "이가 아파요", "소화가 안 돼요" 처럼 증상이 구체적이면: 반드시 진료과를 추천하고 assistant_message로 안내하세요.

    응답 스키마는 'is_emergency', 'structured_interview', 'recommended_department', 'assistant_message'를 포함해야 합니다.
    주의: recommended_department는 반드시 문자열(string)이어야 합니다. 리스트(배열)로 반환하지 마세요. 예: "정형외과, 신경외과"
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript}
        ],
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    result["source"] = "llm"
    return result