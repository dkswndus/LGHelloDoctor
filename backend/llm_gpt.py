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

def generate_triage(transcript: str):
    # 1단계: NER - 구어체 텍스트에서 의료 키워드만 추출합니다.
    symptom_keywords = extract_medical_entities(transcript)
    search_query = symptom_keywords if symptom_keywords else transcript

    # 2단계: 벡터 DB 코사인 유사도 검색 (Top-5)
    db_result = v_store.search_similar_symptom(search_query)
    status = db_result.get("status", "rejected")

    # 거리 기반 3단계 분기 처리
    if status == "rejected":
        # 0.6 초과: 의료 무관 질문 → LLM 호출 없이 즉시 반환
        return {
            "recommended_department": "알 수 없음",
            "assistant_message": "의료 관련 질문에만 답변 가능합니다. 증상이 있으시면 말씀해 주세요.",
            "structured_interview": {}
        }

    if status == "ambiguous":
        # 0.4~0.6: 모호한 질문 → 재질문 유도
        return {
            "recommended_department": "알 수 없음",
            "assistant_message": "증상이 조금 불분명합니다. 구체적으로 어디가 어떻게 아프신지 다시 말씀해 주시겠어요?",
            "structured_interview": {}
        }

    # 0.0~0.4: 확실한 의료 질문 → Top-5 후보를 컨텍스트로 LLM 호출
    top_candidates = db_result.get("top_candidates", [])
    mapping_context = "\n".join([
        f"{i+1}. 유사 증상: {c['document']} → 의학용어: {c['medical_term']}, 권장 진료과: {c['recommended_department']} (거리: {c['distance']:.3f})"
        for i, c in enumerate(top_candidates)
    ])

    system_prompt = f"""
    당신은 고령층 대상 음성 문진 어시스턴트입니다.
    반드시 JSON 형식으로만 응답하며, 아래 제공된 '참조 정보'를 최우선으로 고려하여 진료과를 추천하세요.

    [참조 정보 - 유사 증상 Top-5]
    {mapping_context}

    응답 스키마는 'structured_interview', 'recommended_department', 'assistant_message'를 포함해야 합니다.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript}
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)