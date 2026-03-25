import json
import os
from groq import Groq
from dotenv import load_dotenv
from backend.database.vector_store import SymptomVectorStore # 벡터 DB 불러오기

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 서버 시작 시 벡터 DB를 한 번만 초기화합니다.
v_store = SymptomVectorStore()
v_store.initialize_mapping_data()

def generate_triage(transcript: str):
    # 1단계: 벡터 DB에서 유사한 증상을 먼저 찾습니다.
    db_result = v_store.search_similar_symptom(transcript)
    
    # DB에서 찾은 정보를 프롬프트에 넣을 텍스트로 준비합니다.
    mapping_context = "검색된 유사 증상 정보 없음"
    if db_result and "medical_term" in db_result:
        mapping_context = f"유사 증상: {db_result['medical_term']}, 권장 진료과: {db_result['recommended_department']}"

    # 2단계: LLM 프롬프트 구성 (매핑 테이블을 직접 넣지 않고 DB 결과를 참조하게 함)
    system_prompt = f"""
    당신은 고령층 대상 음성 문진 어시스턴트입니다. 
    반드시 JSON 형식으로만 응답하며, 아래 제공된 '참조 정보'를 최우선으로 고려하여 진료과를 추천하세요.
    
    [참조 정보]
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