import pytest
# 주의: backend.database.vector_store 모듈과 SymptomVectorStore 클래스는 아직 구현되지 않았습니다.
# TDD 원칙에 따라 여기서 ImportError가 발생하는 것이 첫 번째 'Red' 실패 조건입니다.
from backend.database.vector_store import SymptomVectorStore

@pytest.fixture
def vector_store():
    """
    테스트에 사용할 Vector DB 객체를 초기화하는 픽스처(Fixture)입니다.
    데이터베이스 연결 및 초기 매핑 더미 데이터 삽입이 수행된다고 가정합니다.
    """
    store = SymptomVectorStore(collection_name="test_symptoms")
    store.initialize_mapping_data()
    return store

def test_exact_match_symptom(vector_store):
    """
    핵심 시나리오 1: DB에 등록된 텍스트와 100% 동일한 입력이 주어졌을 때의 정확도 테스트
    입력: "쿡쿡 쑤셔요"
    기대 결과: 의학 용어 '자통', 진료과 '정형외과' 또는 '신경과'
    """
    result = vector_store.search_similar_symptom("쿡쿡 쑤셔요")
    
    assert result is not None
    assert result["medical_term"] == "자통"
    assert "정형외과" in result["recommended_department"] or "신경과" in result["recommended_department"]

def test_semantic_match_symptom(vector_store):
    """
    핵심 시나리오 2: 형태소는 다르지만 의미가 같은 구어체 발화에 대한 의미론적 유사도 매칭 테스트
    입력: "관절 쪽이 바늘로 찌르는 것처럼 아프네" (사전에 없는 변형된 문장)
    기대 결과: 문맥을 파악하여 '자통'으로 매핑되어야 함
    """
    result = vector_store.search_similar_symptom("관절 쪽이 바늘로 찌르는 것처럼 아프네")
    
    assert result is not None
    assert result["medical_term"] == "자통"

def test_irrelevant_input_threshold(vector_store):
    """
    핵심 시나리오 3: 의료와 무관한 입력 시 임계값(Threshold)에 의해 필터링되는지 테스트
    입력: "오늘 날씨가 참 좋네요"
    기대 결과: 유사도가 낮으므로 None을 반환하거나, 신뢰도 부족 에러를 반환해야 함
    """
    result = vector_store.search_similar_symptom("오늘 날씨가 참 좋네요")
    
    # 결과가 없거나, 매핑 실패 상태 코드를 반환해야 합니다.
    assert result is None or result.get("status") == "low_confidence"