# LGHelloDoctor 개발 계획서 (TDD 기반)

## [완료] 기능 1: 구어체 매핑 벡터 DB 연동
- **목적**: 하드코딩된 프롬프트를 제거하고 ChromaDB를 통한 유사도 검색 구현
- **Red (실패)**: `SymptomVectorStore` 클래스가 없어 `ImportError` 발생 확인 (완료)
- **Green (성공)**: 임계값(Threshold)을 조정하여 `3 PASSED` 달성 (완료)
- **Refactor (리팩토링)**: `llm_gpt.py`에 DB 조회 로직 주입 (진행 중)

---

## [예정] 기능 2: 중앙 API 서버 (FastAPI) 통합
- **목적**: 모든 백엔드 모듈을 하나로 묶어 프론트엔드와 통신하는 엔드포인트 구축
- **수정 파일**: `backend/app.py`, `tests/test_api.py`

### 🔴 Red 단계 (의도적 실패 시나리오)
아래 테스트 코드를 `tests/test_api.py`에 작성하고 실행하여 `404 Not Found` 또는 `ImportError`를 확인한다.

**테스트 항목:**
1. `/api/analyze` 경로로 POST 요청을 보냈을 때 서버가 응답하는가?
2. 요청 바디에 `transcript`, `lat`, `lng`가 포함되어 있는가?
3. 결과값에 병원 목록(`hospitals`)이 포함되어 있는가?

```python
# tests/test_api.py (Red Stage Code)
import pytest
from httpx import AsyncClient

# 아직 backend/app.py가 없으므로 에러 발생이 정상입니다.
try:
    from backend.app import app
except ImportError:
    app = None

@pytest.mark.anyio
async def test_analyze_endpoint_red():
    if app is None:
        pytest.fail("FastAPI 서버 파일(backend/app.py)이 존재하지 않습니다.")
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/api/analyze", json={
            "transcript": "허리가 너무 쑤셔요",
            "lat": 37.4947,
            "lng": 126.7118
        })
    
    assert response.status_code == 200 # 서버가 없으므로 여기서 실패함