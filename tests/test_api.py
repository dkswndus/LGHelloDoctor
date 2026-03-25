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
    
    assert response.status_code == 200