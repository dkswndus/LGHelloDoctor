import os
import requests
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

KAKAO_KEY = os.getenv("KAKAO_REST_API_KEY")
HIRA_KEY = os.getenv("HIRA_SERVICE_KEY")

def test_kakao_api():
    print("--- 카카오맵 API 테스트 ---")
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    params = {
        "query": "정형외과",
        "y": "37.4947", # 테스트용 위도
        "x": "126.7118", # 테스트용 경도
        "radius": 2000
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        print("카카오 API 연결 성공!")
        print(response.json()['documents'][0]) # 첫 번째 결과만 출력
    else:
        print(f"카카오 API 에러: {response.status_code}, {response.text}")

def test_hira_api():
    print("\n--- 심평원 API 테스트 ---")
    # 주의: 신청하신 공공데이터포털 API 엔드포인트 URL로 변경해야 합니다.
    # 아래는 예시 URL입니다.
    url = "https://apis.data.go.kr/B552657/HsptlAsembySearchService"
    
    params = {
        "ServiceKey": HIRA_KEY,
        "pageNo": "1",
        "numOfRows": "1",
        "yadmNm": "서울대학교병원" # 테스트용 병원명
    }
    
    # 공공데이터포털은 requests가 url 인코딩을 이중으로 하는 것을 막기 위해 파라미터를 문자열로 조합하는 것이 안전합니다.
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        print("심평원 API 연결 성공!")
        print(response.text[:1000]) # 응답 결과의 앞부분만 출력 (XML인지 JSON인지, 필드명 확인용)
    else:
        print(f"심평원 API 에러: {response.status_code}, {response.text}")

if __name__ == "__main__":
    if not KAKAO_KEY or not HIRA_KEY:
        print("오류: .env 파일에 KAKAO_REST_API_KEY와 HIRA_SERVICE_KEY를 모두 입력하세요.")
    else:
        test_kakao_api()
        test_hira_api()