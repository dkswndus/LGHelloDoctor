import os
import re
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()
KAKAO_KEY = os.getenv("KAKAO_REST_API_KEY")
NMC_KEY = os.getenv("HIRA_SERVICE_KEY") 

def normalize_hospital_name(name):
    name = name.replace(" ", "")
    name = re.sub(r'(의원|병원|종합병원|의료원)$', '', name)
    return name

def format_time(time_str):
    """4자리 숫자(0900)를 시간 형식(09:00)으로 변환합니다."""
    if not time_str or len(time_str) != 4:
        return "휴진"
    return f"{time_str[:2]}:{time_str[2:]}"

def get_nmc_operating_hours(hospital_name):
    base_url = "http://apis.data.go.kr/B552657/HsptlAsembySearchService/getHsptlMdcncListInfoInqire"
    
    try:
        request_url = f"{base_url}?ServiceKey={NMC_KEY}&QN={hospital_name}&pageNo=1&numOfRows=10"
        response = requests.get(request_url, timeout=5)
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            items = root.findall('.//item')
            search_keyword = normalize_hospital_name(hospital_name)
            
            for item in items:
                api_hosp_name = item.findtext('dutyName')
                if not api_hosp_name:
                    continue
                
                normalized_api_name = normalize_hospital_name(api_hosp_name)
                
                if search_keyword in normalized_api_name or normalized_api_name in search_keyword:
                    hours = {}
                    days_map = {
                        '1': '월요일', '2': '화요일', '3': '수요일',
                        '4': '목요일', '5': '금요일', '6': '토요일',
                        '7': '일요일', '8': '공휴일'
                    }
                    
                    # 1(월요일)부터 8(공휴일)까지 반복하며 데이터 추출
                    for i in range(1, 9):
                        start = item.findtext(f'dutyTime{i}s')
                        close = item.findtext(f'dutyTime{i}c')
                        
                        if start and close:
                            hours[days_map[str(i)]] = f"{format_time(start)} ~ {format_time(close)}"
                        else:
                            hours[days_map[str(i)]] = "휴진"
                            
                    # 점심시간 및 기타 특이사항 추출
                    duty_inf = item.findtext('dutyInf')
                    hours['특이사항'] = duty_inf if duty_inf else "정보 없음"
                    
                    return {
                        "status": "성공",
                        "schedule": hours
                    }
            
            return {"status": "이름이 일치하는 병원을 공공데이터에서 찾을 수 없음"}
            
        return {"status": f"API 호출 실패 (상태 코드: {response.status_code})"}
        
    except Exception as e:
        return {"status": f"API 연동 에러: {str(e)}"}

def search_nearby_hospitals(department, lat, lng, radius=2000):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    params = {
        "query": department,
        "y": lat,
        "x": lng,
        "radius": radius
    }
    
    response = requests.get(url, headers=headers, params=params)
    hospitals = []
    
    if response.status_code == 200:
        documents = response.json().get('documents', [])
        
        for doc in documents[:3]:
            hosp_name = doc['place_name']
            
            hospital_info = {
                "name": hosp_name,
                "address": doc['road_address_name'] or doc['address_name'],
                "phone": doc['phone'],
                "distance": f"{doc['distance']}m",
                "operating_hours": get_nmc_operating_hours(hosp_name)
            }
            hospitals.append(hospital_info)
            
    return hospitals

if __name__ == "__main__":
    import json
    results = search_nearby_hospitals("정형외과", "37.4947", "126.7118")
    print(json.dumps(results, ensure_ascii=False, indent=2))





def format_hospital_message(hospitals):
    """
    검색된 병원 리스트를 바탕으로 안내 문구를 생성합니다. (app.py 호환용)
    """
    if not hospitals:
        return " 주변에 해당 진료과를 운영하는 병원을 찾지 못했습니다."
    
    top_hosp = hospitals[0]
    msg = f" 가장 가까운 곳은 {top_hosp['distance']} 거리에 있는 {top_hosp['name']}입니다."
    
    # 영업시간 정보 추출 로직
    if top_hosp.get('operating_hours', {}).get('status') == '성공':
        # 오늘 요일(월요일 등)에 맞는 시간을 가져오는 로직 (임시로 월요일 기준)
        monday_time = top_hosp['operating_hours']['schedule'].get('월요일', '정보 없음')
        msg += f" 오늘 진료 시간은 {monday_time}입니다."
        
    return msg