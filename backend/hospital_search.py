import os
import re
import aiohttp
import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
KAKAO_KEY = os.getenv("KAKAO_REST_API_KEY")
NMC_KEY = os.getenv("DATA_GO_KR_API_KEY") or os.getenv("HIRA_SERVICE_KEY")

def normalize_hospital_name(name):
    name = name.replace(" ", "")
    name = re.sub(r'(의원|병원|종합병원|의료원)$', '', name)
    return name

def format_time(time_str):
    if not time_str or len(time_str) != 4:
        return "휴진"
    return f"{time_str[:2]}:{time_str[2:]}"

async def fetch_nmc_operating_hours(session, hospital_name):
    base_url = "http://apis.data.go.kr/B552657/HsptlAsembySearchService/getHsptlMdcncListInfoInqire"
    params = {
        "serviceKey": NMC_KEY,
        "QN": hospital_name,
        "pageNo": 1,
        "numOfRows": 10
    }
    
    try:
        async with session.get(base_url, params=params, timeout=5) as response:
            if response.status == 200:
                content = await response.text()
                root = ET.fromstring(content)
                items = root.findall('.//item')
                search_keyword = normalize_hospital_name(hospital_name)
                
                for item in items:
                    api_hosp_name = item.findtext('dutyName')
                    if not api_hosp_name: continue
                    
                    if search_keyword in normalize_hospital_name(api_hosp_name):
                        hours = {}
                        days_map = {'1':'월요일','2':'화요일','3':'수요일','4':'목요일','5':'금요일','6':'토요일','7':'일요일','8':'공휴일'}
                        
                        for i in range(1, 9):
                            start = item.findtext(f'dutyTime{i}s')
                            close = item.findtext(f'dutyTime{i}c')
                            hours[days_map[str(i)]] = f"{format_time(start)} ~ {format_time(close)}" if start and close else "휴진"
                            
                        duty_inf = item.findtext('dutyInf')
                        hours['특이사항'] = duty_inf if duty_inf else "정보 없음"
                        return {"status": "성공", "schedule": hours}
                return {"status": "데이터 없음"}
            return {"status": f"API 오류({response.status})"}
    except Exception:
        return {"status": "연동 에러"}

async def search_nearby_hospitals(department, lat, lng, radius=2000):
    kakao_url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    params = {"query": department, "y": lat, "x": lng, "radius": radius}
    
    import requests
    res = requests.get(kakao_url, headers=headers, params=params)
    if res.status_code != 200:
        return []

    documents = res.json().get('documents', [])[:3]
    hospitals = []

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_nmc_operating_hours(session, doc['place_name']) for doc in documents]
        operating_hours_results = await asyncio.gather(*tasks)

        for i, doc in enumerate(documents):
            hospitals.append({
                "name": doc['place_name'],
                "address": doc['road_address_name'] or doc['address_name'],
                "phone": doc['phone'],
                "distance": f"{doc['distance']}m",
                "operating_hours": operating_hours_results[i]
            })
    return hospitals

def format_hospital_message(hospitals):
    """검색된 병원 정보를 바탕으로 실시간 음성 안내 문구를 생성합니다."""
    if not hospitals:
        return " 주변에 해당 진료과를 운영하는 병원을 찾지 못했습니다."
    
    top_hosp = hospitals[0]
    msg = f" 가장 가까운 곳은 {top_hosp['distance']} 거리에 있는 {top_hosp['name']}입니다."
    
    # 오늘 요일 계산 (0=월, 1=화, ... 6=일)
    days_list = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    today_name = days_list[datetime.now().weekday()]
    
    if top_hosp.get('operating_hours', {}).get('status') == '성공':
        today_time = top_hosp['operating_hours']['schedule'].get(today_name, '정보 없음')
        if today_time != "휴진":
            msg += f" 오늘 {today_name} 진료 시간은 {today_time}까지입니다."
        else:
            msg += f" 아쉽게도 오늘 {today_name}은 휴진입니다."
        
    return msg