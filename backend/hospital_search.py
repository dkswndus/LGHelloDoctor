import os
import requests
from typing import Optional


def search_nearby_hospitals(
    department: str,
    latitude: float,
    longitude: float,
    radius: int = 3000,
    max_results: int = 1,
) -> list[dict]:
    """
    카카오 로컬 키워드 검색 API로 근처 병원을 검색한다.

    Args:
        department: 진료과 (예: "정형외과", "내과")
        latitude: 사용자 위도
        longitude: 사용자 경도
        radius: 검색 반경 (미터, 최대 20000)
        max_results: 반환할 최대 병원 수

    Returns:
        병원 정보 딕셔너리 리스트
        [{"name": ..., "address": ..., "phone_number": ..., "distance_m": ...}]

    Raises:
        RuntimeError: API 키 미설정 또는 요청 실패 시
    """
    api_key = os.getenv("KAKAO_REST_API_KEY")
    if not api_key:
        raise RuntimeError(
            "KAKAO_REST_API_KEY가 설정되지 않았습니다. "
            ".env 파일에 KAKAO_REST_API_KEY=<키> 를 추가하세요. "
            "카카오 개발자 콘솔(developers.kakao.com)에서 REST API 키를 발급받을 수 있습니다."
        )

    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {
        "query": department,
        "x": longitude,   # 카카오 API: x=경도, y=위도
        "y": latitude,
        "radius": radius,
        "size": max_results,
        "sort": "distance",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"카카오 병원 검색 요청 실패: {e}") from e

    documents = response.json().get("documents", [])

    results = []
    for doc in documents:
        distance_m = int(doc.get("distance") or 0)
        results.append(
            {
                "name": doc.get("place_name", ""),
                "address": doc.get("road_address_name") or doc.get("address_name", ""),
                "phone_number": doc.get("phone", ""),
                "distance_m": distance_m,
            }
        )
    return results


def format_hospital_message(hospitals: list[dict]) -> str:
    """
    병원 리스트를 TTS용 안내 문장으로 변환한다.

    Example:
        "집 근처 정형외과 병원은 연세정형외과의원입니다. 주소는 서울시 강남구 ...이고, 전화번호는 02-123-4567입니다."
    """
    if not hospitals:
        return ""

    h = hospitals[0]
    parts = [f"집 근처 병원은 {h['name']}입니다."]
    if h["address"]:
        parts.append(f"주소는 {h['address']}이고,")
    if h["phone_number"]:
        parts.append(f"전화번호는 {h['phone_number']}입니다.")
    else:
        parts.append("전화번호 정보는 없습니다.")

    return " ".join(parts)
