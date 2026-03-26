"""
Step 1: LLM으로 (구어체, 의학용어) 쌍 데이터 자동 생성
실행: python deeplearning/generate_training_data.py
출력: deeplearning/training_data.json
"""
import json
import os
import sys
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 11개 진료과별 의학 용어 목록
DEPARTMENT_TERMS = {
    "내과":         ["위염", "복통", "소화불량", "구역질", "설사", "변비", "역류성식도염"],
    "정형외과":     ["관절염", "근육통", "디스크", "염좌", "골절", "퇴행성관절염"],
    "한의원":       ["근육뭉침", "혈액순환장애", "냉증", "피로누적"],
    "산부인과":     ["생리통", "생리불순", "폐경증상", "질염"],
    "안과":         ["안구건조증", "백내장", "시력저하", "결막염"],
    "이비인후과":   ["인후염", "비염", "중이염", "편도염", "이명"],
    "피부과":       ["두드러기", "습진", "건선", "접촉성피부염", "여드름"],
    "비뇨의학과":   ["방광염", "요로감염", "전립선비대", "빈뇨"],
    "정신건강의학과": ["불면증", "우울증", "불안장애", "공황장애"],
    "치과":         ["충치", "잇몸염증", "치주염", "치아시림"],
    "성형외과":     ["수술후붓기", "흉터", "피부처짐"],
}

SYSTEM_PROMPT = """당신은 고령층 환자의 언어를 잘 아는 의료 전문가입니다.
주어진 의학 용어에 대해, 고령층(60~80대)이 병원에서 실제로 사용할 구어체 표현 15가지를 생성하세요.

규칙:
- 반드시 고령층 구어체로 작성 (예: "~해요", "~아요", "~네요", "~겠어요")
- 신체 부위 + 증상 양상을 구체적으로 표현
- 각 표현은 서로 다른 방식으로 표현
- JSON 배열로만 반환: ["표현1", "표현2", ...]"""


def generate_expressions(medical_term: str, department: str) -> list[str]:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"의학 용어: {medical_term} (진료과: {department})"}
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
            max_tokens=500,
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        # JSON 응답에서 리스트 추출
        if isinstance(parsed, list):
            return parsed
        for v in parsed.values():
            if isinstance(v, list):
                return v
        return []
    except Exception as e:
        print(f"  [오류] {medical_term}: {e}")
        return []


if __name__ == "__main__":
    all_pairs = []
    total_terms = sum(len(v) for v in DEPARTMENT_TERMS.values())
    processed = 0

    print(f"총 {total_terms}개 의학 용어에 대해 구어체 표현 생성 시작...\n")

    for department, terms in DEPARTMENT_TERMS.items():
        for term in terms:
            processed += 1
            print(f"[{processed}/{total_terms}] {department} - {term} 생성 중...")
            expressions = generate_expressions(term, department)

            for expr in expressions:
                all_pairs.append({
                    "colloquial": expr,
                    "medical_term": term,
                    "department": department
                })

            print(f"  → {len(expressions)}개 생성 완료")
            time.sleep(1)  # Rate limit 방어

    output_path = os.path.join(os.path.dirname(__file__), "training_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(all_pairs)}개 쌍 생성 완료 → {output_path}")
