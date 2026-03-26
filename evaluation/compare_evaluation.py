"""
Baseline vs NER+VectorDB 성능 비교 평가
"""
import csv
import itertools
import time
import os
import sys

# 경로 설정
_root = os.path.join(os.path.dirname(__file__), '..')
_backend = os.path.join(_root, 'backend')
_dl = os.path.join(_root, 'deeplearning')
sys.path.insert(0, _root)
sys.path.insert(0, _backend)
sys.path.insert(0, _dl)

from tenacity import retry, wait_exponential, stop_after_attempt
from backend.llm_baseline import generate_baseline_triage
from backend.llm_gpt import generate_triage

# ---------------------------------------------------------
# 데이터셋 생성 (run_full_evaluation.py와 동일)
# ---------------------------------------------------------
def generate_dataset():
    dataset = []

    t1_contexts = ["어제부터", "갑자기", "가만히 있어도"]
    t1_parts_mapping = {
        "속이 쓰리고 소화가 안 돼서": "내과",
        "코 수술한 곳이 부어서": "성형외과",
        "무릎이 쑤시고 아파서": "정형외과",
        "발목 삔 데 침을 맞고 싶어서": "한의원",
        "생리통이 너무 심해서": "산부인과",
        "눈이 침침하고 뻑뻑해서": "안과",
        "목이 붓고 따가워서": "이비인후과",
        "피부에 두드러기가 나서": "피부과",
        "소변볼 때 찌릿찌릿해서": "비뇨의학과",
        "요즘 잠도 안 오고 우울해서": "정신건강의학과",
        "어금니가 시려서": "치과"
    }
    t1_symptoms = ["너무 힘들어.", "미치겠어.", "병원 가야 할까?"]
    for ctx, (part, dept), sym in itertools.product(t1_contexts, t1_parts_mapping.items(), t1_symptoms):
        dataset.append(["유형 1", f"{ctx} {part} {sym}", dept])

    t2_combinations = list(itertools.product(
        ["요즘 계속", "며칠 전부터"],
        [("머리도", "배도"), ("가슴도", "팔도"), ("다리도", "허리도"), ("눈도", "속도")],
        zip(["어지럽고", "답답하고", "저리고", "뻑뻑하고"], ["아픈데 어떡해.", "쑤시네.", "울렁거려.", "식은땀이 나."])
    ))[:40]
    for ctx, parts, syms in t2_combinations:
        dataset.append(["유형 2", f"{ctx} {parts[0]} {syms[0]} {parts[1]} {syms[1]}", "일관성 확인"])

    t3_noises = [
        "오늘 날씨가 참 춥네. 밥은 먹었어?", "보일러 고장 난 것 같아.", "손주가 대학 입학했어.",
        "핸드폰 화면이 안 켜져.", "내일 노인정에 몇 시에 가지?", "리모컨 건전지 갈았는데 TV 안 나와.",
        "요새 채소값이 너무 올랐어.", "비밀번호 까먹었어.", "친구 여행 간다는데 나도 갈래.",
        "뉴스 보니까 세상 흉흉해.", "버스가 왜 안 오지.", "명절에 자식들 오려나.", "연속극 봤어?",
        "세탁기 소리가 나네.", "옆집 강아지 짖어서 못 잤어.", "연금 언제 들어오지?", "생선 사 와야지.",
        "안경 어디 뒀더라.", "동네 미용실 원장 바뀌었어.", "김치찌개 어떻게 끓여?", "올해 눈 많이 오네.",
        "구청에서 쓰레기봉투 준대.", "택배 아직 안 왔어.", "보청기 약 떨어졌어.", "화투 치다 돈 잃었어.",
        "젊은 사람들 생각 모르겠어.", "손녀가 용돈 줬어.", "가스레인지 불 안 켜져.", "내일 비 온대 우산 챙겨.", "친구한테 전화나 할까."
    ]
    for noise in t3_noises:
        dataset.append(["유형 3", noise, "알 수 없음"])

    return dataset

# ---------------------------------------------------------
# 추론 (429 에러 방어)
# ---------------------------------------------------------
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5), reraise=True)
def safe_baseline(transcript):
    return generate_baseline_triage(transcript)

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5), reraise=True)
def safe_ner(transcript):
    return generate_triage(transcript)

# ---------------------------------------------------------
# 평가 지표 계산
# ---------------------------------------------------------
def calc_metrics(results):
    t1_tot, t1_cor = 0, 0
    t2_tot, t2_bias = 0, 0
    t3_tot, t3_def = 0, 0
    for r in results:
        predicted = str(r["predicted"])
        if r["type"] == "유형 1":
            t1_tot += 1
            if r["expected"] in predicted:
                t1_cor += 1
        elif r["type"] == "유형 2":
            t2_tot += 1
            if "내과" in predicted:
                t2_bias += 1
        elif r["type"] == "유형 3":
            t3_tot += 1
            if "알 수 없음" in predicted:
                t3_def += 1
    return {
        "t1_acc": (t1_cor / t1_tot * 100) if t1_tot else 0,
        "t1_str": f"{t1_cor}/{t1_tot}",
        "t2_bias": (t2_bias / t2_tot * 100) if t2_tot else 0,
        "t2_str": f"{t2_bias}/{t2_tot}",
        "t3_rej": (t3_def / t3_tot * 100) if t3_tot else 0,
        "t3_str": f"{t3_def}/{t3_tot}",
    }

# ---------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    dataset = generate_dataset()
    total = len(dataset)
    print(f"\n총 {total}개 데이터로 비교 평가 시작 (API 호출: 약 {total * 3}회)\n")

    baseline_results = []
    ner_results = []

    output_file = os.path.join(os.path.dirname(__file__), "compare_results.csv")
    with open(output_file, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["유형", "입력", "정답", "Baseline 예측", "NER+VectorDB 예측"])

        for i, (test_type, transcript, expected) in enumerate(dataset, 1):
            print(f"[{i}/{total}] {transcript[:40]}...")

            # Baseline
            try:
                b = safe_baseline(transcript)
                b_dept = b.get("recommended_department", "필드 누락")
            except Exception as e:
                b_dept = "Error"
            time.sleep(0.5)

            # NER + VectorDB
            try:
                n = safe_ner(transcript)
                n_dept = n.get("recommended_department", "필드 누락")
            except Exception as e:
                n_dept = "Error"
            time.sleep(0.5)

            baseline_results.append({"type": test_type, "expected": expected, "predicted": b_dept})
            ner_results.append({"type": test_type, "expected": expected, "predicted": n_dept})
            writer.writerow([test_type, transcript, expected, b_dept, n_dept])

    b_m = calc_metrics(baseline_results)
    n_m = calc_metrics(ner_results)

    print("\n" + "="*60)
    print("  📊 Baseline vs NER+VectorDB 비교 결과")
    print("="*60)
    print(f"{'지표':<30} {'Baseline':>12} {'NER+VectorDB':>14} {'변화':>8}")
    print("-"*60)

    acc_diff = n_m['t1_acc'] - b_m['t1_acc']
    bias_diff = n_m['t2_bias'] - b_m['t2_bias']
    rej_diff = n_m['t3_rej'] - b_m['t3_rej']

    print(f"{'[유형1] 명확한 증상 정답률':<28} {b_m['t1_acc']:>10.1f}%  {n_m['t1_acc']:>12.1f}%  {acc_diff:>+7.1f}%")
    print(f"{'[유형2] 모호 증상 내과 편향률':<28} {b_m['t2_bias']:>10.1f}%  {n_m['t2_bias']:>12.1f}%  {bias_diff:>+7.1f}%")
    print(f"{'[유형3] 일상대화 기각률':<28} {b_m['t3_rej']:>10.1f}%  {n_m['t3_rej']:>12.1f}%  {rej_diff:>+7.1f}%")
    print("="*60)
    print(f"\n📁 상세 결과: evaluation/compare_results.csv\n")
