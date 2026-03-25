import csv
import itertools
import time
import os
from backend.llm_baseline import generate_baseline_triage
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ---------------------------------------------------------
# 1단계: 11개 진료과 기반 데이터 증강 (약 169개)
# ---------------------------------------------------------
def generate_dataset(filename="eval_dataset.csv"):
    dataset = []
    
    # 유형 1: 11개 진료과 명확한 증상 (3 x 11 x 3 = 99개)
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

    # 유형 2: 모호하고 복합적인 증상 (40개)
    t2_combinations = list(itertools.product(
        ["요즘 계속", "며칠 전부터"],
        [("머리도", "배도"), ("가슴도", "팔도"), ("다리도", "허리도"), ("눈도", "속도")],
        zip(["어지럽고", "답답하고", "저리고", "뻑뻑하고"], ["아픈데 어떡해.", "쑤시네.", "울렁거려.", "식은땀이 나."])
    ))[:40]
    for ctx, parts, syms in t2_combinations:
        dataset.append(["유형 2", f"{ctx} {parts[0]} {syms[0]} {parts[1]} {syms[1]}", "일관성 확인"])

    # 유형 3: 일상 대화 (30개)
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

    with open(filename, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Test_Type", "Input_Transcript", "Expected_Result"])
        writer.writerows(dataset)
    return dataset

# ---------------------------------------------------------
# 2단계: 429 에러 방어 로직 및 LLM 추론
# ---------------------------------------------------------
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5), reraise=True)
def safe_infer(transcript):
    return generate_baseline_triage(transcript)

def run_inference(dataset, output_filename="eval_results.csv"):
    results = []
    print(f"\n🚀 총 {len(dataset)}개 데이터 LLM 추론 시작 (Rate Limit 방어 작동 중)...")
    
    with open(output_filename, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Test_Type", "Input_Transcript", "Expected_Result", "LLM_Department", "LLM_Message"])

        for index, row in enumerate(dataset, start=1):
            test_type, transcript, expected = row[0], row[1], row[2]
            print(f"[{index}/{len(dataset)}] {transcript}")
            try:
                result = safe_infer(transcript)
                llm_dept = result.get("recommended_department", "필드 누락")
                llm_msg = result.get("assistant_message", "필드 누락")
            except Exception as e:
                llm_dept = "Execution Error"
                llm_msg = str(e)
            
            writer.writerow([test_type, transcript, expected, llm_dept, llm_msg])
            results.append({"type": test_type, "expected": expected, "predicted": llm_dept})
            time.sleep(1)
            
    return results

# ---------------------------------------------------------
# 3단계: 성능 지표 평가 및 터미널 출력
# ---------------------------------------------------------
def evaluate_metrics(results):
    t1_tot, t1_cor = 0, 0
    t2_tot, t2_bias = 0, 0
    t3_tot, t3_def = 0, 0

    for r in results:
        if r["type"] == '유형 1':
            t1_tot += 1
            if r["predicted"] in r["expected"]: t1_cor += 1
        elif r["type"] == '유형 2':
            t2_tot += 1
            if r["predicted"] == '내과': t2_bias += 1 # 가장 만만한 내과로 쏠리는지 검사
        elif r["type"] == '유형 3':
            t3_tot += 1
            if r["predicted"] == '알 수 없음': t3_def += 1

    print("\n" + "="*50)
    print(" 📊 11개 진료과 순수 LLM(Baseline) 성능 평가 결과")
    print("="*50)
    print(f"✅ [유형 1] 명확한 증상 정답률 : {(t1_cor/t1_tot*100) if t1_tot else 0:.1f}% ({t1_cor}/{t1_tot})")
    print(f"🚨 [유형 2] 모호한 증상 편향률(내과 쏠림) : {(t2_bias/t2_tot*100) if t2_tot else 0:.1f}% ({t2_bias}/{t2_tot})")
    print(f"🛡️ [유형 3] 일상 대화 기각률 : {(t3_def/t3_tot*100) if t3_tot else 0:.1f}% ({t3_def}/{t3_tot})")
    print("="*50 + "\n")

if __name__ == "__main__":
    generated_data = generate_dataset()
    inference_results = run_inference(generated_data)
    evaluate_metrics(inference_results)