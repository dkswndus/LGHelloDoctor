"""
Step 3: 파인튜닝된 모델로 벡터 DB 데이터 재임베딩
실행: python deeplearning/update_vector_db.py
입력: deeplearning/models/ko-medical-sroberta/
      deeplearning/training_data.json
효과: 서버 재시작 시 파인튜닝 모델이 자동으로 사용됨
"""
import json
import os
import sys

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)

from sentence_transformers import SentenceTransformer
import chromadb

FINETUNED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ko-medical-sroberta")
DATA_PATH = os.path.join(os.path.dirname(__file__), "training_data.json")


def verify_improvement():
    """파인튜닝 전후 유사도 비교"""
    test_pairs = [
        ("배가 쥐어짜듯 아파요", "복통"),
        ("찌릿찌릿해요",         "방사통"),
        ("눈이 뿌옇게 보여요",   "백내장"),
        ("오늘 날씨가 춥네요",   "위염"),   # 의료 무관 → 거리 멀어야 함
    ]

    print("\n[파인튜닝 전후 코사인 유사도 비교]")
    print(f"{'입력':<25} {'의학용어':<12} {'기존':>8} {'파인튜닝':>10} {'개선':>8}")
    print("-" * 70)

    base_model  = SentenceTransformer("jhgan/ko-sroberta-multitask")
    tuned_model = SentenceTransformer(FINETUNED_MODEL_PATH)

    from sentence_transformers import util
    for colloquial, medical in test_pairs:
        e1_base  = base_model.encode(colloquial,  convert_to_tensor=True)
        e2_base  = base_model.encode(medical,     convert_to_tensor=True)
        e1_tuned = tuned_model.encode(colloquial, convert_to_tensor=True)
        e2_tuned = tuned_model.encode(medical,    convert_to_tensor=True)

        sim_base  = float(util.cos_sim(e1_base,  e2_base))
        sim_tuned = float(util.cos_sim(e1_tuned, e2_tuned))
        diff = sim_tuned - sim_base

        print(f"{colloquial:<25} {medical:<12} {sim_base:>8.3f} {sim_tuned:>10.3f} {diff:>+8.3f}")


if __name__ == "__main__":
    if not os.path.exists(FINETUNED_MODEL_PATH):
        print(f"[오류] 파인튜닝 모델 없음: {FINETUNED_MODEL_PATH}")
        print("먼저 실행: python deeplearning/finetune_embeddings.py")
        sys.exit(1)

    # 파인튜닝 전후 성능 비교 출력
    verify_improvement()

    # vector_store.py의 모델 경로를 파인튜닝 모델로 업데이트
    vector_store_path = os.path.join(_root, "backend", "database", "vector_store.py")
    with open(vector_store_path, encoding="utf-8") as f:
        content = f.read()

    old_line = "self.embedder = SentenceTransformer('jhgan/ko-sroberta-multitask')"
    new_line  = f"self.embedder = SentenceTransformer(r'{FINETUNED_MODEL_PATH}')"

    if old_line in content:
        content = content.replace(old_line, new_line)
        with open(vector_store_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\n[완료] vector_store.py 모델 경로 업데이트")
        print(f"  기존: jhgan/ko-sroberta-multitask")
        print(f"  변경: {FINETUNED_MODEL_PATH}")
        print("\n서버를 재시작하면 파인튜닝된 모델이 적용됩니다.")
    elif new_line in content:
        print("\n[확인] 이미 파인튜닝 모델로 설정되어 있습니다.")
    else:
        print("\n[주의] vector_store.py에서 모델 경로를 수동으로 변경해주세요.")
