"""
Step 2: ko-sroberta-multitask 위에 Contrastive Learning 파인튜닝
실행: python deeplearning/finetune_embeddings.py
입력: deeplearning/training_data.json
출력: deeplearning/models/ko-medical-sroberta/
"""

import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import math
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation

BASE_MODEL = "jhgan/ko-sroberta-multitask"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "models", "ko-medical-sroberta")
DATA_PATH   = os.path.join(os.path.dirname(__file__), "training_data.json")

BATCH_SIZE = 16
EPOCHS     = 10
WARMUP_RATIO = 0.1


def load_data(path: str):
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)

    # 8:2로 train/eval 분리
    split = int(len(pairs) * 0.8)
    train_pairs = pairs[:split]
    eval_pairs  = pairs[split:]

    train_examples = [
        InputExample(texts=[p["colloquial"], p["medical_term"]])
        for p in train_pairs
    ]
    # evaluation용: (구어체, 의학용어, 유사도=1.0)
    eval_sentences1 = [p["colloquial"]   for p in eval_pairs]
    eval_sentences2 = [p["medical_term"] for p in eval_pairs]
    eval_scores     = [1.0               for p in eval_pairs]

    return train_examples, eval_sentences1, eval_sentences2, eval_scores


if __name__ == "__main__":
    print(f"베이스 모델 로드: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)

    print(f"학습 데이터 로드: {DATA_PATH}")
    train_examples, eval_s1, eval_s2, eval_scores = load_data(DATA_PATH)
    print(f"  학습: {len(train_examples)}쌍 / 평가: {len(eval_s1)}쌍")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    # MultipleNegativesRankingLoss:
    # 같은 배치 안의 다른 샘플을 자동으로 Negative로 사용
    # → Negative 쌍을 직접 만들 필요 없음
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 평가 지표: Spearman 상관계수 (구어체↔의학용어 유사도)
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        eval_s1, eval_s2, eval_scores,
        name="medical-eval"
    )

    warmup_steps = math.ceil(len(train_dataloader) * EPOCHS * WARMUP_RATIO)
    print(f"\n파인튜닝 시작 (epochs={EPOCHS}, batch={BATCH_SIZE}, warmup={warmup_steps}steps)")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        output_path=OUTPUT_PATH,
        save_best_model=True,
        show_progress_bar=True,
    )

    print(f"\n파인튜닝 완료 → {OUTPUT_PATH}")
    print("다음 단계: python deeplearning/update_vector_db.py")
