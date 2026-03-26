import os
import json
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# 거리 기반 3단계 판단 임계값 (짧은 문장도 통과하도록 기준 완화)
THRESHOLD_CLEAR    = 0.55  # 0.0 ~ 0.55: 확실한 의료 질문
THRESHOLD_AMBIGUOUS = 0.7  # 0.55 ~ 0.7: 모호한 질문 → 재질문 유도
                            # 0.7 초과: 의료 무관 질문 → 기각

TOP_K = 5  # 후보 반환 개수

class SymptomVectorStore:
    def __init__(self, collection_name="symptoms_collection"):
        # 메모리 기반 ChromaDB 클라이언트 (코사인 유사도 사용)
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, "deeplearning", "models", "ko-medical-sroberta")
        
        # 3단계: 초기 검색용 Bi-Encoder
        self.embedder = SentenceTransformer(model_path)
        
        # 4단계: 순위 재배열용 Cross-Encoder (한국어 문장 유사도 특화 모델)
        self.cross_encoder = CrossEncoder("bongsoo/albert-small-kor-cross-encoder-v1")

    def initialize_mapping_data(self):
        # 파인튜닝에 사용했던 750개짜리 JSON 데이터 경로 동적 로드
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(base_dir, "deeplearning", "training_data.json")

        # JSON 파일 읽기
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        symptoms = []
        metadatas = []
        ids = []

        # JSON 데이터 파싱해서 리스트에 담기
        for i, item in enumerate(dataset):
            symptoms.append(item["colloquial"])  # 구어체 증상
            metadatas.append({
                "medical_term": item["medical_term"],          # 의학 용어
                "recommended_department": item["department"]   # 진료과
            })
            ids.append(f"symptom_full_{i}")

        # 일괄 임베딩 후 벡터 DB에 삽입 (데이터가 많아 수십 초 정도 걸릴 수 있음)
        embeddings = self.embedder.encode(symptoms).tolist()
        self.collection.add(documents=symptoms, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def search_similar_symptom(self, user_input):
        """
        1차 Bi-Encoder 검색 후, 2차 Cross-Encoder로 순위를 재배열하여 판단 반환.
        """
        # 3단계: Bi-Encoder 검색 (빠르게 Top-K 추출)
        query_embedding = self.embedder.encode([user_input]).tolist()
        n = min(TOP_K, self.collection.count())
        results = self.collection.query(query_embeddings=query_embedding, n_results=n)

        if not results['distances'] or not results['distances'][0]:
            return {"status": "rejected"}

        # 4단계: Cross-Encoder 리랭킹
        candidates = []
        # [사용자 입력, 검색된 후보 문장] 쌍으로 묶기
        cross_inp = [[user_input, doc] for doc in results['documents'][0]]
        
        # Cross-Encoder로 정밀 유사도 점수 산출
        cross_scores = self.cross_encoder.predict(cross_inp)

        for i in range(len(results['distances'][0])):
            candidates.append({
                "document": results['documents'][0][i],
                "bi_distance": results['distances'][0][i], 
                "cross_score": float(cross_scores[i]), # Cross-Encoder 점수
                **results['metadatas'][0][i]
            })

        # Cross-Encoder 점수가 높은 순으로 리스트 정렬 (재배열)
        candidates = sorted(candidates, key=lambda x: x["cross_score"], reverse=True)

        # 1등으로 올라온 후보의 원래 거리(Bi-distance)를 기준으로 상태 판단
        best_distance = candidates[0]["bi_distance"]

        if best_distance < THRESHOLD_CLEAR:
            return {
                "status": "clear",
                "best_distance": best_distance,
                "top_candidates": candidates,
                "medical_term": candidates[0]["medical_term"],
                "recommended_department": candidates[0]["recommended_department"],
            }
        elif best_distance < THRESHOLD_AMBIGUOUS:
            return {
                "status": "ambiguous",
                "best_distance": best_distance,
                "top_candidates": candidates,
            }
        else:
            return {"status": "rejected", "best_distance": best_distance}