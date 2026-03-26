import os
import chromadb
from sentence_transformers import SentenceTransformer

# 거리 기반 3단계 판단 임계값 (코사인 거리: 0=동일, 1=완전 반대)
THRESHOLD_CLEAR    = 0.4  # 0.0 ~ 0.4: 확실한 의료 질문
THRESHOLD_AMBIGUOUS = 0.6  # 0.4 ~ 0.6: 모호한 질문 → 재질문 유도
                            # 0.6 초과: 의료 무관 질문 → 기각

TOP_K = 5  # 후보 반환 개수

class SymptomVectorStore:
    def __init__(self, collection_name="symptoms_collection"):
        # 메모리 기반 ChromaDB 클라이언트 (코사인 유사도 사용)
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # L2 → cosine 거리로 변경
        )
        self.embedder = SentenceTransformer(r'C:\Users\isabe\LGHelloDoctor\deeplearning\models\ko-medical-sroberta')

    def initialize_mapping_data(self):
        symptoms = [
            "쿡쿡 쑤셔요", "바늘로 찌르는 것 같아요",
            "찌릿찌릿해요", "전기가 통하는 것 같아요",
            "뻐근해요", "묵직해요", "뭉친 것 같아요"
        ]
        metadatas = [
            {"medical_term": "자통", "recommended_department": "정형외과, 신경과"},
            {"medical_term": "자통", "recommended_department": "정형외과, 신경과"},
            {"medical_term": "방사통", "recommended_department": "신경외과, 재활의학과"},
            {"medical_term": "방사통", "recommended_department": "신경외과, 재활의학과"},
            {"medical_term": "근육통", "recommended_department": "정형외과, 재활의학과"},
            {"medical_term": "근육통", "recommended_department": "정형외과, 재활의학과"},
            {"medical_term": "근육통", "recommended_department": "정형외과, 재활의학과"}
        ]
        ids = [f"symptom_{i}" for i in range(len(symptoms))]
        embeddings = self.embedder.encode(symptoms).tolist()
        self.collection.add(documents=symptoms, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def search_similar_symptom(self, user_input):
        """
        코사인 거리 기반 Top-K 검색 후 3단계 판단을 반환합니다.

        반환값:
          - status="clear"    : 확실한 의료 질문 (거리 < 0.4), top5 후보 포함
          - status="ambiguous": 모호한 질문 (0.4 <= 거리 < 0.6)
          - status="rejected" : 의료 무관 질문 (거리 >= 0.6)
        """
        query_embedding = self.embedder.encode([user_input]).tolist()
        n = min(TOP_K, self.collection.count())
        results = self.collection.query(query_embeddings=query_embedding, n_results=n)

        if not results['distances'] or not results['distances'][0]:
            return {"status": "rejected"}

        best_distance = results['distances'][0][0]

        # Top-K 후보 목록 구성
        candidates = [
            {
                "document": results['documents'][0][i],
                "distance": results['distances'][0][i],
                **results['metadatas'][0][i]
            }
            for i in range(len(results['distances'][0]))
        ]

        if best_distance < THRESHOLD_CLEAR:
            return {
                "status": "clear",
                "best_distance": best_distance,
                "top_candidates": candidates,
                # 하위 호환: 기존 llm_gpt.py가 바로 쓸 수 있도록 1위 결과도 포함
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