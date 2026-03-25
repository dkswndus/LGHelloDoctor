import os
import chromadb
from sentence_transformers import SentenceTransformer

class SymptomVectorStore:
    def __init__(self, collection_name="symptoms_collection"):
        # 메모리 기반 ChromaDB 클라이언트
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = SentenceTransformer('jhgan/ko-sroberta-multitask')

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

    def search_similar_symptom(self, user_input, threshold=100.0): # 기준치를 100.0으로 재설정
        """
        사용자 입력과 가장 의미가 유사한 증상을 검색합니다.
        """
        query_embedding = self.embedder.encode([user_input]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=1)
        
        if not results['distances'] or not results['distances'][0]:
            return None
            
        distance = results['distances'][0][0]
        
        # 100보다 크면(멀면) 무관한 질문으로 판단
        if distance > threshold:
            return {"status": "low_confidence"}
            
        return results['metadatas'][0][0]