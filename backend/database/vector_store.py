import os
import chromadb
from sentence_transformers import SentenceTransformer

# 거리 기반 3단계 판단 임계값 (코사인 거리: 0=동일, 1=완전 반대)
THRESHOLD_CLEAR    = 0.4  # 0.0 ~ 0.4: 확실한 의료 질문
THRESHOLD_AMBIGUOUS = 0.5  # 0.4 ~ 0.5: 모호한 질문 → LLM 판단
                            # 0.5 초과: 의료 무관 질문 → 기각

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
            # 자통 (찌르는 듯한 통증)
            "쿡쿡 쑤셔요", "바늘로 찌르는 것 같아요", "콕콕 찌려요", "칼로 찌르는 것 같아요", "쑤셔요",
            # 방사통 (퍼지는 통증)
            "찌릿찌릿해요", "전기가 통하는 것 같아요", "저려요", "다리가 저려요", "팔이 저려요",
            # 근육통
            "뻐근해요", "묵직해요", "뭉친 것 같아요", "결려요", "담 걸렸어요", "뻣뻣해요", "온몸이 쑤셔요",
            # 신경통/작열통
            "아려요", "아리다", "시려요", "화끈거려요", "타는 것 같아요", "욱신욱신해요",
            # 관절통
            "무릎이 아파요", "무릎이 쑤셔요", "무릎이 아리다", "관절이 아파요", "무릎이 시려요",
            "손가락이 아파요", "손목이 아파요", "어깨가 아파요", "허리가 아파요",
            # 두통
            "머리가 아파요", "머리가 지끈거려요", "머리가 띵해요", "머리가 깨질 것 같아요", "편두통이 있어요",
            # 복통/소화기
            "배가 아파요", "속이 아파요", "배가 쥐어짜듯 아파요", "소화가 안 돼요", "속이 메스꺼워요",
            "구역질이 나요", "배가 부글부글해요", "설사해요",
            # 흉통/호흡기
            "가슴이 아파요", "가슴이 답답해요", "숨이 차요", "기침이 나요", "가래가 끓어요",
            # 고령층/사투리 표현
            "삭신이 쑤신다", "뼈마디가 시리다", "뼈가 쑤신다", "온몸이 아프다", "기운이 없어요",
            "몸살 났어요", "으슬으슬해요", "몸이 무거워요", "팔다리가 쑤셔요",
            # 어지럼증/이명
            "어지러워요", "빙빙 돌아요", "귀에서 소리가 나요", "귀가 먹먹해요",
            # 피부
            "피부가 가려워요", "발진이 생겼어요", "두드러기가 났어요", "피부가 붉어졌어요",
            # 안과
            "눈이 충혈됐어요", "눈이 침침해요", "눈이 뿌옇게 보여요", "눈에서 눈물이 나요",
            # 이비인후과
            "귀가 아파요", "코가 막혀요", "목이 아파요", "목이 칼칼해요", "콧물이 나요", "목에 뭔가 걸린 것 같아요",
            # 산부인과
            "생리통이 심해요", "아랫배가 아파요", "냉대하가 있어요", "생리가 불규칙해요", "아랫배가 묵직해요",
            # 비뇨의학과
            "소변볼 때 아파요", "소변이 자주 마려워요", "소변 줄기가 약해요", "소변에 피가 섞여 나와요",
            # 정신건강의학과
            "잠을 못 자요", "불안해요", "우울해요", "의욕이 없어요", "자꾸 깜빡해요",
            # 치과
            "이가 아파요", "잇몸이 부었어요", "이가 흔들려요", "이가 시려요", "잇몸에서 피가 나요",
            # 한의원
            "체한 것 같아요", "만성피로예요", "몸이 찬 것 같아요", "소화가 늘 안 좋아요",
        ]
        metadatas = [
            # 자통
            {"medical_term": "자통", "recommended_department": "정형외과"},
            {"medical_term": "자통", "recommended_department": "정형외과"},
            {"medical_term": "자통", "recommended_department": "정형외과"},
            {"medical_term": "자통", "recommended_department": "정형외과"},
            {"medical_term": "자통", "recommended_department": "정형외과"},
            # 방사통
            {"medical_term": "방사통", "recommended_department": "정형외과"},
            {"medical_term": "방사통", "recommended_department": "정형외과"},
            {"medical_term": "방사통", "recommended_department": "정형외과"},
            {"medical_term": "방사통", "recommended_department": "정형외과"},
            {"medical_term": "방사통", "recommended_department": "정형외과"},
            # 근육통
            {"medical_term": "근육통", "recommended_department": "정형외과"},
            {"medical_term": "근육통", "recommended_department": "정형외과"},
            {"medical_term": "근육통", "recommended_department": "정형외과"},
            {"medical_term": "근육통", "recommended_department": "정형외과"},
            {"medical_term": "근육통", "recommended_department": "정형외과"},
            {"medical_term": "근육통", "recommended_department": "정형외과"},
            {"medical_term": "근육통", "recommended_department": "정형외과"},
            # 신경통/작열통
            {"medical_term": "신경통", "recommended_department": "내과"},
            {"medical_term": "신경통", "recommended_department": "내과"},
            {"medical_term": "신경통", "recommended_department": "내과"},
            {"medical_term": "작열통", "recommended_department": "내과"},
            {"medical_term": "작열통", "recommended_department": "내과"},
            {"medical_term": "박동통", "recommended_department": "내과"},
            # 관절통
            {"medical_term": "관절통", "recommended_department": "정형외과"},
            {"medical_term": "관절통", "recommended_department": "정형외과"},
            {"medical_term": "관절통", "recommended_department": "정형외과"},
            {"medical_term": "관절통", "recommended_department": "정형외과"},
            {"medical_term": "관절통", "recommended_department": "정형외과"},
            {"medical_term": "관절통", "recommended_department": "정형외과"},
            {"medical_term": "관절통", "recommended_department": "정형외과"},
            {"medical_term": "어깨통증", "recommended_department": "정형외과"},
            {"medical_term": "요통", "recommended_department": "정형외과"},
            # 두통
            {"medical_term": "두통", "recommended_department": "내과"},
            {"medical_term": "두통", "recommended_department": "내과"},
            {"medical_term": "두통", "recommended_department": "내과"},
            {"medical_term": "두통", "recommended_department": "내과"},
            {"medical_term": "편두통", "recommended_department": "내과"},
            # 복통/소화기
            {"medical_term": "복통", "recommended_department": "내과"},
            {"medical_term": "복통", "recommended_department": "내과"},
            {"medical_term": "복통", "recommended_department": "내과"},
            {"medical_term": "소화불량", "recommended_department": "내과"},
            {"medical_term": "오심", "recommended_department": "내과"},
            {"medical_term": "오심", "recommended_department": "내과"},
            {"medical_term": "복통", "recommended_department": "내과"},
            {"medical_term": "설사", "recommended_department": "내과"},
            # 흉통/호흡기
            {"medical_term": "흉통", "recommended_department": "내과"},
            {"medical_term": "흉부압박감", "recommended_department": "내과"},
            {"medical_term": "호흡곤란", "recommended_department": "내과"},
            {"medical_term": "기침", "recommended_department": "내과"},
            {"medical_term": "객담", "recommended_department": "내과"},
            # 고령층/사투리
            {"medical_term": "근육통", "recommended_department": "정형외과"},
            {"medical_term": "관절통", "recommended_department": "정형외과"},
            {"medical_term": "골통", "recommended_department": "정형외과"},
            {"medical_term": "근육통", "recommended_department": "정형외과"},
            {"medical_term": "피로", "recommended_department": "내과"},
            {"medical_term": "근육통, 발열", "recommended_department": "내과"},
            {"medical_term": "오한", "recommended_department": "내과"},
            {"medical_term": "피로", "recommended_department": "내과"},
            {"medical_term": "근육통", "recommended_department": "정형외과"},
            # 어지럼증/이명
            {"medical_term": "어지럼증", "recommended_department": "내과"},
            {"medical_term": "어지럼증", "recommended_department": "내과"},
            {"medical_term": "이명", "recommended_department": "이비인후과"},
            {"medical_term": "이충만감", "recommended_department": "이비인후과"},
            # 피부
            {"medical_term": "소양감", "recommended_department": "피부과"},
            {"medical_term": "발진", "recommended_department": "피부과"},
            {"medical_term": "두드러기", "recommended_department": "피부과"},
            {"medical_term": "피부염", "recommended_department": "피부과"},
            # 안과
            {"medical_term": "결막충혈", "recommended_department": "안과"},
            {"medical_term": "시력저하", "recommended_department": "안과"},
            {"medical_term": "백내장 의심", "recommended_department": "안과"},
            {"medical_term": "유루증", "recommended_department": "안과"},
            # 이비인후과
            {"medical_term": "이통", "recommended_department": "이비인후과"},
            {"medical_term": "비충혈", "recommended_department": "이비인후과"},
            {"medical_term": "인후통", "recommended_department": "이비인후과"},
            {"medical_term": "인후통", "recommended_department": "이비인후과"},
            {"medical_term": "비루", "recommended_department": "이비인후과"},
            {"medical_term": "이물감", "recommended_department": "이비인후과"},
            # 산부인과
            {"medical_term": "월경통", "recommended_department": "산부인과"},
            {"medical_term": "골반통", "recommended_department": "산부인과"},
            {"medical_term": "냉대하", "recommended_department": "산부인과"},
            {"medical_term": "월경불순", "recommended_department": "산부인과"},
            {"medical_term": "골반압박감", "recommended_department": "산부인과"},
            # 비뇨의학과
            {"medical_term": "배뇨통", "recommended_department": "비뇨의학과"},
            {"medical_term": "빈뇨", "recommended_department": "비뇨의학과"},
            {"medical_term": "전립선비대 의심", "recommended_department": "비뇨의학과"},
            {"medical_term": "혈뇨", "recommended_department": "비뇨의학과"},
            # 정신건강의학과
            {"medical_term": "불면증", "recommended_department": "정신건강의학과"},
            {"medical_term": "불안장애", "recommended_department": "정신건강의학과"},
            {"medical_term": "우울증", "recommended_department": "정신건강의학과"},
            {"medical_term": "무기력증", "recommended_department": "정신건강의학과"},
            {"medical_term": "인지저하", "recommended_department": "정신건강의학과"},
            # 치과
            {"medical_term": "치통", "recommended_department": "치과"},
            {"medical_term": "치은염", "recommended_department": "치과"},
            {"medical_term": "치아동요", "recommended_department": "치과"},
            {"medical_term": "치아과민증", "recommended_department": "치과"},
            {"medical_term": "치은출혈", "recommended_department": "치과"},
            # 한의원
            {"medical_term": "식체", "recommended_department": "한의원"},
            {"medical_term": "만성피로", "recommended_department": "한의원"},
            {"medical_term": "수족냉증", "recommended_department": "한의원"},
            {"medical_term": "소화불량(만성)", "recommended_department": "한의원"},
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