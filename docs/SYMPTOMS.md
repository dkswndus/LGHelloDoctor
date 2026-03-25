---

### 2. `docs/SYMPTOMS.md` (통증 매핑 사전)
이 파일은 프로젝트의 '의학적 뇌'에 해당합니다. 나중에 데이터를 추가할 때 이 문서를 보고 업데이트하게 됩니다.

```markdown
# 통증 표현 - 의학 용어 매핑 가이드라인

본 문서는 사용자의 구어체 발화를 벡터 DB(ChromaDB)에 저장하기 위한 표준 가이드라인입니다.

| 구어체 표현 (환자 발화) | 의학적 통증 용어 | 연관 진료과 |
| :--- | :--- | :--- |
| "쿡쿡 쑤셔요", "바늘로 찌르는 것 같아요" | 자통 (Pricking pain) | 정형외과, 신경과 |
| "찌릿찌릿해요", "전기가 통해요" | 방사통 (Radiating pain) | 신경외과, 재활의학과 |
| "뻐근해요", "묵직해요", "뭉쳤어요" | 근육통 (Myalgia) | 정형외과, 재활의학과 |
| "욱신거려요", "두근거려요" | 박동성 통증 (Throbbing pain) | 치과, 이비인후과 |
| "타는 듯해요", "화끈거려요" | 작열통 (Burning pain) | 피부과, 내과 |
| "쥐어짜는 것 같아요", "꼬여요" | 산통 (Colic pain) | 소화기내과, 산부인과 |
| "찢어지는 것 같아요", "터질 것 같아요" | 열창통 (Tearing pain) | 응급의학과, 외과 |

## 벡터 DB 검색 정책
- **Embedding Model**: `jhgan/ko-sroberta-multitask`
- **Search Metric**: Cosine Similarity (ChromaDB default)
- **Threshold**: `100.0` (이 수치보다 높으면 `low_confidence`로 처리)