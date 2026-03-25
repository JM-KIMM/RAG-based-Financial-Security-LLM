# 📘 금융 QA 응답기 (Financial Q&A System)

## 📋 프로젝트 개요

금융 QA 응답기는 **RAG(Retrieval-Augmented Generation)** 기술을 활용하여 금융 관련 질문에 정확한 답변을 제공하는 AI 시스템입니다.

개인정보보호법, 금융보안법률, 보안 관련 다양한 데이터를 기반으로 학습된 모델이 사용자 질문에 대해 관련 문서를 검색하고 정확한 답변을 생성합니다.

---

## ✨ 주요 기능

- **RAG 파이프라인**: 벡터 데이터베이스를 활용한 빠른 문서 검색 및 답변 생성
- **한국어 특화**: A.X-4.0-Light LLM, Arctic Embed 임베딩, BGE Re-ranker로 금융보안 도메인 최적화
- **장문 처리**: 최대 16,384 토큰 컨텍스트 윈도우 (약 300~400쪽 분량 문서 처리 가능)
- **다양한 데이터 소스**: 
  - 개인정보보호법 판례 (TIFF 이미지 → OCR)
  - 금융보안 법률 문서 (PDF)
  - 위키백과 보안 관련 정보
- **Streamlit UI**: 사용자 친화적인 웹 인터페이스
- **GPU 지원**: CUDA를 통한 고속 추론

---

## 🛠️ 기술 스택

| 구성요소 | 모델/라이브러리 | 설명 |
|---------|-----------------|------|
| **Generator (LLM)** | `A.X-4.0-Light` | 한국어 특화, 금융보안 용어 이해도 우수, 16,384 토큰 컨텍스트 |
| **Embedding Model** | `dragonkue/Snowflake Arctic Embed v2.0-ko` | 한국어 특화 문장 임베딩, 도메인 검색 성능 강화 |
| **Reranker Model** | `dragonkue/BGE Re-ranker v2 m3-ko` | Cross-Encoder 기반 정밀 리랭킹 |
| **Vector Database** | FAISS |
| **UI Framework** | Streamlit |
| **Dependencies** | torch, transformers, sentence-transformers |

---

## 📁 프로젝트 구조

```
금융ai제출물/
├── streamlit_app.py              # 메인 Streamlit 앱
├── src/
│   ├── main.py                   # 데이터 전처리 및 추론 스크립트
│   ├── rag_pipeline.py           # RAG 파이프라인 클래스
│   ├── data_processing.py        # 데이터 처리 모듈
│   ├── config.py                 # 설정 및 경로 관리
│   └── utils.py                  # 유틸리티 함수
├── notebooks/
│   ├── 1_data_preprocessing.ipynb      # 데이터 전처리 노트북
│   ├── 2_inference_setup.ipynb         # 모델 다운로드 및 환경 설정
│   └── 3_final_inference.ipynb         # 최종 추론 노트북
├── scripts/
│   └── download_models.py        # 모델 자동 다운로드 스크립트
├── data/
│   ├── raw/                      # 원본 데이터
│   │   ├── legal_tiff/           # 판례 TIFF 이미지
│   │   ├── legal_pdfs/           # 법률 PDF 문서
│   │   └── misc/                 # 기타 자료
│   ├── processed/                # 전처리된 데이터
│   └── final/                    # 최종 병합 데이터셋
└── 데이터셋/                      # 학습용 데이터셋

```

---

## 🚀 빠른 시작

### 필수 요구사항

- Python 3.8+
- CUDA 가능한 GPU (권장)
- 40GB 이상의 저장 공간 (모델 포함)

### 설치 및 실행

#### 1단계: 환경 설정 및 모델 다운로드

**Jupyter Notebook을 사용하는 경우:**
```bash
# notebooks/2_inference_setup.ipynb 실행
# 필요한 라이브러리 및 모델이 자동 다운로드됩니다.
```

**또는 직접 스크립트 실행:**
```bash
python scripts/download_models.py
```

#### 2단계: Streamlit 앱 실행

```bash
streamlit run streamlit_app.py
```

또는

```bash
py -m streamlit run streamlit_app.py
```

브라우저에서 `http://localhost:8501`로 접속하면 웹 인터페이스를 사용할 수 있습니다.

---

## 📊 데이터 처리 파이프라인

### 1. 데이터 소스

| 데이터 | 출처 | 형식 | 용도 |
|--------|------|------|------|
| 판례 데이터 | 개인정보보호위원회 판례집 | TIFF 이미지 | OCR → 의미론적 청킹 |
| 법률 문서 | 금융보안관련법률 | PDF | 청크 분할 |
| 보안 정보 | 위키백과 | 웹 스크래핑 | 배경지식 |

### 2. 전처리 단계

```
원본 데이터
    ↓
1. OCR (TIFF 이미지 → 텍스트)
    ↓
2. 의미론적 청킹 (Semantic Chunking)
    ↓
3. 중복 제거 (Deduplication)
    ↓
4. 임베딩 생성 및 FAISS 인덱싱
    ↓
최종 RAG 데이터셋 생성
```

---

## 🤖 A.X-4.0-Light RAG 추론 파이프라인
<img width="5906" height="4986" alt="image" src="https://github.com/user-attachments/assets/aaed07cf-2a21-4c51-8f1a-971ebd4f1fe9" />

### 전체 흐름도

```
┌─────────────────────────────────────────────────────────────────┐
│                        사용자 질문 입력                          │
│                    (User Input)                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
            ┌────────────────▼────────────────┐
            │     A.X-4.0-Light 입력 처리      │
            │   - 증강 (Augmentation)         │
            │   - 프롬프트 최적화             │
            └────────────────┬────────────────┘
                             │
            ┌────────────────▼────────────────┐
            │        문서 청킹 (Chunking)      │
            │   - 최대 16,384 토큰            │
            │   - 의미론적 단위로 분할        │
            └────────────────┬────────────────┘
                             │
            ┌────────────────▼───────────────────────────┐
            │  Snowflake Arctic Embed v2.0-ko 임베딩      │
            │  - 한국어 특화 임베딩                      │
            │  - 도메인 최적화                          │
            └────────────────┬───────────────────────────┘
                             │
            ┌────────────────▼──────────────────┐
            │   FAISS 벡터 데이터베이스          │
            │   - O(log n) 검색 최적화           │
            │   - 관련 문서 상위 K개 검색        │
            └────────────────┬──────────────────┘
                             │
            ┌────────────────▼──────────────────────────┐
            │   BGE Re-ranker v2 m3-ko 리랭킹           │
            │   - Cross-Encoder 기반                    │
            │   - 정답 근거를 상위에 배치                │
            │   - FAISS 검색의 한계 보완                │
            └────────────────┬──────────────────────────┘
                             │
            ┌────────────────▼─────────────────────┐
            │   Context 구성                       │
            │   - 재정렬된 문서들을 프롬프트에    │
            │   - 최대 컨텍스트 윈도우 활용       │
            └────────────────┬─────────────────────┘
                             │
            ┌────────────────▼──────────────────┐
            │  A.X-4.0-Light 답변 생성            │
            │  - 금융보안 도메인 최적화           │
            │  - Context 기반 생성                │
            └────────────────┬──────────────────┘
                             │
            ┌────────────────▼──────────────────┐
            │   후처리 (Post-processing)         │
            │   - 응답 포맷팅                    │
            │   - 근거문서 표시                  │
            └────────────────┬──────────────────┘
                             │
        ┌────────────────────▼────────────────────┐
        │         최종 답변 반환                   │
        │   • 생성된 답변                         │
        │   • 관련 문서/근거                      │
        │   • 신뢰도 점수                        │
        └─────────────────────────────────────────┘
```

### 각 단계별 상세 설명

#### 1️⃣ 입력 처리 (Input Augmentation)
- 사용자 질문을 분석하여 범주 분류
- 필요시 추가 컨텍스트 자동 추가
- 금융보안 도메인 용어 인식 및 정규화

#### 2️⃣ 청킹 (Chunking)
- **최대 토큰**: 16,384 (약 300~400쪽 문서)
- **청킹 방식**: 의미론적 단위로 분할
- **오버랩**: 문맥 연속성 유지
- **목적**: 검색 정확도와 생성 품질 균형

#### 3️⃣ 임베딩 (Embedding)
- **모델**: `Snowflake Arctic Embed v2.0-ko`
- **특징**:
  - 한국어 문장 임베딩 최적화
  - 금융보안 도메인 용어 이해도 우수
  - 의미 유사도 검색 성능 강화

#### 4️⃣ 벡터 검색 (Vector Search)
- **방식**: FAISS 인덱싱
- **속도**: O(log n)으로 대규모 문서도 빠른 검색
- **결과**: 관련도 상위 K개 문서 추출

#### 5️⃣ 리랭킹 (Reranking)
- **모델**: `BGE Re-ranker v2 m3-ko`
- **방식**: Cross-Encoder 기반 정밀 평가
- **목적**: FAISS 검색의 한계 보완, 정답 근거를 상위에 배치

#### 6️⃣ Context 구성
- 리랭킹된 상위 문서들을 조직적으로 구성
- A.X-4.0-Light의 최대 컨텍스트 윈도우(16,384 토큰) 활용
- 검색 신뢰도 정보 포함

#### 7️⃣ 답변 생성 (Generation)
- **모델**: `A.X-4.0-Light`
- **특징**:
  - 한국어 특화, 금융보안 도메인 용어 이해도 우수
  - RAG와 높은 호환성
  - 외부 지식 컨텍스트를 반영한 정확한 응답

#### 8️⃣ 후처리 (Post-processing)
- 응답 포맷팅 및 가독성 개선
- 출처 표시 및 신뢰도 점수 추가
- 추가 설명 또는 관련 질문 제시

---

## ⚙️ 주요 설정 (config.py)

```python
# 모델 설정
GENERATOR_MODEL_NAME = 'A.X-4.0-Light'  # 한국어 특화 LLM
EMBEDDING_MODEL_NAME = 'dragonkue/Snowflake Arctic Embed v2.0-ko'  # 한국어 임베딩
RERANKER_MODEL_NAME = 'dragonkue/BGE Re-ranker v2 m3-ko'  # Cross-Encoder 리랭커

# 청킹 파라미터
LAW_CHUNK_MAX_CHARS = 500      # 최대 청크 크기
LAW_CHUNK_OVERLAP = 50         # 청크 겹침
LAW_CHUNK_MIN_CHARS = 100      # 최소 청크 크기

# 파일 경로
FINAL_DATASET_PATH = 'data/final/final_combined_rag_dataset.csv'
FAISS_INDEX_PATH = 'data/processed/faiss_index.bin'
```

---

## 📝 사용 예시

### Streamlit 웹 인터페이스 사용

1. 웹 인터페이스 접속 (`http://localhost:8501`)
2. 질문 입력란에 금융/보안 관련 질문 입력
3. "🔍 응답 생성" 버튼 클릭
4. 생성된 답변 확인

### Python 스크립트에서 직접 사용

```python
from src.rag_pipeline import RAGPipeline

# RAG 파이프라인 초기화
pipeline = RAGPipeline()

# 질문 처리
question = "개인정보 유출 시 정보주체의 권리는?"
answer = pipeline.generate_answer(question)
print(answer)
```

---

## 📦 의존성 설치

```bash
# pip를 통한 설치
pip install torch transformers faiss-cpu sentence-transformers streamlit tqdm pillow

# CUDA 지원 설치 (GPU 사용 시)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🔄 데이터 전처리 실행

전체 데이터 전처리 파이프라인을 실행하려면:

```bash
python src/main.py --run-preprocessing
```

또는 Jupyter Notebook 사용:
```bash
jupyter notebook notebooks/1_data_preprocessing.ipynb
```

---

## 📈 성능 최적화

- **GPU 사용**: CUDA 가능한 GPU에서 실행 시 10배 이상 속도 향상
- **배치 처리**: 여러 질문을 배치로 처리 가능
- **캐싱**: Streamlit의 `@st.cache_resource` 데코레이터로 모델 로드 최적화
- **FAISS 인덱싱**: 백터 검색을 O(log n)으로 최적화

---

## 🐛 트러블슈팅

### 모델 로딩 에러
```
ModuleNotFoundError: No module named 'torch'
```
**해결책**: `pip install torch transformers sentence-transformers` 실행

### CUDA 메모리 부족
```
RuntimeError: CUDA out of memory
```
**해결책**: 
- `config.py`에서 배치 크기 감소
- CPU 모드로 실행: 코드에서 `device='cpu'` 설정

### 모델 다운로드 실패
- 인터넷 연결 확인
- Hugging Face Hub 접근 가능 확인
- `notebooks/2_inference_setup.ipynb` 다시 실행

---

## 📚 참고 자료

- [Hugging Face Transformers 문서](https://huggingface.co/docs/transformers/)
- [FAISS 문서](https://faiss.ai/)
- [Streamlit 문서](https://docs.streamlit.io/)
- [RAG 개념 설명](https://arxiv.org/abs/2005.11401)

---

## 📄 라이선스

이 프로젝트는 [LICENSE](LICENSE) 파일을 참고하세요.

---

## 🤝 기여

버그 리포트 및 기능 요청은 이슈 탭에서 제출해주세요.

---

## ✉️ 연락처

프로젝트 관련 문의: [이메일/연락처]

---

**마지막 업데이트**: 2026년 3월
