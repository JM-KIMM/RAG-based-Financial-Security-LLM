# 📘 금융 QA 응답기 (Financial Q&A System)

## 📋 프로젝트 개요

금융 QA 응답기는 **RAG(Retrieval-Augmented Generation)** 기술을 활용하여 금융 관련 질문에 정확한 답변을 제공하는 AI 시스템입니다.

개인정보보호법, 금융보안법률, 보안 관련 다양한 데이터를 기반으로 학습된 모델이 사용자 질문에 대해 관련 문서를 검색하고 정확한 답변을 생성합니다.

---

## ✨ 주요 기능

- **RAG 파이프라인**: 벡터 데이터베이스를 활용한 빠른 문서 검색 및 답변 생성
- **한국어 최적화**: 한국어 특화 모델(KoSimCSE, KoRerank) 사용
- **다양한 데이터 소스**: 
  - 개인정보보호법 판례 (TIFF 이미지 → OCR)
  - 금융보안 법률 문서 (PDF)
  - 위키백과 보안 관련 정보
- **Streamlit UI**: 사용자 친화적인 웹 인터페이스
- **GPU 지원**: CUDA를 통한 고속 추론

---

## 🛠️ 기술 스택

| 구성요소 | 모델/라이브러리 |
|---------|-----------------|
| **Embedding Model** | `BM-K/KoSimCSE-roberta-multitask` |
| **Generator Model** | `google/gemma-2b-it` |
| **Reranker Model** | `BM-K/KoRerank-v1-distil` |
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

## 🤖 RAG 추론 과정

```
사용자 질문
    ↓
1. 임베딩 (Embedding)
    ↓
2. 벡터 검색 (Vector Search via FAISS)
    ↓
3. 문서 재정렬 (Reranking)
    ↓
4. 답변 생성 (Generation)
    ↓
최종 답변 반환
```

---

## ⚙️ 주요 설정 (config.py)

```python
# 모델 설정
EMBEDDING_MODEL_NAME = 'BM-K/KoSimCSE-roberta-multitask'
GENERATOR_MODEL_NAME = 'google/gemma-2b-it'
RERANKER_MODEL_NAME = 'BM-K/KoRerank-v1-distil'

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
