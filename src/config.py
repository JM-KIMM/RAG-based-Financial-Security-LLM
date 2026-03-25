import os

# 프로젝트 루트 디렉터리
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 데이터 디렉터리
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
FINAL_DATA_DIR = os.path.join(DATA_DIR, 'final')

# 모델 디렉터리
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')

# 개인정보보호법 판례 데이터 경로
LEGAL_DATA_DIR = os.path.join(RAW_DATA_DIR, 'legal_tiff')
TIFF_IMAGE_DIR = os.path.join(LEGAL_DATA_DIR)
OCR_OUTPUT_CSV = os.path.join(PROCESSED_DATA_DIR, 'ocr_results.csv')
LEGAL_SEMANTIC_CHUNK_OUTPUT_CSV = os.path.join(PROCESSED_DATA_DIR, 'legal_semantic_chunks.csv')

# 법률 PDF 데이터 경로
PDF_DIR = os.path.join(RAW_DATA_DIR, 'legal_pdfs')
LAW_CHUNK_OUTPUT_CSV = os.path.join(PROCESSED_DATA_DIR, 'law_chunks.csv')

# 위키백과 데이터 경로
WIKI_DATA_DIR = os.path.join(RAW_DATA_DIR, 'misc')
WIKI_SECURITY_TITLES_CSV = os.path.join(PROCESSED_DATA_DIR, 'wiki_security_titles.csv')
MISSING_WIKI_TITLES_LOG = os.path.join(PROCESSED_DATA_DIR, 'missing_wiki_titles.log')
WIKI_CHUNKS_OUTPUT_CSV = os.path.join(PROCESSED_DATA_DIR, 'wiki_chunks.csv')

# 최종 병합된 데이터셋
FINAL_DATASET_PATH = os.path.join(FINAL_DATA_DIR, 'final_combined_rag_dataset.csv')

# 데이터 처리 파라미터
LAW_CHUNK_MAX_CHARS = 500
LAW_CHUNK_OVERLAP = 50
LAW_CHUNK_MIN_CHARS = 100
DEDUP_HASHES = True

# 모델 이름
# Generator: 한국어 특화, 금융보안 도메인 용어 이해도 우수
GENERATOR_MODEL_NAME = 'A.X-4.0-Light'
# Embedding: 한국어 특화 문장 임베딩, 도메인 검색 성능 강화
EMBEDDING_MODEL_NAME = 'dragonkue/Snowflake Arctic Embed v2.0-ko'
# Reranker: Cross-Encoder 기반 정밀 리랭킹
RERANKER_MODEL_NAME = 'dragonkue/BGE Re-ranker v2 m3-ko'

# Long Context Support: 최대 16,384 토큰 (약 300~400쪽 문서 처리 가능)

# FAISS 인덱스 경로
FAISS_INDEX_PATH = os.path.join(PROCESSED_DATA_DIR, 'faiss_index.bin')

# 추론용 데이터 경로
TEST_CSV_PATH = os.path.join(DATA_DIR, 'test.csv')
SUBMISSION_CSV_PATH = os.path.join(PROJECT_DIR, 'submission.csv')
SAMPLE_SUBMISSION_CSV_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

