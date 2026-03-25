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

# 모델 Hugging Face 이름
EMBEDDING_MODEL_NAME = 'BM-K/KoSimCSE-roberta-multitask'
GENERATOR_MODEL_NAME = 'google/gemma-2b-it'
RERANKER_MODEL_NAME = 'BM-K/KoRerank-v1-distil'

# FAISS 인덱스 경로
FAISS_INDEX_PATH = os.path.join(PROCESSED_DATA_DIR, 'faiss_index.bin')

# 추론용 데이터 경로
TEST_CSV_PATH = os.path.join(DATA_DIR, 'test.csv')
SUBMISSION_CSV_PATH = os.path.join(PROJECT_DIR, 'submission.csv')
SAMPLE_SUBMISSION_CSV_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

