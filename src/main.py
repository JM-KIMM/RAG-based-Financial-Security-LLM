import argparse
import os
import pandas as pd
from tqdm import tqdm

from src import config
from src.data_processing import (
    process_tiff_images_to_csv, 
    apply_semantic_chunking_to_csv,
    LawPDFProcessor,
    WikipediaProcessor
)
from src.rag_pipeline import RAGPipeline
from scripts.download_models import download_models

def run_data_preprocessing():
    """전체 데이터 전처리 파이프라인을 실행합니다."""
    print("Starting data preprocessing pipeline...")
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.FINAL_DATA_DIR, exist_ok=True)

    # 1. 개인정보보호법 판례 데이터 처리
    print("\nStep 1: Processing TIFF images...")
    process_tiff_images_to_csv(
        image_folder=config.TIFF_IMAGE_DIR,
        output_csv=config.OCR_OUTPUT_CSV
    )
    print("\nStep 2: Applying semantic chunking to legal data...")
    apply_semantic_chunking_to_csv(
        input_csv=config.OCR_OUTPUT_CSV,
        output_csv=config.LEGAL_SEMANTIC_CHUNK_OUTPUT_CSV
    )

    # 2. 법률 PDF 데이터 처리
    print("\nStep 3: Processing law PDFs...")
    pdf_processor = LawPDFProcessor(
        pdf_dir=config.PDF_DIR,
        output_csv=config.LAW_CHUNK_OUTPUT_CSV,
        max_chars=config.LAW_CHUNK_MAX_CHARS,
        overlap=config.LAW_CHUNK_OVERLAP,
        min_chars=config.LAW_CHUNK_MIN_CHARS,
        dedup=config.DEDUP_HASHES
    )
    pdf_processor.run_pipeline()

    # 3. 위키백과 데이터 처리 (실제 실행을 위해서는 titles 리스트 필요)
    print("\nStep 4: Processing Wikipedia data...")
    # 예시: 보안 관련 제목 리스트 - 실제 프로젝트에서는 파일이나 DB에서 가져와야 함
    wiki_titles = ["정보 보안", "개인 정보", "데이터베이스 보안", "네트워크 보안", "암호학"]
    wiki_processor = WikipediaProcessor(output_dir=config.PROCESSED_DATA_DIR)
    wiki_processor.fetch_articles_by_title(
        titles=wiki_titles,
        output_csv=config.WIKI_SECURITY_TITLES_CSV,
        missing_log_file=config.MISSING_WIKI_TITLES_LOG
    )
    wiki_processor.run_chunking_pipeline(
        input_csv=config.WIKI_SECURITY_TITLES_CSV,
        output_csv=config.WIKI_CHUNKS_OUTPUT_CSV
    )

    # 4. 모든 청크 데이터 병합
    print("\nStep 5: Merging all chunked data...")
    df1 = pd.read_csv(config.LEGAL_SEMANTIC_CHUNK_OUTPUT_CSV)
    df2 = pd.read_csv(config.LAW_CHUNK_OUTPUT_CSV)
    df3 = pd.read_csv(config.WIKI_CHUNKS_OUTPUT_CSV)
    
    final_df = pd.concat([df1, df2, df3], ignore_index=True)
    final_df.to_csv(config.FINAL_DATASET_PATH, index=False, encoding='utf-8-sig')
    
    print(f"Data preprocessing complete. Final dataset saved to {config.FINAL_DATASET_PATH}")
    print(f"Total chunks: {len(final_df)}")

def run_inference():
    """추론 파이프라인을 실행하여 submission 파일을 생성합니다."""
    print("Starting inference pipeline...")
    
    if not os.path.exists(config.TEST_CSV_PATH):
        print(f"Test data not found at {config.TEST_CSV_PATH}. Aborting.")
        return
        
    test_df = pd.read_csv(config.TEST_CSV_PATH)
    
    try:
        rag_pipeline = RAGPipeline()
    except FileNotFoundError as e:
        print(f"Error initializing RAG pipeline: {e}")
        print("Please make sure models are downloaded and data is preprocessed.")
        return

    predictions = []
    for question in tqdm(test_df['Question'], desc="Running Inference"):
        try:
            answer = rag_pipeline.infer(question)
            predictions.append(answer)
        except Exception as e:
            print(f"Error processing question: '{question[:50]}...'. Error: {e}")
            predictions.append("Error") # 에러 발생 시 'Error'로 표기

    submission_df = pd.read_csv(config.SAMPLE_SUBMISSION_CSV_PATH)
    submission_df['Answer'] = predictions
    submission_df.to_csv(config.SUBMISSION_CSV_PATH, index=False, encoding='utf-8-sig')

    print(f"Inference complete. Submission file saved to {config.SUBMISSION_CSV_PATH}")

def main():
    """메인 실행 함수."""
    parser = argparse.ArgumentParser(description="Financial AI RAG Project CLI")
    
    parser.add_argument(
        'action', 
        choices=['download', 'preprocess', 'inference'],
        help="Action to perform: 'download' models, 'preprocess' data, or run 'inference'."
    )
    
    args = parser.parse_args()
    
    if args.action == 'download':
        print("--- Running Model Downloader ---")
        download_models()
    elif args.action == 'preprocess':
        print("--- Running Data Preprocessing Pipeline ---")
        run_data_preprocessing()
    elif args.action == 'inference':
        print("--- Running Inference Pipeline ---")
        run_inference()

if __name__ == "__main__":
    main()
