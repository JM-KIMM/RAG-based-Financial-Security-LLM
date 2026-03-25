import os
import pandas as pd
from tqdm import tqdm

# OCR, PDF 처리 등 실제 구현을 위해서는 아래 라이브러리 설치가 필요합니다.
# pip install pytesseract Pillow PyMuPDF semantic-text-splitter wikipedia
# 또한 Tesseract OCR 엔진이 시스템에 설치되어 있어야 합니다.

try:
    import pytesseract
    from PIL import Image
    import fitz  # PyMuPDF
    from semantic_text_splitter import TextSplitter
    import wikipedia
except ImportError:
    print("Warning: Required libraries for data processing are not installed. \
        Install them with 'pip install pytesseract Pillow PyMuPDF semantic-text-splitter wikipedia'")
    pytesseract = None
    Image = None
    fitz = None
    TextSplitter = None
    wikipedia = None


def process_tiff_images_to_csv(image_folder: str, output_csv: str):
    """
    지정된 폴더의 모든 TIFF 이미지를 OCR 처리하여 텍스트를 추출하고 CSV 파일로 저장합니다.
    실제 구현에서는 Tesseract OCR 등을 사용합니다.
    """
    if not pytesseract or not Image:
        print("Skipping TIFF processing as required libraries are not installed.")
        # Create a dummy file for pipeline continuation
        pd.DataFrame(columns=['filename', 'text']).to_csv(output_csv, index=False, encoding='utf-8-sig')
        return

    print(f"Processing TIFF images from: {image_folder}")
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.tiff', '.tif'))]
    data = []
    for filename in tqdm(image_files, desc="OCR Processing"):
        try:
            image_path = os.path.join(image_folder, filename)
            text = pytesseract.image_to_string(Image.open(image_path), lang='kor+eng')
            data.append({'filename': filename, 'text': text})
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"OCR results saved to {output_csv}")


def apply_semantic_chunking_to_csv(input_csv: str, output_csv: str, model_name: str = "klue/bert-base"):
    """
    CSV 파일의 텍스트에 Semantic Chunking을 적용하여 결과를 새 CSV 파일로 저장합니다.
    """
    if not TextSplitter:
        print("Skipping semantic chunking as required libraries are not installed.")
        # Create a dummy file for pipeline continuation
        pd.DataFrame(columns=['source', 'chunk']).to_csv(output_csv, index=False, encoding='utf-8-sig')
        return

    print(f"Applying semantic chunking to {input_csv}")
    df = pd.read_csv(input_csv)
    splitter = TextSplitter.from_huggingface_hub(model_name)
    
    chunks = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Semantic Chunking"):
        # semantic_text_splitter expects a list of documents
        doc_chunks = splitter.chunks(row['text'])
        for chunk in doc_chunks:
            chunks.append({'source': row['filename'], 'chunk': chunk})
            
    chunk_df = pd.DataFrame(chunks)
    chunk_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Semantic chunks saved to {output_csv}")


class LawPDFProcessor:
    """PDF 법률 문서를 처리하여 텍스트 청크로 분할하는 클래스."""
    def __init__(self, pdf_dir: str, output_csv: str, max_chars: int, overlap: int, min_chars: int, dedup: bool):
        self.pdf_dir = pdf_dir
        self.output_csv = output_csv
        self.max_chars = max_chars
        self.overlap = overlap
        self.min_chars = min_chars
        self.dedup = dedup
        if not fitz:
             print("Warning: PyMuPDF is not installed. PDF processing will be skipped.")

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """단일 PDF 파일에서 텍스트를 추출합니다."""
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        return text

    def run_pipeline(self):
        """지정된 디렉터리의 모든 PDF를 처리하는 전체 파이프라인을 실행합니다."""
        if not fitz:
            print("Skipping PDF processing as PyMuPDF is not installed.")
            # Create a dummy file for pipeline continuation
            pd.DataFrame(columns=['source', 'chunk']).to_csv(self.output_csv, index=False, encoding='utf-8-sig')
            return

        print(f"Processing PDFs from: {self.pdf_dir}")
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        all_chunks = []
        
        for filename in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(self.pdf_dir, filename)
            text = self._extract_text_from_pdf(pdf_path)
            
            # 간단한 텍스트 분할 (Semantic 대신)
            # 실제로는 더 정교한 분할 로직 필요
            text_chunks = [text[i:i+self.max_chars] for i in range(0, len(text), self.max_chars - self.overlap)]
            
            for chunk in text_chunks:
                if len(chunk) > self.min_chars:
                    all_chunks.append({'source': filename, 'chunk': chunk})
        
        df = pd.DataFrame(all_chunks)
        if self.dedup:
            df.drop_duplicates(subset=['chunk'], inplace=True)
            
        df.to_csv(self.output_csv, index=False, encoding='utf-8-sig')
        print(f"PDF processing complete. Chunks saved to {self.output_csv}")


class WikipediaProcessor:
    """위키백과 데이터를 가져오고 처리하는 클래스."""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        wikipedia.set_lang("ko") # 언어 설정
        if not wikipedia:
            print("Warning: 'wikipedia' library not installed. Wikipedia processing will be skipped.")

    def fetch_articles_by_title(self, titles: list, output_csv: str, missing_log_file: str):
        """주어진 제목 목록에 대해 위키백과 문서를 가져옵니다."""
        if not wikipedia:
            print("Skipping Wikipedia article fetching.")
            # Create a dummy file for pipeline continuation
            pd.DataFrame(columns=['title', 'content']).to_csv(output_csv, index=False, encoding='utf-8-sig')
            return

        print(f"Fetching Wikipedia articles for {len(titles)} titles.")
        articles = []
        missing_titles = []
        for title in tqdm(titles, desc="Fetching Wikipedia Articles"):
            try:
                page = wikipedia.page(title, auto_suggest=False)
                articles.append({'title': title, 'content': page.content})
            except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError) as e:
                print(f"Could not find page for '{title}': {e}")
                missing_titles.append(title)
        
        df = pd.DataFrame(articles)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"Wikipedia articles saved to {output_csv}")
        
        # 누락된 제목 로그
        with open(missing_log_file, 'w', encoding='utf-8') as f:
            for title in missing_titles:
                f.write(f"{title}\n")

    def run_chunking_pipeline(self, input_csv: str, output_csv: str, max_chars: int = 500, overlap: int = 50):
        """가져온 위키백과 문서를 청크로 분할합니다."""
        if not os.path.exists(input_csv):
            print(f"Skipping Wikipedia chunking as input file not found: {input_csv}")
            pd.DataFrame(columns=['source', 'chunk']).to_csv(output_csv, index=False, encoding='utf-8-sig')
            return
            
        print(f"Chunking Wikipedia articles from {input_csv}")
        df = pd.read_csv(input_csv)
        all_chunks = []
        
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Chunking Wikipedia Content"):
            content = row['content']
            chunks = [content[i:i+max_chars] for i in range(0, len(content), max_chars - overlap)]
            for chunk in chunks:
                all_chunks.append({'source': row['title'], 'chunk': chunk})
        
        chunk_df = pd.DataFrame(all_chunks)
        chunk_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"Wikipedia chunks saved to {output_csv}")
