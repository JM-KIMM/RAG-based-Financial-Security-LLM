import os
import pandas as pd
import numpy as np
from src import config

# 실제 구현을 위해서는 아래 라이브러리 설치가 필요합니다.
# pip install torch transformers faiss-cpu sentence-transformers
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration
    import torch
except ImportError:
    print("Warning: Required libraries for RAG pipeline are not installed. \
        Install them with 'pip install torch transformers faiss-cpu sentence-transformers'")
    faiss = None
    SentenceTransformer = None
    AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration = None, None, None
    torch = None

class RAGPipeline:
    """
    RAG(Retrieval-Augmented Generation) 파이프라인을 관리하는 클래스.
    Retriever, Reranker, Generator 모델을 로드하고 추론을 수행합니다.
    """
    def __init__(self):
        if not all([faiss, SentenceTransformer, AutoTokenizer, AutoModelForCausalLM, torch]):
            print("Cannot initialize RAGPipeline due to missing libraries.")
            return

        print("Initializing RAG Pipeline...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # 1. 문서 데이터 및 FAISS 인덱스 로드
        self.documents_df = self._load_documents()
        self.index = self._load_faiss_index()

        # 2. 모델 로드
        self.retriever = self._load_retriever(config.EMBEDDING_MODEL_NAME)
        self.reranker = self._load_reranker(config.RERANKER_MODEL_NAME)
        self.generator = self._load_generator(config.GENERATOR_MODEL_NAME)
        self.tokenizer = self._load_tokenizer(config.GENERATOR_MODEL_NAME)
        print("RAG Pipeline initialized successfully.")

    def _load_documents(self) -> pd.DataFrame:
        """전체 문서(청크) 데이터를 로드합니다."""
        if not os.path.exists(config.FINAL_DATASET_PATH):
            raise FileNotFoundError(f"Final dataset not found at: {config.FINAL_DATASET_PATH}")
        print(f"Loading documents from {config.FINAL_DATASET_PATH}")
        return pd.read_csv(config.FINAL_DATASET_PATH)

    def _load_faiss_index(self):
        """미리 빌드된 FAISS 인덱스를 로드합니다."""
        if not os.path.exists(config.FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at: {config.FAISS_INDEX_PATH}. \
                Please build it first using a script.")
        print(f"Loading FAISS index from {config.FAISS_INDEX_PATH}")
        return faiss.read_index(config.FAISS_INDEX_PATH)

    def _load_retriever(self, model_name: str) -> SentenceTransformer:
        """임베딩 모델(Retriever)을 로드합니다."""
        print(f"Loading Retriever model: {model_name}")
        model_path = os.path.join(config.MODEL_DIR, model_name.replace('/', '_'))
        if not os.path.isdir(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}. Please download models first.")
        return SentenceTransformer(model_path, device=self.device)

    def _load_reranker(self, model_name: str):
        """Reranker 모델을 로드합니다."""
        # Reranker는 보통 Cross-Encoder이므로 SentenceTransformer와 다를 수 있습니다.
        # 여기서는 간단하게 동일한 클래스를 사용했지만, 실제로는 맞는 아키텍처를 사용해야 합니다.
        print(f"Loading Reranker model: {model_name}")
        model_path = os.path.join(config.MODEL_DIR, model_name.replace('/', '_'))
        if not os.path.isdir(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}. Please download models first.")
        # 실제로는 CrossEncoder 같은 클래스를 사용해야 할 수 있습니다.
        return SentenceTransformer(model_path, device=self.device)

    def _load_generator(self, model_name: str):
        """생성 모델(LLM)을 로드합니다."""
        print(f"Loading Generator model: {model_name}")
        model_path = os.path.join(config.MODEL_DIR, model_name.replace('/', '_'))
        if not os.path.isdir(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}. Please download models first.")
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.float16,
        )

    def _load_tokenizer(self, model_name: str) -> AutoTokenizer:
        """생성 모델용 토크나이저를 로드합니다."""
        print(f"Loading Tokenizer for: {model_name}")
        model_path = os.path.join(config.MODEL_DIR, model_name.replace('/', '_'))
        if not os.path.isdir(model_path):
             raise FileNotFoundError(f"Tokenizer not found at {model_path}. Please download models first.")
        return AutoTokenizer.from_pretrained(model_path)

    def _retrieve(self, question: str, k: int = 50) -> list:
        """질문에 대한 문서를 검색(Retrieve)합니다."""
        query_embedding = self.retriever.encode([question], convert_to_tensor=True)
        _, top_k_indices = self.index.search(query_embedding.cpu().numpy(), k)
        
        retrieved_docs = self.documents_df.iloc[top_k_indices[0]]
        return retrieved_docs['chunk'].tolist()

    def _rerank(self, question: str, documents: list, top_n: int = 5) -> list:
        """검색된 문서를 질문과의 관련성을 기준으로 재정렬(Rerank)합니다."""
        # Reranker (Cross-Encoder)는 [question, doc] 쌍으로 점수를 매깁니다.
        pairs = [[question, doc] for doc in documents]
        
        # SentenceTransformer의 cross-encoder는 predict 메소드를 사용합니다.
        # 여기서는 retriever 모델로 점수를 계산하는 것으로 대체합니다.
        scores = self.reranker.encode(pairs, convert_to_tensor=True, show_progress_bar=False)
        
        # 이 예제에서는 코사인 유사도로 점수를 계산합니다. 실제 reranker는 다른 출력을 가질 수 있습니다.
        # 점수가 높을수록 관련성이 높다고 가정합니다.
        # scores는 각 쌍에 대한 점수 벡터가 됩니다. 실제 구현에 맞게 수정 필요.
        # 여기서는 단순히 첫번째 문서와의 유사도로 가정합니다.
        sim_scores = torch.nn.functional.cosine_similarity(scores[0], scores[1:])
        
        sorted_indices = torch.argsort(sim_scores, descending=True)
        
        return [documents[i] for i in sorted_indices[:top_n]]

    def _generate_answer(self, question: str, context_docs: list) -> str:
        """문맥(context)과 질문을 바탕으로 답변을 생성합니다."""
        context = "\n".join(context_docs)
        prompt = f"""[CONTEXT]
{context}

[QUESTION]
{question}

위의 CONTEXT를 바탕으로 QUESTION에 대한 답변을 한국어로 생성하세요.
[ANSWER]
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.generator.generate(**inputs, max_new_tokens=256, num_return_sequences=1)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트를 답변에서 제거
        answer = answer.split("[ANSWER]")[-1].strip()
        
        return answer

    def infer(self, question: str) -> str:
        """
        전체 RAG 추론 파이프라인을 실행합니다.
        1. Retrieve
        2. Rerank
        3. Generate
        """
        # 1. Retrieve
        retrieved_docs = self._retrieve(question)
        
        # 2. Rerank
        reranked_docs = self._rerank(question, retrieved_docs)
        
        # 3. Generate
        answer = self._generate_answer(question, reranked_docs)
        
        return answer
