import os
import sys

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from huggingface_hub import snapshot_download

def download_models():
    """
    config.py에 정의된 모든 모델을 Hugging Face Hub에서 다운로드합니다.
    """
    model_list = [
        config.EMBEDDING_MODEL_NAME,
        config.GENERATOR_MODEL_NAME,
        config.RERANKER_MODEL_NAME
    ]
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    for model_name in model_list:
        print(f"Downloading model: {model_name}")
        
        # Hugging Face Hub에서의 이름을 로컬 디렉터리 이름으로 변환 ( '/' -> '_' )
        local_model_name = model_name.replace('/', '_')
        save_path = os.path.join(config.MODEL_DIR, local_model_name)
        
        if os.path.exists(save_path):
            print(f"Model already exists at {save_path}. Skipping download.")
            continue
            
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=save_path,
                local_dir_use_symlinks=False, # Windows에서는 False로 설정
                resume_download=True,
            )
            print(f"Successfully downloaded {model_name} to {save_path}")
        except Exception as e:
            print(f"Failed to download {model_name}. Error: {e}")

if __name__ == "__main__":
    download_models()
