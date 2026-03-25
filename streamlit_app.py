import streamlit as st
import time
import sys
import os

# 프로젝트 루트를 sys.path에 추가하여 src 모듈을 임포트
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.rag_pipeline import RAGPipeline

# --- App
st.set_page_config(page_title="금융 QA 응답기", layout="wide")
st.title("📘 금융 QA 응답기")
st.write("질문을 입력하면 RAG 파이프라인을 통해 답변을 생성합니다.")

# --- RAG 파이프라인 로드 ---
# @st.cache_resource는 앱 실행 중 리소스를 한 번만 로드하도록 캐싱합니다.
@st.cache_resource
def load_rag_pipeline():
    \"\"\"RAGPipeline을 로드하고 캐시합니다.\"\"\"
    try:
        pipeline = RAGPipeline()
        return pipeline
    except Exception as e:
        st.error(f"파이프라인 로딩 중 에러 발생: {e}")
        st.error("모델 파일이 'models' 디렉터리에 올바르게 다운로드되었는지 확인하세요.")
        st.info("처음 실행 시, 'notebooks/2_inference_setup.ipynb'의 지침에 따라 모델을 다운로드해야 합니다.")
        return None

with st.spinner("RAG 파이프라인을 로드하는 중입니다. 몇 분 정도 소요될 수 있습니다..."):
    pipeline = load_rag_pipeline()

# --- UI
if pipeline is not None:
    # 입력 필드 초기화
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    st.session_state.user_input = st.text_area(
        "질문 입력:", 
        value=st.session_state.user_input, 
        height=150,
        placeholder="여기에 금융 관련 질문을 입력하세요..."
    )

    col1, col2, _ = st.columns([1, 1, 5])
    with col1:
        run_button = st.button("🔍 응답 생성", type="primary")
    with col2:
        clear_button = st.button("🧹 지우기")

    # 응답 생성 버튼 로직
    if run_button and st.session_state.user_input.strip():
        with st.spinner("모델이 답변을 생성 중입니다..."):
            try:
                start_time = time.time()
                result = pipeline.infer(st.session_state.user_input)
                elapsed_time = time.time() - start_time
                
                st.session_state.result = result
                st.session_state.elapsed_time = round(elapsed_time, 2)
            except Exception as e:
                st.error(f"❌ 답변 생성 중 에러 발생: {e}")
    elif run_button:
        st.warning("질문을 입력해주세요.")

    # 지우기 버튼 로직
    if clear_button:
        st.session_state.user_input = ""
        st.session_state.result = ""
        st.session_state.elapsed_time = 0
        st.rerun()

    # 결과 표시
    if "result" in st.session_state and st.session_state.result:
        st.divider()
        st.success("✅ 생성된 답변:")
        st.write(st.session_state.result)
        if "elapsed_time" in st.session_state and st.session_state.elapsed_time > 0:
            st.info(f"⏱ 응답 생성 시간: {st.session_state.elapsed_time}초")
else:
    st.error("파이프라인이 로드되지 않아 앱을 실행할 수 없습니다.")
