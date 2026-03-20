# RAG-based-Financial-Security-LLM
# 🛡️ FinSec-RAG-Pipeline: 금융보안 특화 RAG 기반 질의응답 시스템

## 📌 Project Overview
[cite_start]본 프로젝트는 **'2025 금융 AI challenge: 금융 AI 모델 경쟁'**에 출품하여 **최종 6위 (283팀 중)**를 기록한 모델의 파이프라인입니다[cite: 134, 135, 156]. 
[cite_start]금융권의 AI 활용 한계를 극복하기 위해 SFT(Supervised Fine-Tuning) 대신 **RAG(Retrieval-Augmented Generation) 기반의 아키텍처를 채택**하여, 환각(Hallucination)을 최소화하고 법률/보안 데이터의 최신성을 유지할 수 있는 질의응답 시스템을 구축했습니다[cite: 26, 136, 276].

* [cite_start]**진행 기간:** 2025.08 ~ 2025.09 [cite: 156, 267, 268]
* [cite_start]**팀명 / 인원:** 쓰디쓴커피 / 4명 [cite: 2, 270]
* [cite_start]**담당 역할:** 팀 리더 (데이터 수집, RAG 파이프라인 설계 및 구축) [cite: 3, 135, 274]

## 🎯 Key Challenges & Solutions

### 1. 제한된 컴퓨팅 자원 극복
* [cite_start]**Problem:** RTX 4090 (VRAM 20GB)이라는 매우 제한적인 추론 환경에서 모델을 구동해야 함[cite: 26].
* [cite_start]**Solution:** 모델 훈련(SFT)에 소모되는 시간과 자원을 아끼고자 LLM의 Fine-tuning을 배제하고, 고성능 RAG 파이프라인을 결합하여 성능을 최대치로 끌어올렸습니다[cite: 26, 136].

### 2. 법률 데이터의 환각 억제 및 문서 구축 한계
* [cite_start]**Problem:** 주최 측 제공 자료가 없고, 기존 판독문은 라이선스 제약으로 재가공이 불가함[cite: 276, 298].
* [cite_start]**Solution:** 누구나 열람 가능한 국가법령정보 및 금융보안 관련 법률 텍스트를 직접 수집하여 지식 베이스를 구축했습니다[cite: 277, 299]. [cite_start]청킹(Chunking) 시 각 문단의 머리말에 `[법명] + [위치]` 또는 `[제목] + [본문]`을 부여하여 문서 검색 시 문맥이 보존되도록 설계했습니다[cite: 32, 33, 78].

### 3. 정밀한 검색(Retrieval)을 위한 2-Stage 검색 및 쿼리 증강
* **Problem:** 입력 텍스트가 짧을 경우 핵심 단어가 아닌 부사나 형용사에 매칭되어 검색 품질이 떨어짐.
* [cite_start]**Solution:** * **Query Augmentation:** 원문 질문과 동일한 지식을 묻는 새로운 질문을 LLM으로 생성(증강)하여, 검색 시 원문과 함께 활용해 Recall을 높였습니다[cite: 40, 41, 61, 89].
  * [cite_start]**2-Stage Retrieval:** 1. **1차 검색:** `FAISS` 벡터 검색으로 코사인 유사도 0.5 이상의 Top-K(최대 50개) 후보 문단을 빠르게 추출[cite: 43, 63, 64].
    2. [cite_start]**2차 검색:** `Cross-Encoder` 기반 리랭커(Re-ranker)를 활용해 관련성 점수 0.85 이상인 Top-N(최대 5개)의 핵심 Context만 최종 프롬프트에 주입[cite: 44, 68, 69, 100].

## 🛠️ Tech Stack & Environment
* [cite_start]**Environment:** Python 3.10, RTX 4090 [cite: 26]
* [cite_start]**LLM:** `SKT/A.X-4.0-Light` (한국어 특화) [cite: 26, 115]
* [cite_start]**Embedding / Vector DB:** `dragonkue/snowflake-arctic-embed-l-v2.0-ko` / `FAISS (IndexFlatIP)` [cite: 26, 57, 85, 117]
* [cite_start]**Re-ranker:** `dragonkue/bge-reranker-v2-m3-ko` [cite: 26, 68, 116]

## 💡 Lessons Learned
* [cite_start]**문서화와 협업의 중요성:** 파이프라인이 복잡해지면서 발생한 실험 내용의 휘발 문제를 해결하기 위해, 하이퍼파라미터 변경 이력과 회의 결과를 Notion에 체계적으로 기록하는 습관을 들였습니다[cite: 310, 311, 312].
* [cite_start]**유연한 데이터 파이프라인 설계:** 향후 새로운 데이터가 모이더라도 전체 알고리즘 수정 없이 CSV에 이어 붙이는 형식만으로 모델을 쉽게 발전시킬 수 있는 구조를 완성했습니다[cite: 26].
