from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from typing import Optional, Annotated, TypedDict, Generator
import json
import requests
import time
import sys
import os
import re
import logging

# 설정 및 유틸리티 import
from utils.config import config_manager
from utils.embedding import (
    EmbeddingModel, 
    load_documents_from_directory, 
    create_document_chunks,
    create_document_chunks_parallel
)
from utils.search import create_faiss_vectorstore, load_faiss_vectorstore, get_retriever, similarity_search
from utils.prompt_templates import (
    SUMMARY_PROMPT_TEMPLATE,
    REPORT_PROMPT_TEMPLATE,
    FALLBACK_PROMPT_TEMPLATE
)
from utils.output_format import format_answer, remove_duplicates
from utils.llm import LLMClient
from utils.parallel import ProgressTracker, parallel_process

# 로깅 설정
logger = logging.getLogger(__name__)

# ==========================
# ✅ JSON 전처리 함수 추가
# ==========================
def clean_json_text(text: str) -> str:
    """JSON 파싱 전 제어 문자 제거"""
    # 제어 문자 제거 (줄바꿈, 탭, 캐리지 리턴 등)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # 연속된 공백 정리
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ==========================
# ✅ 설정 기반 상수
# ==========================
def get_config():
    """설정 가져오기"""
    return config_manager

# 기본 설정 (설정 파일이 없을 때 사용)
DEFAULT_LLM_CONFIG = {
    "model": "mistral",
    "temperature": 0.1,
    "num_predict": 3000,
    "top_k": 3,
    "top_p": 0.9,
    "timeout": 900
}

# ==========================
# ✅ 인덱스 자동 생성/로드 함수 (병렬 처리 및 진행률 표시 강화)
# ==========================
def get_or_build_embedding_index(use_parallel: bool = True, max_workers: int = 4):
    """
    임베딩 인덱스 생성 또는 로드 (병렬 처리 및 진행률 표시 강화)
    
    Args:
        use_parallel: 병렬 처리 사용 여부
        max_workers: 최대 워커 수
    """
    try:
        paths_config = config_manager.get_paths_config()
        docs_dir = paths_config.get("docs_dir", "./docs/embedding/")
        index_path = paths_config.get("index_path", "./faiss_index")
        
        faiss_path = f"{index_path}.faiss"
        pkl_path = f"{index_path}.pkl"
        
        embedding_config = config_manager.get_embedding_config()
        embedding_model = EmbeddingModel()
        
        # 기존 인덱스 로드 시도
        if os.path.exists(faiss_path) and os.path.exists(pkl_path):
            logger.info(f"기존 임베딩 인덱스를 로드합니다: {index_path}")
            try:
                vectorstore = load_faiss_vectorstore(index_path, embedding_model)
                retriever = get_retriever(vectorstore)
                logger.info("임베딩 인덱스 로드 완료")
                return retriever
            except Exception as e:
                logger.error(f"기존 인덱스 로드 실패: {e}")
                logger.info("손상된 인덱스를 재생성합니다.")
        
        # 새 인덱스 생성
        logger.info(f"임베딩 인덱스를 생성합니다: {docs_dir}")
        if not os.path.exists(docs_dir):
            logger.error(f"문서 디렉토리가 없습니다: {docs_dir}")
            return None
        
        # 문서 로드 (병렬 처리 지원)
        documents = load_documents_from_directory(
            docs_dir, 
            use_parallel=use_parallel, 
            max_workers=max_workers
        )
        if not documents:
            logger.error(f"문서를 찾을 수 없습니다: {docs_dir}")
            return None
        
        # 청크 생성 (병렬 처리 지원)
        all_chunks, all_metadata = create_document_chunks_parallel(
            documents,
            use_parallel=use_parallel,
            max_workers=max_workers
        )
        
        if not all_chunks:
            logger.error("문서 청크가 생성되지 않았습니다.")
            return None
        
        # 문서 임베딩 및 벡터스토어 생성/저장
        logger.info(f"벡터스토어를 생성합니다: {len(all_chunks)}개 청크")
        vectorstore = create_faiss_vectorstore(all_chunks, all_metadata, embedding_model, index_path)
        retriever = get_retriever(vectorstore)
        logger.info(f"임베딩 인덱스 생성 및 저장 완료: {index_path}")
        return retriever
        
    except Exception as e:
        logger.error(f"임베딩 인덱스 처리 중 오류: {e}")
        return None

# ==========================
# ✅ 1. 보고서 & 요약 스키마 정의
# ==========================
class TechReport(BaseModel):
    title: str = Field(description="보고서 제목")
    abstract: str = Field(description="요약")
    introduction: str = Field(description="서론")
    background: str = Field(description="배경")
    main_content: str = Field(description="본문 내용")
    conclusion: str = Field(description="결론")

class Summary(BaseModel):
    summary: str = Field(description="간략 요약")

# ==========================
# ✅ 2. LangGraph 상태 정의 (스트리밍 지원)
# ==========================
class ReportState(TypedDict):
    # 입력 데이터
    source_text: str
    question: str
    top_k: int
    
    # 프롬프트 및 응답
    prompt: str
    response: str
    parsed: dict
    
    # 상태 관리
    error: str
    retry_count: int
    mode: str
    
    # 검색 관련
    search_results: list
    search_success: bool
    
    # LLM 설정 (선택적)
    llm_config: Optional[dict]
    
    # 스트리밍 지원
    streaming_enabled: bool

# ==========================
# ✅ 3. 임베딩 검색 함수
# ==========================
def highlight_keywords(text: str, keywords: list) -> str:
    # CLI에서 키워드를 노란색(ANSI)으로 하이라이트
    def ansi_highlight(match):
        return f"\033[1;33m{match.group(0)}\033[0m"
    for kw in sorted(set(keywords), key=len, reverse=True):
        if kw.strip():
            # 단어 경계 기준으로만 하이라이트
            text = re.sub(rf'(?i)\b{re.escape(kw)}\b', ansi_highlight, text)
    return text

def extract_keywords(question: str) -> list:
    # 간단하게 띄어쓰기 기준 분리 + 길이 2 이상만 (심화: konlpy 등 형태소 분석 가능)
    words = re.findall(r'\w+', question)
    return [w for w in words if len(w) > 1]

def search_relevant_documents(question: str, retriever, k: Optional[int] = None, max_chars: Optional[int] = None):
    """질문과 관련된 문서를 검색하여 반환 (예외 처리 강화)"""
    try:
        search_config = config_manager.get_search_config()
        k = int(k) if k is not None else int(search_config.get("default_top_k", 5))
        max_chars = int(max_chars) if max_chars is not None else int(config_manager.get("embedding.max_chars", 15000))
        
        if retriever is None:
            logger.error("임베딩 인덱스가 없습니다.")
            return "검색 결과가 없습니다. (임베딩 인덱스 없음)", []
        
        results = similarity_search(question, retriever, k=k)
        if not results:
            logger.warning("검색 결과가 없습니다.")
            return "검색 결과가 없습니다.\n(문서가 없거나, 질문과 관련된 내용을 찾지 못했습니다. 다른 질문을 입력해 주세요.)", []
        
        relevant_docs = []
        total_chars = 0
        sources = []
        keywords = extract_keywords(question)
        max_doc_chars = search_config.get("max_doc_chars", 3500)
        
        for rank, doc in enumerate(results[:k], 1):
            similarity = "-"
            content = doc.page_content
            if len(content) > max_doc_chars:
                content = content[:max_doc_chars] + f"\n[문서가 길어서 잘렸습니다... (유사도: {similarity})]"
            
            doc_highlighted = highlight_keywords(content, keywords)
            doc_text = f"[{rank}] 유사도: {similarity}\n{doc_highlighted}"
            
            if total_chars + len(doc_text) > max_chars:
                remaining_chars = max_chars - total_chars
                if remaining_chars > 200:
                    doc_text = doc_text[:remaining_chars] + f"\n[문서가 길어서 잘렸습니다... (유사도: {similarity})]"
                    relevant_docs.append(doc_text)
                    sources.append(doc.metadata.get('filename', doc.metadata.get('filepath', 'unknown')))
                break
            
            relevant_docs.append(doc_text)
            sources.append(doc.metadata.get('filename', doc.metadata.get('filepath', 'unknown')))
            total_chars += len(doc_text)
        
        result = "\n\n".join(relevant_docs)
        logger.info(f"검색 완료: {len(relevant_docs)}개 문서, {len(result)}자")
        return result, sources
        
    except Exception as e:
        logger.error(f"문서 검색 중 오류: {e}")
        return f"검색 중 오류가 발생했습니다. (에러: {e})", []
    if retriever is None:
        return "검색 결과가 없습니다. (임베딩 인덱스 없음)", []
    try:
        results = similarity_search(question, retriever, k=k)
        if not results:
            return "검색 결과가 없습니다.\n(문서가 없거나, 질문과 관련된 내용을 찾지 못했습니다. 다른 질문을 입력해 주세요.)", []
        # 디버그 정보는 주석 처리 (필요시 주석 해제)
        # print("[DEBUG] 검색된 문서 개수:", len(results))
        # for i, doc in enumerate(results, 1):
        #     content = doc.page_content
        #     print(f"[DEBUG] 문서 {i} 길이: {len(content)} | 앞부분: {content[:80]}")
        relevant_docs = []
        total_chars = 0
        sources = []
        keywords = extract_keywords(question)
        for rank, doc in enumerate(results[:5], 1):  # 상위 5개 사용
            similarity = "-"
            max_doc_chars = 3500  # 각 문서 3500자까지
            content = doc.page_content
            if len(content) > max_doc_chars:
                content = content[:max_doc_chars] + f"\n[문서가 길어서 잘렸습니다... (유사도: {similarity})]"
            doc_highlighted = highlight_keywords(content, keywords)
            doc_text = f"[{rank}] 유사도: {similarity}\n{doc_highlighted}"
            if total_chars + len(doc_text) > max_chars:
                remaining_chars = max_chars - total_chars
                if remaining_chars > 200:
                    doc_text = doc_text[:remaining_chars] + f"\n[문서가 길어서 잘렸습니다... (유사도: {similarity})]"
                    relevant_docs.append(doc_text)
                    sources.append(doc.metadata.get('filename', doc.metadata.get('filepath', 'unknown')))
                break
            relevant_docs.append(doc_text)
            sources.append(doc.metadata.get('filename', doc.metadata.get('filepath', 'unknown')))
            total_chars += len(doc_text)
        result = "\n\n".join(relevant_docs)
        # print(f"📄 검색된 문서 크기: {len(result)}자 (제한: {max_chars}자)")
        # print(f"🔍 코사인 유사도 기반 문서 선택: {len(relevant_docs)}개 문서")
        return result, sources
    except Exception as e:
        print(f"문서 검색 중 오류: {e}")
        return f"검색 중 오류가 발생했습니다. (에러: {e})", []

# ==========================
# ✅ 4. LangGraph 노드 정의
# ==========================
def generate_prompt(state: ReportState) -> ReportState:
    new_state = {
        "source_text": state["source_text"],
        "question": state["question"],
        "top_k": state["top_k"],
        "prompt": "",
        "response": state.get("response", ""),
        "parsed": state.get("parsed", {}),
        "error": state.get("error", ""),
        "retry_count": state.get("retry_count", 0),
        "mode": state["mode"],
        "search_results": state.get("search_results", []),
        "search_success": state.get("search_success", True)
    }
    if "llm_config" in state and state["llm_config"] is not None:
        new_state["llm_config"] = state["llm_config"]
    # 유연한 프롬프트
    new_state["prompt"] = f"""
아래 문서 내용을 바탕으로 기술 보고서를 작성해주세요.

🚨 필수 지시사항:
- 반드시 한국어로만 답변하세요
- JSON 형식으로만 출력하세요 (JSON 외 텍스트 출력 금지)
- 모든 필드는 한국어로 작성하세요

📝 작성 가이드:
- 문서의 핵심 정보를 유지하되, 창의적으로 재구성하세요
- 각 항목은 고유한 관점과 목적을 가져야 합니다
- 구체적이고 실용적인 정보를 포함하세요
- 각 필드는 최소 길이를 반드시 지켜주세요 (abstract: 500자, introduction: 500자, background: 500자, main_content: 800자, conclusion: 500자)
- 예시, 사례, 구체적 수치, 실무 팁 등을 포함하여 풍부하게 작성하세요

⚠️ 엄격한 규칙:
- JSON 외 텍스트는 출력하지 마세요
- 원본 문서의 문장을 그대로 사용하지 마세요
- "중요합니다", "복잡합니다" 같은 모호한 표현 대신 구체적이고 창의적인 설명을 사용하세요
- 모든 내용은 자신만의 언어로 재구성하여 작성하세요

입력 문서:
========================
{state['source_text']}
========================
"""
    # print("[DEBUG] LLM 프롬프트 전체:\n" + new_state["prompt"][:2000] + ("..." if len(new_state["prompt"]) > 2000 else ""))
    return ReportState(**new_state)

# call_model의 num_predict 제한 원복 (DEFAULT_LLM_CONFIG 사용) + 스트리밍 지원
def call_model(state: ReportState) -> ReportState:
    from utils.llm import LLMClient
    new_state = {
        "source_text": state["source_text"],
        "question": state["question"],
        "top_k": state["top_k"],
        "prompt": state["prompt"],
        "response": "",
        "parsed": state.get("parsed", {}),
        "error": "",
        "retry_count": state.get("retry_count", 0),
        "mode": state["mode"],
        "search_results": state.get("search_results", []),
        "search_success": state.get("search_success", True),
        "streaming_enabled": state.get("streaming_enabled", False)
    }
    if "llm_config" in state and state["llm_config"] is not None:
        new_state["llm_config"] = state["llm_config"]
    llm_config = DEFAULT_LLM_CONFIG.copy()
    if "llm_config" in state and state["llm_config"] is not None:
        llm_config.update(state["llm_config"])
    timeout = max(llm_config["timeout"], 60)
    try:
        logger.info(f"LLM 호출 시작: {llm_config['model']}")
        llm = LLMClient(model_name=llm_config["model"])
        
        # 스트리밍 모드 확인
        if state.get("streaming_enabled", False):
            logger.info("스트리밍 모드로 LLM 호출")
            # 스트리밍 응답을 수집
            full_response = ""
            for chunk in llm.stream_generate(
                state["prompt"], 
                num_predict=llm_config["num_predict"], 
                temperature=llm_config["temperature"]
            ):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()  # 줄바꿈
            new_state["response"] = full_response
        else:
            logger.info("일반 모드로 LLM 호출")
            response = llm.generate(
                state["prompt"], 
                num_predict=llm_config["num_predict"], 
                temperature=llm_config["temperature"]
            )
            new_state["response"] = response
        
        new_state["error"] = ""
        logger.info("LLM 응답 수신 완료")
    except Exception as e:
        error_msg = f"LLM 호출 중 오류: {str(e)}"
        logger.error(error_msg)
        new_state["error"] = error_msg
    return ReportState(**new_state)


def parse_response(state: ReportState) -> ReportState:
    from utils.llm import LLMClient
    new_state = {
        "source_text": state["source_text"],
        "question": state["question"],
        "top_k": state["top_k"],
        "prompt": state["prompt"],
        "response": state["response"],
        "parsed": {},
        "error": "",
        "retry_count": state.get("retry_count", 0),
        "mode": state["mode"],
        "search_results": state.get("search_results", []),
        "search_success": state.get("search_success", True)
    }
    if "llm_config" in state and state["llm_config"] is not None:
        new_state["llm_config"] = state["llm_config"]
    if state["error"]:
        new_state["error"] = state["error"]
        print(f"⚠️ 이전 단계에서 오류 발생: {state['error']}")
        return ReportState(**new_state)
    if not state["response"]:
        new_state["error"] = "LLM 응답이 비어있습니다."
        print("❌ LLM 응답이 비어있습니다.")
        return ReportState(**new_state)
    try:
        print("🔄 응답 파싱 중...")
        # print(f"[DEBUG] 원본 응답 길이: {len(state['response'])}")
        # print(f"[DEBUG] 원본 응답 앞부분: {state['response'][:500]}...")
        
        llm = LLMClient(model_name=DEFAULT_LLM_CONFIG["model"])
        result = llm.parse_response(state["response"])
        
        # print(f"[DEBUG] 파싱 결과 타입: {type(result)}")
        
        if isinstance(result, dict):
            # print(f"[DEBUG] 파싱된 딕셔너리 키: {list(result.keys())}")
            # for k, v in result.items():
            #     print(f"[DEBUG] 파싱된 항목: {k} | 길이: {len(str(v))} | 값: {str(v)[:100]}")
            
            # 필수 필드 확인 (오류가 있을 때만 출력)
            required_fields = ["title", "abstract", "introduction", "background", "main_content", "conclusion"]
            missing_fields = [field for field in required_fields if field not in result or not result[field]]
            if missing_fields:
                print(f"⚠️ 누락된 필드: {missing_fields}")
        # else:
        #     print(f"[DEBUG] 파싱 결과가 딕셔너리가 아님: {result}")
        
        new_state["parsed"] = result
        new_state["error"] = ""
        print("✅ 응답 파싱 완료")
    except Exception as e:
        error_msg = f"응답 파싱 실패: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"📝 원본 응답: {state['response'][:500]}...")
        new_state["parsed"] = {
            "raw_response": state["response"],
            "error": error_msg
        }
        new_state["error"] = error_msg
    return ReportState(**new_state)

def fix_prompt_and_retry(state: ReportState) -> ReportState:
    """프롬프트 수정 및 재시도 노드 - source_text를 변경하지 않음"""
    # 상태를 복사하되 source_text는 그대로 유지
    new_state = {
        "source_text": state["source_text"],  # 명시적으로 유지
        "question": state["question"],
        "top_k": state["top_k"],
        "prompt": "",
        "response": state["response"],
        "parsed": state.get("parsed", {}),
        "error": state.get("error", ""),
        "retry_count": state.get("retry_count", 0) + 1,
        "mode": state["mode"],
        "search_results": state.get("search_results", []),
        "search_success": state.get("search_success", True)
    }
    
    # LLM 설정이 있으면 추가 (재시도 시 토큰 수 줄임)
    if "llm_config" in state and state["llm_config"] is not None:
        new_state["llm_config"] = state["llm_config"].copy()
        # 재시도 시 토큰 수를 줄여서 타임아웃 방지
        new_state["llm_config"]["num_predict"] = min(2000, state["llm_config"].get("num_predict", 4000))
    else:
        new_state["llm_config"] = DEFAULT_LLM_CONFIG.copy()
        new_state["llm_config"]["num_predict"] = 2000  # 재시도용으로 토큰 수 줄임
    
    print(f"🔄 재시도 {new_state['retry_count']}회 - 프롬프트 수정 중...")
            # print(f"📝 재시도 설정: 토큰 수 {new_state['llm_config']['num_predict']}")
    
    original_response = state["response"]
    
    if state["mode"] == "summary":
        new_state["prompt"] = f"""
이전 응답이 올바른 형식이 아니었습니다. 다음 문서를 기반으로 요약을 다시 작성해주세요.

문서:
{state['source_text'][:2000]}  # 문서 길이 제한

질문:
{state['question']}

요구사항:
- JSON 형식으로 출력
- summary 필드에 요약 내용 포함
- 한국어로 작성
- 300자 이상

예시 형식:
{{"summary": "요약 내용"}}
"""
    else:
        new_state["prompt"] = f"""
이전 응답이 올바른 형식이 아니었습니다. 다음 문서를 기반으로 보고서를 다시 작성해주세요.

🚨 필수 지시사항:
- 반드시 한국어로만 답변하세요
- JSON 형식으로만 출력하세요 (JSON 외 텍스트 출력 금지)
- 모든 필드는 한국어로 작성하세요

문서:
{state['source_text'][:2000]}  # 문서 길이 제한

질문:
{state['question']}

요구사항:
- JSON 형식으로만 출력 (JSON 외 텍스트 금지)
- 다음 필드만 포함: title, abstract, introduction, background, main_content, conclusion
- sections 배열이나 다른 필드는 사용하지 마세요
- 모든 필드는 한국어로 작성
- 각 필드는 최소 길이를 반드시 지켜주세요 (abstract: 500자, introduction: 500자, background: 500자, main_content: 800자, conclusion: 500자)
- 예시, 사례, 구체적 수치, 실무 팁 등을 포함하여 풍부하게 작성하세요

⚠️ 엄격한 규칙:
- JSON 외 텍스트는 출력하지 마세요
- 원본 문서의 문장을 그대로 사용하지 마세요
- "중요합니다", "복잡합니다" 같은 모호한 표현 대신 구체적이고 창의적인 설명을 사용하세요
- 모든 내용은 자신만의 언어로 재구성하여 작성하세요
"""
    
    # print("✅ 프롬프트 수정 완료 (문서 길이 제한 적용)")
    return ReportState(**new_state)

# ==========================
# ✅ 5. LangGraph 구성 (수정)
# ==========================
builder = StateGraph(ReportState)
builder.add_node("generate_prompt", RunnableLambda(generate_prompt))
builder.add_node("call_model", RunnableLambda(call_model))
builder.add_node("parse_response", RunnableLambda(parse_response))
builder.add_node("fix_prompt_and_retry", RunnableLambda(fix_prompt_and_retry))

builder.set_entry_point("generate_prompt")
builder.add_edge("generate_prompt", "call_model")
builder.add_edge("call_model", "parse_response")
builder.add_conditional_edges(
    "parse_response",
    lambda state: END if not state["error"] or (state.get("retry_count") or 0) >= 1 else "fix_prompt_and_retry"
)
builder.add_edge("fix_prompt_and_retry", "call_model")

graph = builder.compile()

# ==========================
# ✅ 6. 유틸 함수
# ==========================
def determine_mode(question: str) -> str:
    summary_keywords = ["요약", "간단히", "한줄", "짧게"]
    report_keywords = ["보고서", "작성해줘", "정리해줘", "구조화", "항목별", "전문적으로", "분석해줘", "서술형"]

    if any(kw in question for kw in summary_keywords):
        return "summary"
    if any(kw in question for kw in report_keywords):
        return "report"
    return "free"

def regenerate_field(field: str, source_text: str, question: str, llm_config: Optional[dict] = None) -> str:
    """특정 필드 재생성"""
    print(f"🔁 {field} 항목 길이 부족 → 자동 재작성")
    
    # LLM 설정 가져오기
    config = DEFAULT_LLM_CONFIG.copy()
    if llm_config:
        config.update(llm_config)
    
    # 토큰 수를 줄여서 타임아웃 방지
    config["num_predict"] = min(1500, config.get("num_predict", 2000))
    
    try:
        prompt = f"""
다음 문서를 기반으로 {field} 항목을 다시 작성해주세요.

문서:
{source_text[:3000]}

질문:
{question}

요구사항:
- JSON 형식으로 출력
- {field} 필드에 내용 포함
- 한국어로 작성
- 300자 이상
- 제어 문자(줄바꿈, 탭 등) 사용 금지

예시 형식:
{{"{field}": "내용"}}
"""
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "model": config["model"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config["temperature"],
                    "num_predict": config["num_predict"],
                    "top_k": config["top_k"],
                    "top_p": config["top_p"]
                }
            }),
            timeout=config["timeout"]
        )
        
        if response.status_code == 200:
            resp_json = response.json()
            if "response" in resp_json:
                # JSON 전처리 적용
                cleaned_response = clean_json_text(resp_json["response"])
                
                # JSON 파싱 시도
                try:
                    parsed = json.loads(cleaned_response)
                    if field in parsed:
                        return parsed[field]
                except json.JSONDecodeError as e:
                    print(f"필드 재생성 중 오류: {e}")
                    # 파싱 실패 시 원본 텍스트 반환
                    return resp_json["response"]
        
        return f"{field} 재생성 실패"
        
    except Exception as e:
        print(f"필드 재생성 중 오류: {e}")
        return f"{field} 재생성 중 오류 발생"

# ==========================
# ✅ 6. LLM 설정 구성 함수
# ==========================
def configure_llm_settings(current_config: dict) -> dict:
    """LLM 설정을 대화형으로 구성하는 함수"""
    print("\n⚙️ LLM 설정 변경")
    print("=" * 40)
    
    config = current_config.copy()
    
    # 모델 선택
    print(f"현재 모델: {config['model']}")
    new_model = input("새 모델명 (Enter로 유지): ").strip()
    if new_model:
        config['model'] = new_model
    
    # Temperature
    print(f"현재 temperature: {config['temperature']}")
    try:
        new_temp = input("새 temperature (0.0-1.0, Enter로 유지): ").strip()
        if new_temp:
            temp_val = float(new_temp)
            if 0.0 <= temp_val <= 1.0:
                config['temperature'] = temp_val
            else:
                print("❌ temperature는 0.0-1.0 사이여야 합니다.")
    except ValueError:
        print("❌ 유효한 숫자를 입력하세요.")
    
    # num_predict
    print(f"현재 num_predict: {config['num_predict']}")
    try:
        new_predict = input("새 num_predict (토큰 수, Enter로 유지): ").strip()
        if new_predict:
            predict_val = int(new_predict)
            if predict_val > 0:
                config['num_predict'] = predict_val
            else:
                print("❌ num_predict는 양수여야 합니다.")
    except ValueError:
        print("❌ 유효한 숫자를 입력하세요.")
    
    # top_k
    print(f"현재 top_k: {config['top_k']}")
    try:
        new_top_k = input("새 top_k (Enter로 유지): ").strip()
        if new_top_k:
            top_k_val = int(new_top_k)
            if top_k_val > 0:
                config['top_k'] = top_k_val
            else:
                print("❌ top_k는 양수여야 합니다.")
    except ValueError:
        print("❌ 유효한 숫자를 입력하세요.")
    
    # top_p
    print(f"현재 top_p: {config['top_p']}")
    try:
        new_top_p = input("새 top_p (0.0-1.0, Enter로 유지): ").strip()
        if new_top_p:
            top_p_val = float(new_top_p)
            if 0.0 <= top_p_val <= 1.0:
                config['top_p'] = top_p_val
            else:
                print("❌ top_p는 0.0-1.0 사이여야 합니다.")
    except ValueError:
        print("❌ 유효한 숫자를 입력하세요.")
    
    # timeout
    print(f"현재 timeout: {config['timeout']}초")
    try:
        new_timeout = input("새 timeout (초, Enter로 유지): ").strip()
        if new_timeout:
            timeout_val = int(new_timeout)
            if timeout_val > 0:
                config['timeout'] = timeout_val
            else:
                print("❌ timeout은 양수여야 합니다.")
    except ValueError:
        print("❌ 유효한 숫자를 입력하세요.")
    
    print(f"\n✅ 설정이 업데이트되었습니다:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return config

# ==========================
# ✅ 7. 실행 함수
# ==========================
def main(question: str, embedding_index, top_k: int = 5, llm_config: Optional[dict] = None, streaming: bool = False, use_parallel: bool = True, max_workers: int = 4, split_report: bool = True):
    """
    메인 실행 함수 (병렬 처리 지원)
    split_report: True면 각 항목별로 분할 생성 및 출력, False면 기존 전체 생성
    """
    mode = determine_mode(question)
    print(f"\n📌 감지된 모드: {mode}\n")

    if mode == "free":
        print("💬 일반 질문으로 인식 → LLM 직접 답변\n")
        
        # 문서 검색 시도 (선택적)
        relevant_docs = ""
        if embedding_index is not None:
            relevant_docs = search_relevant_documents(question, embedding_index, k=2, max_chars=8000)
        
        try:
            # LLM 설정 가져오기
            free_llm_config = DEFAULT_LLM_CONFIG.copy()
            if llm_config is not None:
                free_llm_config.update(llm_config)
            free_llm_config["num_predict"] = 2000  # 자유질문용으로 토큰 수 조정
            
            # 프롬프트 구성 (문서가 있으면 참고, 없으면 일반 답변)
            if relevant_docs:
                prompt = f"""다음 문서를 참고하여 질문에 답변해주세요. 문서에 관련 내용이 없으면 일반적인 지식으로 답변하세요.

문서:
{relevant_docs}

질문: {question}

답변:"""
            else:
                prompt = f"""질문에 대해 정확하고 도움이 되는 답변을 해주세요.

질문: {question}

답변:"""
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "model": free_llm_config["model"],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": free_llm_config["temperature"],
                        "num_predict": free_llm_config["num_predict"],
                        "top_k": free_llm_config["top_k"],
                        "top_p": free_llm_config["top_p"]
                    }
                }),
                timeout=free_llm_config["timeout"]
            )
            if response.status_code == 200:
                resp_json = response.json()
                if "response" in resp_json:
                    print(resp_json["response"])
                else:
                    print("❌ LLM 응답 형식 오류")
            else:
                print("❌ LLM API 오류")
        except Exception as e:
            print(f"❌ LLM 호출 중 오류: {e}")
        return

    # 임베딩 검색으로 관련 문서 찾기 (문서 크기 제한 적용)
    if embedding_index is None:
        print("❌ 임베딩 인덱스가 초기화되지 않았습니다.")
        return
        
    source_text, sources = search_relevant_documents(question, embedding_index, k=top_k, max_chars=15000)
    search_success = bool(source_text)
    
    if not source_text:
        print("❌ 관련 문서를 찾을 수 없습니다.")
        return

    if mode == "report" and split_report:
        print("\n🪄 보고서 각 항목을 분할로 생성합니다.\n")
        fields = [
            ("abstract", "요약"),
            ("introduction", "서론"),
            ("background", "배경"),
            ("main_content", "본문"),
            ("conclusion", "결론")
        ]
        parsed = {}
        for field, label in fields:
            print(f"\n⏳ [{label}] 생성 중...")
            val = regenerate_field(field, source_text, question, llm_config)
            parsed[field] = val if val else "내용 없음"
            print(f"\n✅ [{label}]\n{val if val else '내용 없음'}\n")
        # 제목은 요약에서 추출하거나, 별도 생성(간단화)
        parsed["title"] = question.strip()[:40] + ("..." if len(question.strip()) > 40 else "")
        from utils.output_format import format_report_output
        print("\n📝 전체 보고서(모아보기):\n")
        print(format_report_output(parsed, mode, sources))
        # 피드백 및 후속 루프는 기존과 동일하게 유지
        def save_feedback_log(question: str, answer: str, sources: list, feedback: str):
            import datetime
            with open('feedback_log.txt', 'a', encoding='utf-8') as f:
                f.write(f"=== {datetime.datetime.now()} ===\n")
                f.write(f"질문: {question}\n")
                f.write(f"출처: {', '.join(sources)}\n")
                f.write(f"피드백: {feedback}\n")
                f.write(f"답변:\n{answer}\n\n")
        try:
            feedback = input("이 답변이 도움이 되었나요? (y/n): ").strip().lower()
            if feedback in ('y', 'n'):
                from utils.output_format import format_report_output
                answer_text = format_report_output(parsed, mode, sources)
                save_feedback_log(question, answer_text, sources, feedback)
                print("피드백이 저장되었습니다. 감사합니다!")
            followup = input("보고서에 대해 추가로 수정하거나 궁금한 점이 있으면 입력하세요 (엔터시 건너뜀): ").strip()
            if followup:
                modification_keywords = ["수정", "바꿔", "고쳐", "다시", "재작성", "변경", "조정", "개선"]
                is_modification_request = any(keyword in followup for keyword in modification_keywords)
                if is_modification_request and mode == "report" and parsed:
                    print("🔄 보고서 수정 요청 감지 - JSON 형식으로 재생성합니다...")
                    modification_prompt = f"""
기존 보고서를 사용자의 수정 요청에 따라 다시 작성해주세요.

[기존 보고서]
{answer_text}

[사용자 수정 요청]
{followup}

요구사항:
- JSON 형식으로만 출력 (JSON 외 텍스트 금지)
- 다음 필드만 포함: title, abstract, introduction, background, main_content, conclusion
- sections 배열이나 다른 필드는 사용하지 마세요
- 모든 필드는 한국어로 작성
- 각 필드는 300자 이상
- 사용자의 수정 요청을 반영하여 내용을 개선하세요

⚠️ 엄격한 규칙:
- JSON 외 텍스트는 출력하지 마세요
- 원본 문서의 문장을 그대로 사용하지 마세요
- 사용자의 수정 요청을 반드시 반영하세요
- 모든 내용은 자신만의 언어로 재구성하여 작성하세요
"""
                    from utils.llm import LLMClient
                    llm = LLMClient(model_name=DEFAULT_LLM_CONFIG["model"])
                    modified_response = llm.generate(modification_prompt, num_predict=3000, temperature=DEFAULT_LLM_CONFIG["temperature"])
                    modified_parsed = llm.parse_response(modified_response)
                    if isinstance(modified_parsed, dict) and not modified_parsed.get("error"):
                        print("\n===== 수정된 보고서 =====")
                        from utils.output_format import format_report_output
                        print(format_report_output(modified_parsed, mode, sources))
                        parsed.update(modified_parsed)
                    else:
                        print("\n===== 수정 요청 답변 =====")
                        print(modified_response)
                else:
                    from utils.llm import LLMClient
                    llm = LLMClient(model_name=DEFAULT_LLM_CONFIG["model"])
                    followup_prompt = f"""
아래는 기존 보고서입니다. 사용자의 추가 요청(수정/질문 등)에 따라 적절히 답변하거나, 해당 부분을 수정해 주세요.

[기존 보고서]
{answer_text}

[사용자 추가 요청]
{followup}

반드시 한국어로만 답변하세요.
"""
                    followup_answer = llm.generate(followup_prompt, num_predict=1500, temperature=DEFAULT_LLM_CONFIG["temperature"])
                    print("\n===== 추가 요청/수정/답변 결과 =====\n")
                    print(followup_answer)
        except Exception as e:
            print(f"피드백 저장 중 오류: {e}")
        return

    # 기존 전체 생성 방식 (split_report=False)
    # 기본 상태 생성
    state_dict = {
        "source_text": source_text,
        "question": question,
        "top_k": top_k,
        "prompt": "",
        "response": "",
        "parsed": {},
        "error": "",
        "retry_count": 0,
        "mode": mode,
        "search_results": [],
        "search_success": search_success,
        "streaming_enabled": streaming
    }
    
    # LLM 설정이 제공된 경우 상태에 추가
    if llm_config:
        state_dict["llm_config"] = llm_config
    
    initial_state = ReportState(**state_dict)
    final_state = graph.invoke(initial_state)

    # 최종 상태에서 결과 추출 (LangGraph 결과는 직접 final_state)
    source_text = final_state.get("source_text", "")
    parsed = final_state.get("parsed", {})
    error = final_state.get("error", "")
    # 파싱 결과가 dict가 아니면 dict로 감싸기
    if not isinstance(parsed, dict):
        parsed = {"raw_response": str(parsed)}
    
    # 결과 출력
    if parsed and not final_state["error"]:
        # 보고서 모드일 때 각 필드 길이 검사 및 자동 보완
        if mode == "report":
            min_lengths = {
                "abstract": 300,
                "introduction": 300,
                "background": 300,
                "main_content": 300,
                "conclusion": 300
            }
            updated = False
            for field, min_len in min_lengths.items():
                val = parsed.get(field, "")
                if not val or len(str(val).strip()) < min_len:
                    print(f"[자동 보완] {field} 항목이 {min_len}자 미만입니다. LLM으로 재생성합니다.")
                    new_val = regenerate_field(field, source_text, question, llm_config)
                    if new_val and len(str(new_val).strip()) >= min_len:
                        parsed[field] = new_val
                        updated = True
            if updated:
                print("[자동 보완] 일부 항목이 보완되었습니다.")
        from utils.output_format import format_report_output
        print(format_report_output(parsed, mode, sources))
    elif parsed and "raw_response" in parsed:
        # 파싱 실패했지만 원본 텍스트가 있는 경우
        raw_response = remove_duplicates(parsed["raw_response"])
        print(f"""
{'='*60}
⚠️ JSON 파싱 실패 - 원본 응답 출력
{'='*60}

{raw_response}

{'='*60}""")
    elif error:
        print(f"\n❌ 오류 발생: {error}")
        # 오류가 있지만 원본 응답이 있는 경우 출력
        if "response" in final_state and final_state["response"]:
            raw_response = remove_duplicates(final_state["response"])
            print(f"""
{'='*60}
📝 LLM 원본 응답
{'='*60}

{raw_response}

{'='*60}""")
        if source_text:
            print(f"""
📄 검색된 문서 내용
{'-' * 30}
{source_text[:500]}{'...' if len(source_text) > 500 else ''}""")
    else:
        print(f"\n❌ 예상치 못한 오류 발생")
        if source_text:
            print(f"""
📄 검색된 문서 내용
{'-' * 30}
{source_text[:500]}{'...' if len(source_text) > 500 else ''}""")

    # 결과 출력 이후에 피드백 저장 함수 정의 및 호출
    def save_feedback_log(question: str, answer: str, sources: list, feedback: str):
        import datetime
        with open('feedback_log.txt', 'a', encoding='utf-8') as f:
            f.write(f"=== {datetime.datetime.now()} ===\n")
            f.write(f"질문: {question}\n")
            f.write(f"출처: {', '.join(sources)}\n")
            f.write(f"피드백: {feedback}\n")
            f.write(f"답변:\n{answer}\n\n")
    # 피드백 루프
    try:
        feedback = input("이 답변이 도움이 되었나요? (y/n): ").strip().lower()
        if feedback in ('y', 'n'):
            # 답변 텍스트 추출
            answer_text = ''
            if parsed and not final_state["error"]:
                from utils.output_format import format_report_output
                answer_text = format_report_output(parsed, mode, sources)
            elif parsed and "raw_response" in parsed:
                answer_text = parsed["raw_response"]
            elif final_state.get("response"):
                answer_text = final_state["response"]
            else:
                answer_text = str(parsed)
            save_feedback_log(question, answer_text, sources, feedback)
            print("피드백이 저장되었습니다. 감사합니다!")
        # 추가 피드백/수정/질문 루프
        followup = input("보고서에 대해 추가로 수정하거나 궁금한 점이 있으면 입력하세요 (엔터시 건너뜀): ").strip()
        if followup:
            # 수정 요청인지 확인 (특정 키워드로 판단)
            modification_keywords = ["수정", "바꿔", "고쳐", "다시", "재작성", "변경", "조정", "개선"]
            is_modification_request = any(keyword in followup for keyword in modification_keywords)
            
            if is_modification_request and mode == "report" and parsed and not final_state["error"]:
                # JSON 형식으로 보고서 재생성
                print("🔄 보고서 수정 요청 감지 - JSON 형식으로 재생성합니다...")
                modification_prompt = f"""
기존 보고서를 사용자의 수정 요청에 따라 다시 작성해주세요.

[기존 보고서]
{answer_text}

[사용자 수정 요청]
{followup}

요구사항:
- JSON 형식으로만 출력 (JSON 외 텍스트 금지)
- 다음 필드만 포함: title, abstract, introduction, background, main_content, conclusion
- 모든 필드는 한국어로 작성
- 각 필드는 300자 이상
- 사용자의 수정 요청을 반영하여 내용을 개선하세요

⚠️ 엄격한 규칙:
- JSON 외 텍스트는 출력하지 마세요
- 원본 문서의 문장을 그대로 사용하지 마세요
- 사용자의 수정 요청을 반드시 반영하세요
- 모든 내용은 자신만의 언어로 재구성하여 작성하세요
"""
                
                try:
                    from utils.llm import LLMClient
                    llm = LLMClient(model_name=DEFAULT_LLM_CONFIG["model"])
                    modified_response = llm.generate(modification_prompt, num_predict=3000, temperature=DEFAULT_LLM_CONFIG["temperature"])
                    
                    # 수정된 응답 파싱
                    modified_parsed = llm.parse_response(modified_response)
                    
                    if isinstance(modified_parsed, dict) and not modified_parsed.get("error"):
                        print("\n===== 수정된 보고서 =====")
                        from utils.output_format import format_report_output
                        print(format_report_output(modified_parsed, mode, sources))
                        
                        # 수정된 결과를 원본에 반영
                        parsed.update(modified_parsed)
                    else:
                        print("\n===== 수정 요청 답변 =====")
                        print(modified_response)
                        
                except Exception as e:
                    print(f"보고서 수정 중 오류: {e}")
                    # 일반 답변으로 fallback
                    from utils.llm import LLMClient
                    llm = LLMClient(model_name=DEFAULT_LLM_CONFIG["model"])
                    followup_prompt = f"""
아래는 기존 보고서입니다. 사용자의 추가 요청(수정/질문 등)에 따라 적절히 답변하거나, 해당 부분을 수정해 주세요.

[기존 보고서]
{answer_text}

[사용자 추가 요청]
{followup}

반드시 한국어로만 답변하세요.
"""
                    followup_answer = llm.generate(followup_prompt, num_predict=1500, temperature=DEFAULT_LLM_CONFIG["temperature"])
                    print("\n===== 추가 요청/수정/답변 결과 =====\n")
                    print(followup_answer)
            else:
                # 일반 질문/답변
                from utils.llm import LLMClient
                llm = LLMClient(model_name=DEFAULT_LLM_CONFIG["model"])
                followup_prompt = f"""
아래는 기존 보고서입니다. 사용자의 추가 요청(수정/질문 등)에 따라 적절히 답변하거나, 해당 부분을 수정해 주세요.

[기존 보고서]
{answer_text}

[사용자 추가 요청]
{followup}

반드시 한국어로만 답변하세요.
"""
                followup_answer = llm.generate(followup_prompt, num_predict=1500, temperature=DEFAULT_LLM_CONFIG["temperature"])
                print("\n===== 추가 요청/수정/답변 결과 =====\n")
                print(followup_answer)
    except Exception as e:
        print(f"피드백 저장 중 오류: {e}")

# ==========================
# ✅ 8. CLI 인터페이스
# ==========================
def run_cli():
    """CLI 인터페이스 실행 함수 (스트리밍 및 병렬 처리 지원)"""
    print("=" * 60)
    print("📚 RAG LangGraph QA 시스템 (개선된 버전)")
    print("🤖 Ollama Mistral 모델 기반")
    print("🔄 스트리밍 출력 지원")
    print("⚡ 병렬 처리 지원")
    print("📊 진행률 표시 지원")
    print("📝 로깅 시스템 통합")
    print("=" * 60)
    
    # 병렬 처리 설정 (설정 파일에서 가져오기)
    use_parallel = config_manager.is_parallel_enabled()
    max_workers = config_manager.get_max_workers()
    
    # 임베딩 인덱스 자동 생성/로드 (병렬 처리 지원)
    logger.info("임베딩 인덱스 준비 중...")
    embedding_index = get_or_build_embedding_index(use_parallel=use_parallel, max_workers=max_workers)
    if embedding_index is None:
        logger.error("임베딩 인덱스 준비 실패")
        print("❌ 임베딩 인덱스 준비 실패. 문서를 ./docs/embedding/에 넣어주세요!")
        return
    logger.info("임베딩 인덱스 준비 완료")
    
    # 현재 LLM 설정
    current_config = DEFAULT_LLM_CONFIG.copy()
    streaming_mode = False
    
    print("\n📖 사용법:")
    print("  • 요약: '요약해줘', '간단히' 등의 키워드")
    print("  • 보고서: '보고서로 작성해줘', '구조화해서' 등의 키워드")
    print("  • 일반질문: 그 외 모든 질문 (문서 참고 + 일반 지식)")
    print("  • 스트리밍: 'stream' 입력 후 질문")
    print("  • 병렬처리: 'parallel' 입력 후 on/off")
    print("  • 워커수: 'workers' 입력 후 숫자")
    print("  • 설정 변경: 'config' 입력")
    print("  • 종료: 'quit', 'exit', 'q'")
    
    print(f"\n⚙️ 현재 설정:")
    print(f"  • 병렬 처리: {'활성화' if use_parallel else '비활성화'}")
    print(f"  • 워커 수: {max_workers}")
    print(f"  • 스트리밍: {'활성화' if streaming_mode else '비활성화'}")
    
    while True:
        try:
            question = input("\n🤔 질문을 입력하세요: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['quit', 'exit', 'q']:
                logger.info("사용자가 시스템을 종료했습니다.")
                print("👋 시스템을 종료합니다.")
                break
                
            if question.lower() == 'config':
                current_config = configure_llm_settings(current_config)
                continue
                
            if question.lower() == 'stream':
                streaming_mode = not streaming_mode
                status = "활성화" if streaming_mode else "비활성화"
                print(f"🔄 스트리밍 모드 {status}")
                print(f"⚙️ 현재 설정: 병렬 처리 {'활성화' if use_parallel else '비활성화'}, 워커 수 {max_workers}, 스트리밍 {status}")
                continue
                
            if question.lower() == 'parallel':
                use_parallel = not use_parallel
                status = "활성화" if use_parallel else "비활성화"
                print(f"⚡ 병렬 처리 모드 {status}")
                print(f"⚙️ 현재 설정: 병렬 처리 {status}, 워커 수 {max_workers}")
                continue
                
            if question.lower() == 'workers':
                try:
                    new_workers = int(input("워커 수를 입력하세요 (1-8): "))
                    if 1 <= new_workers <= 8:
                        max_workers = new_workers
                        status = "활성화" if use_parallel else "비활성화"
                        print(f"⚡ 워커 수가 {max_workers}로 설정되었습니다.")
                        print(f"⚙️ 현재 설정: 병렬 처리 {status}, 워커 수 {max_workers}")
                    else:
                        print("❌ 워커 수는 1-8 사이여야 합니다.")
                except ValueError:
                    print("❌ 유효한 숫자를 입력하세요.")
                continue
            
            # 스트리밍 모드가 활성화되어 있으면 다음 질문을 스트리밍으로 처리
            main(question, embedding_index, llm_config=current_config, streaming=streaming_mode, use_parallel=use_parallel, max_workers=max_workers)
            
        except KeyboardInterrupt:
            logger.info("사용자가 Ctrl+C로 시스템을 종료했습니다.")
            print("\n\n👋 시스템을 종료합니다.")
            break
        except Exception as e:
            logger.error(f"CLI 실행 중 오류: {e}")
            print(f"❌ 오류 발생: {e}")

# ✅ 진입점
if __name__ == "__main__":
    run_cli() 