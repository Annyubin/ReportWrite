# search.py
"""
LangChain FAISS 벡터스토어 기반 검색 함수 모듈
"""
from langchain.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from typing import List, Dict, Any
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# 벡터스토어 생성 및 저장

def create_faiss_vectorstore(docs: List[str], metadatas: List[Dict[str, Any]], embedding_model, save_path: str):
    """
    문서와 메타데이터, 임베딩 모델로 LangChain FAISS 벡터스토어 생성 및 저장
    """
    doc_objs = [Document(page_content=doc, metadata=meta) for doc, meta in zip(docs, metadatas)]
    vectorstore = FAISS.from_documents(doc_objs, embedding_model)
    vectorstore.save_local(save_path)
    return vectorstore

# 벡터스토어 로드 및 검색

def load_faiss_vectorstore(load_path: str, embedding_model):
    """
    저장된 FAISS 벡터스토어 로드
    """
    vectorstore = FAISS.load_local(load_path, embedding_model)
    return vectorstore

def get_retriever(vectorstore) -> VectorStoreRetriever:
    """
    벡터스토어에서 retriever 객체 반환
    """
    return vectorstore.as_retriever()

# 검색 함수 예시 (query: str, retriever: VectorStoreRetriever)
def similarity_search(query: str, retriever, k: int = 5):
    """
    쿼리로 유사 문서 검색 (코사인 유사도 + 문서 길이 혼합 정렬)
    """
    if hasattr(retriever, 'search_kwargs'):
        retriever.search_kwargs["k"] = max(k * 2, 10)  # 후보군을 넉넉히 뽑음
    results = retriever.invoke(query)
    # 혼합 점수 계산: (코사인 유사도 * 0.7 + 문서 길이 정규화 * 0.3)
    scored = []
    max_len = max((len(doc.page_content) for doc in results), default=1)
    for doc in results:
        # 유사도 점수 추출 (Document.metadata["score"] 또는 없음)
        score = doc.metadata.get("score", 0)
        # 길이 정규화
        len_norm = len(doc.page_content) / max_len if max_len > 0 else 0
        hybrid_score = score * 0.7 + len_norm * 0.3
        scored.append((hybrid_score, doc))
    # 혼합 점수로 정렬 (내림차순)
    scored.sort(key=lambda x: x[0], reverse=True)
    # 상위 k개만 반환
    return [doc for _, doc in scored[:k]]

def hybrid_search(query: str, retriever, docs: List[Document], k: int = 5):
    """
    FAISS(코사인 유사도) + BM25 키워드 기반 hybrid 검색
    """
    # FAISS(코사인 유사도 + 길이) 후보군
    faiss_results = similarity_search(query, retriever, k=max(k*2, 10))
    faiss_scores = {}
    for i, doc in enumerate(faiss_results):
        score = doc.metadata.get("score", 0)
        faiss_scores[doc] = score
    # BM25 후보군
    corpus = [doc.page_content for doc in docs]
    tokenized_corpus = [c.split() for c in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    # 각 문서별 hybrid 점수 계산 (FAISS 0.7 + BM25 0.3)
    hybrid_candidates = set(faiss_results)
    for i, doc in enumerate(docs):
        if bm25_scores[i] > 0:
            hybrid_candidates.add(doc)
    scored = []
    max_bm25 = max(bm25_scores) if bm25_scores else 1
    for doc in hybrid_candidates:
        faiss_score = faiss_scores.get(doc, 0)
        try:
            bm25_idx = docs.index(doc)
            bm25_score = bm25_scores[bm25_idx] / max_bm25 if max_bm25 > 0 else 0
        except ValueError:
            bm25_score = 0
        hybrid_score = faiss_score * 0.7 + bm25_score * 0.3
        scored.append((hybrid_score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:k]] 