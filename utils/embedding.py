# embedding.py
"""
임베딩 모델, 문서 로드, 청크 분할 등 임베딩 관련 기능 모듈
"""
# 이 파일(utils/embedding.py)에서만 SentenceTransformer를 직접 임포트/사용하세요.
# 외부에서 SentenceTransformer를 직접 임포트/사용하면 ImportError가 발생합니다.
import sys
if __name__ != '__main__' and 'sentence_transformers' in sys.modules and sys.modules['__name__'] != 'utils.embedding':
    raise ImportError('SentenceTransformer는 반드시 EmbeddingModel 래퍼를 통해서만 사용하세요!')

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
import os
import json
import traceback
from tqdm import tqdm
import re
from langchain_core.embeddings import Embeddings
import numpy as np
import logging

# 병렬 처리 유틸리티 import
from .parallel import parallel_process, ProgressTracker

logger = logging.getLogger(__name__)

def extract_texts_from_json(data):
    """
    다양한 json 구조에서 텍스트를 최대한 추출하는 범용 함수
    정책/기술 설명이 포함된 본문 필드까지 깊게 탐색 (우선순위: 본문, 내용, description, main_content, details, explanation 등)
    """
    texts = []
    # 우선순위 키워드: 실제 본문/내용/설명/세부사항 등
    priority_keys = [
        '본문', '내용', 'description', 'main_content', 'details', 'explanation',
        'text', 'content', 'body', 'value', 'summary', 'abstract', 'background', 'introduction', 'conclusion'
    ]
    def recursive_extract(obj):
        if isinstance(obj, dict):
            for key in priority_keys:
                if key in obj and isinstance(obj[key], str) and obj[key].strip():
                    texts.append(obj[key])
            for v in obj.values():
                recursive_extract(v)
        elif isinstance(obj, list):
            for item in obj:
                recursive_extract(item)
    recursive_extract(data)
    return texts

def preprocess_text(text: str) -> str:
    """
    본문 텍스트 전처리 함수
    - HTML 태그 제거
    - 특수문자 및 이모지 제거
    - 중복 문장 제거 (한 문단 내)
    - 번호/줄바꿈/구분자 제거 및 자연어 문장 리포맷팅
    - 길이 제한(2000자)
    - 전화번호/팩스번호/문의처/안내 등 불필요한 문장 제외
    """
    if not isinstance(text, str):
        return ''
    text = text.strip()
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    # 특수문자 및 이모지 제거 (한글, 영문, 숫자, 공백만 남김)
    text = re.sub(r'[^ -~가-힣\s]', '', text)
    # 번호/구분자/불필요한 줄바꿈 제거
    text = re.sub(r'\n+', ' ', text)  # 연속 줄바꿈 -> 공백
    text = re.sub(r'(\d+\s*[.)]|[•\-\*])\s*', '', text)  # 1. 2) 3) - • * 등 제거
    text = re.sub(r'\s{2,}', ' ', text)  # 연속 공백 정리
    # 문장 단위로 쪼개기
    sentences = re.split(r'[.!?]', text)
    seen = set()
    unique_sentences = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # 안내/문의/전화번호 등 불필요 문장 필터링
        if re.search(r'(전화번호|팩스번호|문의처|연락처|전화\s*[:：]|fax|FAX|Tel|TEL|전화\d|\d{2,4}-\d{3,4}-\d{4}|문의하시기 바랍니다|문의 바랍니다|문의해주시기 바랍니다|문의해 주세요|문의해 주시기 바랍니다|안내)', s):
            continue
        # 문장 끝에 마침표가 없으면 붙여줌
        if not s.endswith('.'):
            s += '.'
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)
    text = ' '.join(unique_sentences)
    # 길이 제한 (2000자)
    if len(text) > 2000:
        text = text[:2000]
    return text

# load_documents_from_directory에서 50자 미만 텍스트는 건너뛰기

def load_documents_from_directory(directory: str, file_extensions: Optional[List[str]] = None, use_parallel: bool = True, max_workers: int = 4) -> List[Tuple[str, Dict[str, Any]]]:
    """
    디렉토리에서 문서를 로드하는 함수 (병렬 처리 지원)
    
    Args:
        directory: 문서 디렉토리 경로
        file_extensions: 지원할 파일 확장자 리스트
        use_parallel: 병렬 처리 사용 여부
        max_workers: 최대 워커 수
        
    Returns:
        (content, metadata) 튜플 리스트
    """
    if file_extensions is None:
        file_extensions = ['.txt', '.md', '.json']
    
    def process_single_file(file_info):
        filename, filepath, ext = file_info
        try:
            if ext == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    texts = extract_texts_from_json(data)
                    file_documents = []
                    if texts:
                        for idx, text in enumerate(texts):
                            processed = preprocess_text(text)
                            if not processed or len(processed) < 50:
                                continue
                            metadata = {
                                "filename": filename,
                                "filepath": filepath,
                                "extension": ext,
                                "chunk_index": idx + 1
                            }
                            file_documents.append((processed, metadata))
                    return file_documents
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    processed = preprocess_text(content)
                    if not processed or len(processed) < 50:
                        return []
                    metadata = {
                        "filename": filename,
                        "filepath": filepath,
                        "extension": ext
                    }
                    return [(processed, metadata)]
        except Exception as e:
            logger.error(f"파일 로드 실패 {filename}: {e}")
            return []
    
    # 파일 목록 수집
    file_infos = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext in file_extensions:
                file_infos.append((filename, filepath, ext))
    
    if not file_infos:
        logger.warning(f"처리할 파일을 찾을 수 없습니다: {directory}")
        return []
    
    # 병렬 처리 또는 순차 처리
    from .config import config_manager
    show_progress = config_manager.is_progress_enabled()
    
    if use_parallel and len(file_infos) > 1:
        logger.info(f"병렬 처리로 {len(file_infos)}개 파일을 로드합니다.")
        file_documents_list = parallel_process(
            process_single_file,
            file_infos,
            max_workers=max_workers,
            description="문서 로드 중",
            show_progress=show_progress
        )
    else:
        logger.info(f"순차 처리로 {len(file_infos)}개 파일을 로드합니다.")
        file_documents_list = [process_single_file(file_info) for file_info in file_infos]
    
    # 모든 문서 합치기
    documents = []
    for file_documents in file_documents_list:
        documents.extend(file_documents)
    
    logger.info(f"총 {len(documents)}개 문서를 로드했습니다.")
    return documents

def create_document_chunks(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """
    chunk_size, overlap을 파라미터로 노출(기본값 2000/200)
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # 빈 청크 제외
            chunks.append(chunk)
        start = end - overlap
    return chunks

def create_document_chunks_parallel(
    documents: List[Tuple[str, Dict[str, Any]]], 
    chunk_size: Optional[int] = None, 
    overlap: Optional[int] = None,
    use_parallel: bool = True,
    max_workers: int = 4
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    문서 청크를 병렬로 생성하는 함수
    
    Args:
        documents: (content, metadata) 튜플 리스트
        chunk_size: 청크 크기 (None이면 설정 파일에서 가져옴)
        overlap: 청크 간 겹침 크기 (None이면 설정 파일에서 가져옴)
        use_parallel: 병렬 처리 사용 여부
        max_workers: 최대 워커 수
        
    Returns:
        (chunks, metadata) 튜플
    """
    # 설정 파일에서 기본값 가져오기
    from .config import config_manager
    embedding_config = config_manager.get_embedding_config()
    chunk_size = chunk_size or embedding_config.get("chunk_size", 2000)
    overlap = overlap or embedding_config.get("chunk_overlap", 200)
    show_progress = config_manager.is_progress_enabled()
    
    def process_document_chunks(doc_tuple):
        content, metadata = doc_tuple
        try:
            chunks = create_document_chunks(content, chunk_size, overlap)
            return [(chunk, metadata) for chunk in chunks]
        except Exception as e:
            logger.error(f"문서 청크 생성 실패: {metadata.get('filename', 'unknown')}, 오류: {e}")
            return []
    
    if use_parallel and len(documents) > 1:
        logger.info(f"병렬 처리로 {len(documents)}개 문서의 청크를 생성합니다.")
        chunk_results = parallel_process(
            process_document_chunks,
            documents,
            max_workers=max_workers,
            description="청크 생성 중",
            show_progress=show_progress
        )
    else:
        logger.info(f"순차 처리로 {len(documents)}개 문서의 청크를 생성합니다.")
        chunk_results = [process_document_chunks(doc) for doc in documents]
    
    # 모든 청크 합치기
    all_chunks = []
    all_metadata = []
    for chunk_list in chunk_results:
        for chunk, metadata in chunk_list:
            all_chunks.append(chunk)
            all_metadata.append(metadata)
    
    logger.info(f"총 {len(all_chunks)}개 청크를 생성했습니다.")
    return all_chunks, all_metadata

class EmbeddingModel(Embeddings):
    def __init__(self, model_name: str = "jhgan/ko-sbert-sts", batch_size: int = 16):
        """
        model_name: SentenceTransformer에서 지원하는 모델명을 지정하세요.
        추천: "jhgan/ko-sbert-sts" (한국어 의미 유사도)
        예시: "paraphrase-multilingual-MiniLM-L12-v2", "jhgan/ko-sroberta-multitask"
        """
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
    def encode(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        if batch_size is None:
            batch_size = self.batch_size
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            vecs = self.model.encode(batch, show_progress_bar=False)
            # L2 정규화
            vecs = np.array(vecs)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
            vecs = (vecs / norms).tolist()
            results.extend(vecs)
        return results
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.encode(texts)
    def embed_query(self, text: str) -> List[float]:
        return self.encode([text])[0]

def embed_query(query: str):
    import numpy as np
    np.random.seed(abs(hash(query)) % (10**8))
    return np.random.rand(512).astype("float32") 