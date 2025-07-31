"""
병렬 처리 및 진행률 표시 유틸리티
"""
import asyncio
import concurrent.futures
from typing import List, Callable, Any, Dict, Optional, Generator
from tqdm import tqdm
import logging
import time
from functools import partial

logger = logging.getLogger(__name__)

class ProgressTracker:
    """진행률 추적 및 표시 클래스"""
    
    def __init__(self, total: int, description: str = "처리 중", unit: str = "개"):
        self.total = total
        self.description = description
        self.unit = unit
        self.current = 0
        self.start_time = time.time()
        self.pbar = None
        
    def __enter__(self):
        self.pbar = tqdm(
            total=self.total,
            desc=self.description,
            unit=self.unit,
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            self.pbar.close()
            
    def update(self, n: int = 1, description: Optional[str] = None):
        """진행률 업데이트"""
        if self.pbar:
            if description:
                self.pbar.set_description(description)
            self.pbar.update(n)
            self.current += n
            
    def set_description(self, description: str):
        """설명 업데이트"""
        if self.pbar:
            self.pbar.set_description(description)

def parallel_process(
    func: Callable,
    items: List[Any],
    max_workers: int = 4,
    description: str = "병렬 처리 중",
    show_progress: bool = True,
    **kwargs
) -> List[Any]:
    """
    병렬 처리 함수
    
    Args:
        func: 실행할 함수
        items: 처리할 아이템 리스트
        max_workers: 최대 워커 수
        description: 진행률 표시 설명
        show_progress: 진행률 표시 여부
        **kwargs: 함수에 전달할 추가 인자
        
    Returns:
        처리 결과 리스트
    """
    if not items:
        return []
        
    results = []
    total_items = len(items)
    
    # 진행률 표시 설정
    progress_tracker = None
    if show_progress:
        progress_tracker = ProgressTracker(total_items, description)
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 함수에 추가 인자 바인딩
            bound_func = partial(func, **kwargs) if kwargs else func
            
            # 작업 제출
            future_to_item = {
                executor.submit(bound_func, item): item 
                for item in items
            }
            
            # 결과 수집
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if progress_tracker:
                        progress_tracker.update(1)
                        
                except Exception as e:
                    item = future_to_item[future]
                    logger.error(f"아이템 처리 실패: {item}, 오류: {e}")
                    if progress_tracker:
                        progress_tracker.update(1)
                        
    except Exception as e:
        logger.error(f"병렬 처리 중 오류: {e}")
        
    finally:
        if progress_tracker:
            progress_tracker.__exit__(None, None, None)
            
    return results

async def async_parallel_process(
    func: Callable,
    items: List[Any],
    max_concurrent: int = 4,
    description: str = "비동기 병렬 처리 중",
    show_progress: bool = True,
    **kwargs
) -> List[Any]:
    """
    비동기 병렬 처리 함수
    
    Args:
        func: 실행할 비동기 함수
        items: 처리할 아이템 리스트
        max_concurrent: 최대 동시 실행 수
        description: 진행률 표시 설명
        show_progress: 진행률 표시 여부
        **kwargs: 함수에 전달할 추가 인자
        
    Returns:
        처리 결과 리스트
    """
    if not items:
        return []
        
    results = []
    total_items = len(items)
    
    # 진행률 표시 설정
    progress_tracker = None
    if show_progress:
        progress_tracker = ProgressTracker(total_items, description)
    
    try:
        # 세마포어로 동시 실행 수 제한
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_item(item):
            async with semaphore:
                try:
                    # 함수에 추가 인자 전달
                    if kwargs:
                        result = await func(item, **kwargs)
                    else:
                        result = await func(item)
                    return result
                except Exception as e:
                    logger.error(f"아이템 처리 실패: {item}, 오류: {e}")
                    return None
        
        # 모든 작업 생성
        tasks = [process_item(item) for item in items]
        
        # 진행률 업데이트를 위한 콜백
        completed = 0
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            completed += 1
            
            if progress_tracker:
                progress_tracker.update(1)
                
    except Exception as e:
        logger.error(f"비동기 병렬 처리 중 오류: {e}")
        
    finally:
        if progress_tracker:
            progress_tracker.__exit__(None, None, None)
            
    return results

def batch_process(
    func: Callable,
    items: List[Any],
    batch_size: int = 10,
    max_workers: int = 4,
    description: str = "배치 처리 중",
    show_progress: bool = True,
    **kwargs
) -> List[Any]:
    """
    배치 단위 병렬 처리 함수
    
    Args:
        func: 실행할 함수
        items: 처리할 아이템 리스트
        batch_size: 배치 크기
        max_workers: 최대 워커 수
        description: 진행률 표시 설명
        show_progress: 진행률 표시 여부
        **kwargs: 함수에 전달할 추가 인자
        
    Returns:
        처리 결과 리스트
    """
    if not items:
        return []
        
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    # 진행률 표시 설정
    progress_tracker = None
    if show_progress:
        progress_tracker = ProgressTracker(total_batches, description, "배치")
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 배치로 나누기
            batches = [
                items[i:i + batch_size] 
                for i in range(0, len(items), batch_size)
            ]
            
            # 배치 처리 함수
            def process_batch(batch):
                batch_results = []
                for item in batch:
                    try:
                        if kwargs:
                            result = func(item, **kwargs)
                        else:
                            result = func(item)
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"배치 내 아이템 처리 실패: {item}, 오류: {e}")
                        batch_results.append(None)
                return batch_results
            
            # 배치 작업 제출
            future_to_batch = {
                executor.submit(process_batch, batch): batch 
                for batch in batches
            }
            
            # 결과 수집
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    
                    if progress_tracker:
                        progress_tracker.update(1)
                        
                except Exception as e:
                    batch = future_to_batch[future]
                    logger.error(f"배치 처리 실패: {batch}, 오류: {e}")
                    if progress_tracker:
                        progress_tracker.update(1)
                        
    except Exception as e:
        logger.error(f"배치 처리 중 오류: {e}")
        
    finally:
        if progress_tracker:
            progress_tracker.__exit__(None, None, None)
            
    return results

def streaming_progress(
    generator: Generator,
    total: Optional[int] = None,
    description: str = "스트리밍 처리 중",
    unit: str = "개"
) -> Generator:
    """
    제너레이터에 진행률 표시 추가
    
    Args:
        generator: 원본 제너레이터
        total: 총 아이템 수 (None이면 자동 계산 불가)
        description: 진행률 표시 설명
        unit: 단위
        
    Yields:
        제너레이터의 각 아이템
    """
    progress_tracker = None
    if total is not None:
        progress_tracker = ProgressTracker(total, description, unit)
        progress_tracker.__enter__()
    
    try:
        for item in generator:
            yield item
            if progress_tracker:
                progress_tracker.update(1)
    finally:
        if progress_tracker:
            progress_tracker.__exit__(None, None, None)

# 문서 처리 전용 병렬 함수들
def parallel_document_processing(
    documents: List[tuple],
    process_func: Callable,
    max_workers: int = 4,
    description: str = "문서 처리 중",
    **kwargs
) -> List[tuple]:
    """
    문서 병렬 처리 전용 함수
    
    Args:
        documents: (content, metadata) 튜플 리스트
        process_func: 문서 처리 함수
        max_workers: 최대 워커 수
        description: 진행률 표시 설명
        **kwargs: 함수에 전달할 추가 인자
        
    Returns:
        처리된 문서 리스트
    """
    def process_document(doc_tuple):
        content, metadata = doc_tuple
        try:
            if kwargs:
                processed_content = process_func(content, **kwargs)
            else:
                processed_content = process_func(content)
            return (processed_content, metadata)
        except Exception as e:
            logger.error(f"문서 처리 실패: {metadata.get('filename', 'unknown')}, 오류: {e}")
            return (content, metadata)  # 원본 반환
    
    return parallel_process(
        process_document,
        documents,
        max_workers=max_workers,
        description=description,
        **kwargs
    )

def parallel_chunk_processing(
    chunks: List[str],
    process_func: Callable,
    max_workers: int = 4,
    description: str = "청크 처리 중",
    **kwargs
) -> List[str]:
    """
    청크 병렬 처리 전용 함수
    
    Args:
        chunks: 텍스트 청크 리스트
        process_func: 청크 처리 함수
        max_workers: 최대 워커 수
        description: 진행률 표시 설명
        **kwargs: 함수에 전달할 추가 인자
        
    Returns:
        처리된 청크 리스트
    """
    return parallel_process(
        process_func,
        chunks,
        max_workers=max_workers,
        description=description,
        **kwargs
    ) 