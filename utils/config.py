"""
설정 관리 모듈
"""
import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"설정 파일 로드 완료: {self.config_path}")
                return config
            else:
                logger.warning(f"설정 파일이 없습니다: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            "llm": {
                "model": "mistral",
                "temperature": 0.1,
                "num_predict": 3000,
                "top_k": 3,
                "top_p": 0.9,
                "timeout": 900,
                "base_url": "http://localhost:11434/api"
            },
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "max_chars": 15000
            },
            "search": {
                "default_top_k": 5,
                "max_doc_chars": 3500,
                "similarity_threshold": 0.7
            },
            "paths": {
                "docs_dir": "./docs/embedding/",
                "index_path": "./faiss_index",
                "logs_dir": "./logs/",
                "feedback_log": "./feedback_log.txt"
            },
            "output": {
                "min_lengths": {
                    "abstract": 300,
                    "introduction": 300,
                    "background": 300,
                    "main_content": 300,
                    "conclusion": 300
                },
                "enable_streaming": True,
                "enable_highlighting": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "./logs/system.log",
                "max_size": "10MB",
                "backup_count": 5
            },
            "parallel": {
                "enabled": True,
                "max_workers": 4,
                "batch_size": 10,
                "show_progress": True,
                "description": "처리 중"
            }
        }
    
    def _setup_logging(self):
        """로깅 설정"""
        try:
            log_config = self.config.get("logging", {})
            log_level = getattr(logging, log_config.get("level", "INFO"))
            log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            log_file = log_config.get("file", "./logs/system.log")
            
            # 로그 디렉토리 생성
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # 로깅 핸들러 설정
            from logging.handlers import RotatingFileHandler
            
            # 파일 핸들러
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter(log_format))
            
            # 콘솔 핸들러
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            
            # 루트 로거 설정
            root_logger = logging.getLogger()
            root_logger.setLevel(log_level)
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            
            logger.info("로깅 시스템 초기화 완료")
            
        except Exception as e:
            print(f"로깅 설정 실패: {e}")
            # 기본 로깅 설정
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정 값 가져오기 (점 표기법 지원)"""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """설정 값 설정"""
        try:
            keys = key.split('.')
            config = self.config
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value
            logger.info(f"설정 업데이트: {key} = {value}")
        except Exception as e:
            logger.error(f"설정 업데이트 실패: {e}")
    
    def save(self):
        """설정 파일 저장"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"설정 파일 저장 완료: {self.config_path}")
        except Exception as e:
            logger.error(f"설정 파일 저장 실패: {e}")
    
    def get_llm_config(self) -> Dict[str, Any]:
        """LLM 설정 반환"""
        return self.config.get("llm", {})
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """임베딩 설정 반환"""
        return self.config.get("embedding", {})
    
    def get_search_config(self) -> Dict[str, Any]:
        """검색 설정 반환"""
        return self.config.get("search", {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """경로 설정 반환"""
        return self.config.get("paths", {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """출력 설정 반환"""
        return self.config.get("output", {})
    
    def get_parallel_config(self) -> Dict[str, Any]:
        """병렬 처리 설정 반환"""
        return self.config.get("parallel", {})
    
    def is_parallel_enabled(self) -> bool:
        """병렬 처리 활성화 여부 반환"""
        return self.config.get("parallel", {}).get("enabled", True)
    
    def get_max_workers(self) -> int:
        """최대 워커 수 반환"""
        return self.config.get("parallel", {}).get("max_workers", 4)
    
    def get_batch_size(self) -> int:
        """배치 크기 반환"""
        return self.config.get("parallel", {}).get("batch_size", 10)
    
    def is_progress_enabled(self) -> bool:
        """진행률 표시 활성화 여부 반환"""
        return self.config.get("parallel", {}).get("show_progress", True)

# 전역 설정 인스턴스
config_manager = ConfigManager() 