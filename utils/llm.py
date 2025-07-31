# llm.py
"""
LLM 프롬프트 생성, 호출, 응답 파싱 모듈
"""
import requests
import json
import re
import logging
from typing import Any, Generator, Optional
from contextlib import contextmanager

# 로깅 설정
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api"
        logger.info(f"LLMClient 초기화: {model_name}")

    def generate(self, prompt: str, num_predict: int = 3000, temperature: float = 0.3) -> str:
        """일반 LLM 호출 (스트리밍 없음)"""
        logger.debug(f"LLM 호출 시작: {len(prompt)}자 프롬프트")
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": num_predict
                    }
                }),
                timeout=900  # 15분 타임아웃
            )
            
            if response.status_code != 200:
                error_msg = f"Ollama API 오류: {response.status_code}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
            resp_json = response.json()
            if "response" not in resp_json:
                error_msg = f"Ollama 응답에 'response' 키가 없습니다: {resp_json}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
            logger.debug(f"LLM 응답 수신 완료: {len(resp_json['response'])}자")
            return resp_json["response"]
            
        except requests.exceptions.Timeout:
            error_msg = "LLM 호출 타임아웃"
            logger.error(error_msg)
            raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"LLM API 요청 오류: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"LLM 호출 중 예상치 못한 오류: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def stream_generate(self, prompt: str, num_predict: int = 3000, temperature: float = 0.3) -> Generator[str, None, None]:
        """스트리밍 LLM 호출"""
        logger.debug(f"스트리밍 LLM 호출 시작: {len(prompt)}자 프롬프트")
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": num_predict
                    }
                }),
                timeout=900,
                stream=True
            )
            
            if response.status_code != 200:
                error_msg = f"스트리밍 API 오류: {response.status_code}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if "response" in chunk:
                            text_chunk = chunk["response"]
                            full_response += text_chunk
                            yield text_chunk
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError as e:
                        logger.warning(f"스트리밍 청크 파싱 실패: {e}")
                        continue
            
            logger.debug(f"스트리밍 완료: {len(full_response)}자")
            
        except requests.exceptions.Timeout:
            error_msg = "스트리밍 LLM 호출 타임아웃"
            logger.error(error_msg)
            raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"스트리밍 API 요청 오류: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"스트리밍 호출 중 예상치 못한 오류: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    @contextmanager
    def safe_generate(self, prompt: str, num_predict: int = 3000, temperature: float = 0.3):
        """안전한 LLM 호출 컨텍스트 매니저"""
        try:
            response = self.generate(prompt, num_predict, temperature)
            yield response
        except Exception as e:
            logger.error(f"LLM 호출 실패: {e}")
            yield f"LLM 호출 중 오류가 발생했습니다: {str(e)}"

    def parse_response(self, response: str) -> Any:
        """LLM 응답을 파싱하여 구조화된 데이터로 변환"""
        if not response or not response.strip():
            logger.warning("빈 응답 수신")
            return {"error": "빈 응답"}
        
        logger.debug(f"응답 파싱 시작: {len(response)}자")
        
        # 1. JSON 블록 찾기 시도
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_match = re.search(json_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if json_match:
            try:
                json_str = json_match.group(1)
                parsed = json.loads(json_str)
                logger.debug("JSON 코드 블록 파싱 성공")
                return parsed
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 파싱 실패 (코드 블록): {e}")
        
        # 2. 일반 JSON 찾기 시도
        try:
            # 응답에서 JSON 부분만 추출
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                logger.debug("일반 JSON 파싱 성공")
                return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패 (일반): {e}")
        
        # 3. 구조화된 텍스트에서 필드 추출 시도
        try:
            parsed = {}
            
            # 제목 추출
            title_match = re.search(r'"title":\s*"([^"]+)"', response)
            if title_match:
                parsed["title"] = title_match.group(1)
            
            # 요약 추출
            summary_match = re.search(r'"summary":\s*"([^"]+)"', response)
            if summary_match:
                parsed["summary"] = summary_match.group(1)
            
            # abstract 추출
            abstract_match = re.search(r'"abstract":\s*"([^"]+)"', response)
            if abstract_match:
                parsed["abstract"] = abstract_match.group(1)
            
            # introduction 추출
            intro_match = re.search(r'"introduction":\s*"([^"]+)"', response)
            if intro_match:
                parsed["introduction"] = intro_match.group(1)
            
            # background 추출
            bg_match = re.search(r'"background":\s*"([^"]+)"', response)
            if bg_match:
                parsed["background"] = bg_match.group(1)
            
            # main_content 추출
            main_match = re.search(r'"main_content":\s*"([^"]+)"', response)
            if main_match:
                parsed["main_content"] = main_match.group(1)
            
            # conclusion 추출
            concl_match = re.search(r'"conclusion":\s*"([^"]+)"', response)
            if concl_match:
                parsed["conclusion"] = concl_match.group(1)
            
            if parsed:
                logger.debug(f"정규식 파싱 성공: {list(parsed.keys())}")
                return parsed
        except Exception as e:
            logger.warning(f"정규식 파싱 실패: {e}")
        
        # 4. 모든 파싱 실패 시 원본 응답 반환
        logger.warning("모든 파싱 방법 실패, 원본 응답 반환")
        return {"raw_response": response, "error": "JSON 파싱 실패"} 