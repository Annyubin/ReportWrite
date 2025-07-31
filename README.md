# 의료기기 문서 기반 RAG QA 시스템

## 📋 개요
의료기기 인허가/가이드라인 문서를 기반으로 **LangGraph + Ollama LLM**을 활용한 고급 RAG QA 시스템입니다.

### ✨ 주요 기능
- **완전한 모듈화**: 임베딩, 검색, LLM, 출력, 프롬프트 등 모든 기능이 utils 하위 모듈로 분리
- **코사인 유사도 기반 검색**: 항상 5개 청크를 추출하여 정확한 검색 결과 제공
- **자동 길이 보완**: 각 항목이 300자 미만이면 LLM이 자동으로 재생성
- **개선된 피드백 시스템**: 수정 요청 시 JSON 형식으로 보고서 재생성
- **깔끔한 UI**: 불필요한 디버그 메시지 제거로 사용자 친화적 인터페이스
- **출처 자동 표기**: 보고서 마지막에 사용된 문서 파일명/경로 자동 표기
- **보고서/요약/자유질문 모드**: 키워드에 따라 자동 분기
- **LLM 설정 동적 변경**: config 명령어로 런타임에 모델, temperature 등 조정 가능

---

## 🛠 환경 설정

### 1. 시스템 요구사항
- **Python 3.9+**
- **Ollama 서버** (로컬 또는 원격)

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. Ollama 모델 준비
```bash
# Mistral 모델 다운로드 및 실행
ollama pull mistral
ollama run mistral
```

---

## 📁 폴더/파일 구조

```
rag_langgraph_cli.py          # 메인 실행 파일 (LangGraph 워크플로우)
utils/
├── embedding.py              # 임베딩 모델, 문서 로드, 청크 분할
├── search.py                 # FAISS 인덱스 관리(코사인 유사도 + BM25)
├── llm.py                    # LLM 호출 및 JSON 응답 파싱
├── output_format.py          # 출력 포맷, 중복 제거, 보고서/출처
└── prompt_templates.py       # 프롬프트 템플릿
docs/
└── embedding/                # 실제 문서 저장 폴더
faiss_index.faiss/.pkl        # 인덱스 파일(자동 생성)
feedback_log.txt              # 피드백 로그 (자동 생성)
requirements.txt
README.md
```

---

## 🚀 실행 방법

1. 문서를 `./docs/embedding/` 폴더에 넣으세요.
2. Ollama 서버(mistral) 실행
3. 패키지 설치: `pip install -r requirements.txt`
4. 실행: `python rag_langgraph_cli.py`

---

## 💬 사용법

### 기본 사용
- 질문을 입력하면, 관련 문서 검색 → LLM 답변 → 자동 길이 보완 → 깔끔한 출력까지 자동 진행됩니다.
- 종료: `quit`, `exit`, `q` 입력
- LLM 설정 변경: `config` 입력

### 질문 모드 자동 감지
- **요약 모드**: "요약해줘", "간단히", "한줄로" 등의 키워드
- **보고서 모드**: "보고서로", "구조화해서", "분석해줘" 등의 키워드  
- **자유질문 모드**: 그 외 모든 질문 (문서 참고 + 일반 지식 활용)

### 자동 길이 보완
- 각 항목(abstract, introduction, background, main_content, conclusion)이 300자 미만이면 자동으로 LLM이 재생성
- 사용자 개입 없이 완전 자동화

### 피드백 및 수정
- 기본 피드백: "이 답변이 도움이 되었나요? (y/n)"
- 수정 요청: "보고서에 대해 추가로 수정하거나 궁금한 점이 있으면 입력하세요"
- 수정 키워드 감지: "수정", "바꿔", "고쳐", "다시", "재작성", "변경", "조정", "개선"
- 수정 요청 시 JSON 형식으로 전체 보고서 재생성

### 예시
```
🤔 의료기기 인허가 절차에 대해 보고서 작성해줘
🤔 이 문서를 요약해줘
🤔 간단히 정리해줘
🤔 2등급 의료기기 승인 기준은?

# 피드백 예시
결론 부분을 더 구체적으로 수정해줘
배경 부분을 더 자세히 설명해줘
```

---

## 🏗 시스템 아키텍처

### LangGraph 워크플로우
1. **generate_prompt**: 프롬프트 생성
2. **call_model**: LLM 호출
3. **parse_response**: JSON 응답 파싱
4. **fix_prompt_and_retry**: 실패 시 재시도 (최대 1회)

### 모듈 구조
- **rag_langgraph_cli.py**: LangGraph 워크플로우 및 메인 로직
- **utils/embedding.py**: 임베딩 모델, 문서 로드, 청크 분할
- **utils/search.py**: FAISS 인덱스 관리, 코사인 유사도 + BM25 하이브리드 검색
- **utils/llm.py**: LLM 호출, JSON 파싱 (코드 블록, 정규식 등 다중 파싱 방식)
- **utils/output_format.py**: Markdown 형식 출력, 출처 표시
- **utils/prompt_templates.py**: 프롬프트 템플릿 관리

---

## ⚙️ 설정 및 커스터마이징

### LLM 설정 예시
```python
DEFAULT_LLM_CONFIG = {
    "model": "mistral",
    "temperature": 0.1,  # 보수적인 답변
    "num_predict": 3000, # 토큰 수
    "top_k": 3,
    "top_p": 0.9,
    "timeout": 900       # 15분 타임아웃
}
```

### 검색 파라미터
- **고정 5개 청크**: 항상 5개의 관련 문서 청크를 추출
- **최대 15,000자**: 검색 결과 문서 크기 제한
- **각 문서 3,500자**: 개별 문서 길이 제한

### 자동 길이 보완 설정
```python
min_lengths = {
    "abstract": 300,
    "introduction": 300,
    "background": 300,
    "main_content": 300,
    "conclusion": 300
}
```

---

## 🔥 최신 기능

### 🆕 자동 길이 보완
- 각 항목이 300자 미만이면 자동으로 LLM이 재생성
- 사용자 개입 없이 완전 자동화

### 🆕 개선된 피드백 시스템
- 수정 요청 감지 및 JSON 형식 재생성
- 키워드 기반 수정 요청 자동 인식

### 🆕 깔끔한 UI
- 불필요한 디버그 메시지 제거
- 사용자 친화적 인터페이스
- 진행 상황만 간단히 표시

### 🆕 고정 5개 청크 검색
- 항상 5개의 관련 문서 청크를 추출
- 일관된 검색 결과 제공

### 🆕 다중 JSON 파싱 방식
- 코드 블록 파싱
- 일반 JSON 파싱
- 정규식 기반 파싱
- 파싱 실패 시 fallback 처리

---

## ❌ 더 이상 사용하지 않는 파일/기능

- `utils/embedding_utils.py`, `tool_search.py`, `tool_diagnose.py`, `tool_answer.py`, `make_sample_docs.py` 등은 삭제됨
- 문서는 반드시 `./docs/embedding/` 폴더에 위치해야 함

---

## 📝 기타

- 샘플 문서가 필요하면 별도 요청 시 스크립트 제공 가능
- 기타 문의/기능 요청은 이슈로 남겨주세요
- 피드백 로그는 `feedback_log.txt`에 자동 저장됩니다