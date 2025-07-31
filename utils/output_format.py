# output_format.py
"""
출력 포맷팅, 중복 제거 등 답변 가공 모듈
"""
def format_answer(answer: str) -> str:
    # 이모지, 구분선, 줄바꿈 등 가독성 개선
    return f"\n====================\n{answer}\n====================\n"

def remove_duplicates(text: str) -> str:
    # 중복 문장/문단 제거 로직 (간단 예시)
    lines = text.split('\n')
    seen = set()
    result = []
    for line in lines:
        if line.strip() and line not in seen:
            seen.add(line)
            result.append(line)
    return '\n'.join(result)

def remove_duplicate_expressions(text: str) -> str:
    # 기존 중복 표현 제거 함수 (rag_langgraph_cli.py에서 이동)
    return remove_duplicates(text)

def format_report_output(parsed: dict, mode: str, sources: list = None) -> str:
    # 각 필드가 비어 있으면 '내용 없음' 기본값 출력
    def safe(val):
        return val if val and str(val).strip() else "내용 없음"
    if not parsed:
        return "❌ 파싱된 결과가 없습니다."
    if mode == "summary":
        result = format_answer(safe(parsed.get("summary", "")))
    else:
        title = safe(parsed.get("title", ""))
        abstract = safe(parsed.get("abstract", ""))
        introduction = safe(parsed.get("introduction", ""))
        background = safe(parsed.get("background", ""))
        main_content = safe(parsed.get("main_content", ""))
        conclusion = safe(parsed.get("conclusion", ""))
        result = f"\n# {title}\n\n## 요약\n{abstract}\n\n## 서론\n{introduction}\n\n## 배경\n{background}\n\n## 본문\n{main_content}\n\n## 결론\n{conclusion}\n"
    # 출처 섹션 추가
    if sources:
        unique_sources = list(dict.fromkeys(sources))
        sources_section = '\n'.join(f"- {src}" for src in unique_sources)
        result += f"\n\n## 출처\n{sources_section}\n"
    return format_answer(result) 