---
name: file-analysis
description: 파일을 요약·번역·분석·핵심 포인트 추출할 때 사용. "요약해줘", "번역해줘", "분석해줘", "핵심만 뽑아줘", PDF/docx/txt 파일 관련 요청에 적합.
license: MIT
allowed-tools: summarize_file translate_file extract_key_points analyze_file load_file_content
---

# File Analysis Skill

## 언제 사용하나
- 사용자가 파일 경로를 제공하며 처리를 요청할 때
- "요약", "번역", "분석", "핵심", "검토" 키워드와 파일이 함께 언급될 때
- 첨부 파일에 대한 질문이나 처리 요청

## 도구 선택 기준

| 요청 유형 | 사용 도구 |
|-----------|-----------|
| 파일 요약 | `summarize_file` |
| 파일 번역 | `translate_file` |
| 핵심 포인트 추출 | `extract_key_points` |
| 커스텀 분석·검토 | `analyze_file` |
| 파일 내용 직접 열람 | `load_file_content` |

## 작업 순서
1. 파일 경로와 요청 유형 파악
2. 위 기준표에 따라 적절한 도구 선택
3. 여러 파일인 경우 각각 처리 후 비교
4. 결과를 사용자 요청에 맞게 정리

## 여러 파일 처리 시
- 각 파일을 개별 처리 후 결과를 통합
- 파일 간 공통점/차이점 강조
- 처리 순서: 개별 분석 → 비교 → 종합

## 주의사항
- 파일 경로가 제공되면 반드시 해당 도구를 사용 (텍스트만 출력 금지)
- 파일 형식을 먼저 확인하고 적절한 도구 선택
