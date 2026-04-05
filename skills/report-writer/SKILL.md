---
name: report-writer
description: 분석 결과를 보고서 파일(md/txt/docx)로 저장할 때 사용. "저장해줘", "보고서로 만들어줘", "파일로 출력해줘", "docx/md 형태로" 등의 요청에 적합.
license: MIT
allowed-tools: summarize_file analyze_file save_report
---

# Report Writer Skill

## 언제 사용하나
- 사용자가 결과물을 파일로 저장하길 원할 때
- "저장", "파일로", "보고서", "docx", "마크다운", "출력" 키워드가 포함될 때
- 분석·요약 결과를 재사용 가능한 형태로 만들어야 할 때

## 작업 순서
1. 먼저 필요한 분석/요약을 완료 (file-analysis skill 참고)
2. 보고서 구조 결정 (제목, 섹션, 내용)
3. `save_report`로 파일 저장
4. 저장된 파일 경로를 사용자에게 알림

## 보고서 구조 (기본 템플릿)
```
# [제목]

## 개요
- 작성일, 대상 파일, 목적

## 분석 내용
- 각 섹션별 내용

## 결론 및 시사점

## 부록 (필요시)
```

## 파일 형식 선택
| 요청 | 형식 |
|------|------|
| 기본 / "저장해줘" | md (마크다운) |
| "txt로" | txt |
| "docx로" / "워드로" | docx |

## 주의사항
- 저장 요청이 있으면 텍스트만 출력하지 말고 반드시 `save_report` 호출
- 저장 후 파일 경로와 파일명을 사용자에게 명시적으로 알릴 것
- 작업 흐름 예시: summarize_file → analyze_file → save_report
