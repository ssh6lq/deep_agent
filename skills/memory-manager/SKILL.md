---
name: memory-manager
description: 정보를 기억하거나 이전에 저장한 정보를 조회할 때 사용. "기억해줘", "저장해둬", "아까 말한 거", "이전에 저장한", "~를 메모해줘" 등의 요청에 적합.
license: MIT
allowed-tools: memory_save memory_recall memory_list memory_search memory_delete memory_clear
---

# Memory Manager Skill

## 언제 사용하나
- 사용자가 특정 정보를 기억/저장해달라고 요청할 때
- 이전 세션의 정보를 조회해야 할 때
- "기억", "저장", "메모", "이전에", "아까" 키워드가 포함될 때

## 도구 선택 기준

| 상황 | 사용 도구 |
|------|-----------|
| 정보 저장 | `memory_save` |
| 특정 키 조회 | `memory_recall` |
| 저장된 키 목록 확인 | `memory_list` |
| 키워드로 검색 | `memory_search` |
| 특정 항목 삭제 | `memory_delete` |
| 전체 초기화 | `memory_clear` |

## 작업 순서 — 저장
1. 저장할 정보의 키(key) 결정 (짧고 명확하게: "user_name", "project_goal")
2. `memory_save(key, value)` 호출
3. 저장 완료 확인 후 사용자에게 알림

## 작업 순서 — 조회
1. 관련 키를 모르면 먼저 `memory_list` 또는 `memory_search`로 탐색
2. 키를 알면 `memory_recall(key)`로 직접 조회
3. 결과를 사용자 컨텍스트에 맞게 해석해 제공

## 주의사항
- 중요한 사용자 정보(이름, 설정, 목표 등)는 대화 중 자동으로 저장
- 메모리는 세션 종료 후에도 유지됨 (JSON 파일 영구 보존)
- `memory_clear` 전에 반드시 사용자 확인 필요
