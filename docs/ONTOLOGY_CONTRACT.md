# ONTOLOGY CONTRACT

이 문서는 ontology stack을 다음 3계층으로 분리해 정의한다.

## A) OWL (개념/관계 스키마)
- 파일: `ontology/terror.ttl`
- 역할: Actor/Role/Relation/Event/Evidence/Provenance 같은 개념적 분류 체계 제공
- 보장: 도메인 vocabulary와 관계 타입의 공통 의미

## B) SHACL (정적 제약)
- 파일: `ontology/constraints.shacl.ttl`
- 역할: 데이터 무결성 제약(범위/카디널리티/타입)
- 보장: shape 수준 정합성(필수 속성, 값 범위, self-loop 금지 등)

## C) Runtime Validator (절차적 규칙)
- 파일: `src/ontology/validator.py`
- 역할: SHACL로 표현이 까다로운 절차적 규칙 수행
  - role whitelist/compatibility
  - hierarchy source role 제약
  - provenance 일관성 검사 (`is_false`, `copied_from`, `confidence`)
  - temporal interaction ordering/lag 제약

## Runtime rule registry

- `src/ontology/validator.py`는 `RULE_REGISTRY`를 통해 check 이름과 실행 함수를 매핑한다.
- 새로운 규칙은 registry에 함수만 추가하면 report 집계(`violations_by_check`, `errors_by_check`)에 자동 반영된다.

## Rule catalog

| rule_id | severity | message template | affected_ids |
| --- | --- | --- | --- |
| roles.allowed | error | `node {id} has unknown role` | `[node_id]` |
| hierarchy.no_self_loop | error | `hierarchy edge {idx} is self-loop` | `[source,target]` |
| hierarchy.command_source_role | error | `hierarchy source role must be command role` | `[source]` |
| finance.amount_numeric | error | `invalid txn_amount_sum` | `[source,target]` |
| finance.amount_positive | error | `txn_amount_sum must be > 0` | `[source,target]` |
| events.type_allowed | error | `unsupported event_type` | `[u,v,event_type]` |
| events.time_non_negative | error | `event time must be non-negative` | `[u,v,time]` |
| provenance.is_false_binary | error | `is_false must be 0/1` | `[source,target]` |
| provenance.copied_from_known_layer | error | `copied_from must be known layer` | `[source,target,copied_from]` |
| provenance.confidence_range | error | `confidence must be in [0,1]` | `[source,target]` |
| interaction[*].ordering | error | `operation event occurs before precursor layer` | `[from_layer,to_layer]` |
| interaction[*].max_lag | error | `lag exceeds temporal_window_days` | `[from_layer,to_layer]` |

## Validator report JSON schema (fixed)
- 코드: `src/ontology/report_schema.py`
- 최상위 필드:
  - `schema_version: str` (현재 `1.1.0`)
  - `conforms: bool`
  - `constraints_checked: int`
  - `errors: list[str]`
  - `errors_by_check: dict[str, list[str]]`
  - `violations: list[{check, rule_id, message, severity, affected_ids}]`
  - `violations_by_check: dict[str, list[violation]]`
  - `violation_histogram: dict[str, int]`
  - `assets: dict[str, str]`
  - `counts: {nodes, layers, events, violations_total, violations_error}`

## Mode semantics

### strict
- 위반(error/critical)이 있으면 실행 실패
- 사용처: 실험/배포 전 품질 게이트

### constrained
- 재시도 정책(`max_retries`, `seed_stride`, `retry_rule_ids`, `retry_severities`)으로 conformance를 시도
- 실패 시 기본은 strict와 동일하게 실패
- 단, `fallback_report_only`가 활성화되면 report_only로 전환해 산출물 유지

### report_only
- 위반을 보고서로 남기고 실행 지속
- 사용처: 탐색/디버깅 단계에서 아티팩트 확보
