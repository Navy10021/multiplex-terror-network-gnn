# Ontology Contract

This document defines the ontology stack as three layers.

## A) OWL (concept and relation schema)
- File: `ontology/terror.ttl`
- Role: provides conceptual vocabulary such as Actor, Role, Relation, Event, Evidence, and Provenance.
- Guarantee: shared domain meaning for entity and relation types.

## B) SHACL (static constraints)
- File: `ontology/constraints.shacl.ttl`
- Role: enforces data-integrity constraints (range, cardinality, type).
- Guarantee: shape-level conformance (required properties, value ranges, self-loop prohibition, etc.).

## C) Runtime validator (procedural rules)
- File: `src/ontology/validator.py`
- Role: executes procedural constraints that are hard to express in SHACL, including:
  - role whitelist/compatibility
  - hierarchy source-role constraints
  - provenance consistency checks (`is_false`, `copied_from`, `confidence`)
  - temporal interaction ordering/lag constraints

## Runtime rule registry
- `src/ontology/validator.py` maps check names to rule functions via `RULE_REGISTRY`.
- Adding a new rule to the registry automatically feeds report aggregation (`violations_by_check`, `errors_by_check`).

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
- Code: `src/ontology/report_schema.py`
- Top-level fields:
  - `schema_version: str` (currently `1.1.0`)
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
- Fails the run when error/critical violations are found.
- Recommended for pre-release quality gates.

### constrained
- Attempts conformance through retry policy (`max_retries`, `seed_stride`, `retry_rule_ids`, `retry_severities`).
- By default, behaves like `strict` if retries are exhausted.
- If `fallback_report_only` is enabled, switches to report-only and preserves artifacts.

### report_only
- Records violations in reports but continues execution.
- Recommended for exploration and debugging phases.
