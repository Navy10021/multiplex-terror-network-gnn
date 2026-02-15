# STEP 11 Implementation Plan (Detailed)

## Goal
- Break data/ontology/graph-model improvements into small, incremental tickets.
- Define file scope and test criteria per ticket.

---

## STEP11-1
### Title
Add a generator config validation layer

### Files
- `src/data/multiplex_generator_v3.py`
- `tests/test_generator_config_validation.py` (new)

### Work
- Add `validate_generator_config(cfg)`:
  - enforce `[0,1]` bounds for probability/rate parameters
  - validate min/max relationships for event settings
  - validate positive values for `size`, `num_days`, `campaign_count`, etc.
  - validate `cross_layer_copy` structure and rate range
- call validator from `load_generator_config` and default config load paths

### Done criteria
- invalid configs fail with clear error messages
- valid configs pass

### Test
- `pytest -q tests/test_generator_config_validation.py`

---

## STEP11-2
### Title
Ontology rule registry externalization + report schema versioning

### Files
- `src/ontology/validator.py`
- `src/ontology/report_schema.py`
- `docs/ONTOLOGY_CONTRACT.md`
- `tests/test_ontology_report_schema.py`

---

## STEP11-3
### Title
Link prediction protocol v2 (temporal split + advanced negatives)

### Files
- `src/models/train_linkpred_layer_v3.py`
- `tests/test_linkpred_protocol_v2.py` (new)
- `README.md`

### Work
- add `split_mode` (`random` / `temporal`) to link prediction split protocol
- use timestamp-based temporal splits when timestamps exist (fallback to random otherwise)
- add advanced negative sampling modes (`degree`, `hybrid`)
- add protocol v2 unit tests

### Done criteria
- temporal split path is covered by tests
- advanced negative sampling satisfies base constraints (avoid self-loops and true edges)

### Test
- `pytest -q tests/test_linkpred_protocol.py tests/test_linkpred_protocol_v2.py`

---

## STEP11-4
### Title
Auto-reflect model calibration summary in MODEL_CARD

### Files
- `src/models/train_multitask_gnn_v3.py`
- `src/reporting/cards.py`
- `tests/test_reporting_cards.py`

### Work
- include `hvt_calibration` summary from multitask metrics in MODEL_CARD
- report before/after calibration values (ECE, Brier), temperature, and threshold strategy/distribution (IQR)
- strengthen reporting card tests

### Done criteria
- MODEL_CARD prints calibration summary metrics
- tests validate expected calibration text/metrics

### Test
- `pytest -q tests/test_reporting_cards.py tests/test_reporting_phase_e.py`
