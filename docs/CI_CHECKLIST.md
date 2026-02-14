# CI Checklist (Reusable PR Gate)

이 문서는 PR마다 반복 적용할 **최종 점검 기준**을 고정한 체크리스트다.
CI는 `scripts/ci_checklist.sh`를 실행해 아래 기준을 자동 검증한다.

## 1) Static checks
- [ ] `ruff check src tests`
- [ ] `mypy src/ontology src/validation src/cli src/utils`

## 2) Test suite
- [ ] `pytest -q`

## 3) CLI health checks
- [ ] `python -m src.cli.main --help`
- [ ] `python -m src.cli.main run-all --help`

## 4) Repro script sanity
- [ ] `bash -n scripts/run_easy_baseline_hard.sh scripts/summarize_all.sh`

## 5) Docs alignment review (manual)
- [ ] README Quick Start / CLI / Typical outputs가 현재 코드 동작과 일치
- [ ] 새로 추가된 사용자-facing 기능이 docs에 반영

---

## Notes
- Torch/PyG는 환경 의존성이 크므로 설치 실패 시 CI 전체를 깨지 않도록 테스트는 `importorskip` 패턴을 사용한다.
- 체크리스트 변경이 필요하면 `docs/CI_CHECKLIST.md`와 `scripts/ci_checklist.sh`를 함께 업데이트한다.
- CI job은 `timeout-minutes: 20`, `pip --retries 1 --timeout 30`로 설정해 무한 대기/장기 재시도를 줄인다.
- PR이 오래 대기하면 브랜치를 최신 `main`으로 rebase 후 다시 push해 mergeability 계산을 강제 갱신한다.
