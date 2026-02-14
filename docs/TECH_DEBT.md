# TECH_DEBT 진단 리포트 (STEP 0)

작성일: 2026-02-14  
범위: 레포 전체(`src/`, `tests/`, `configs/`, `ontology/`, 루트 문서/의존성 파일)

## 스캔 방법

아래 커맨드로 스냅샷 진단을 수행했다.

- `rg --files`
- `python - <<'PY' ...` (한 줄 파일 탐지: `.py/.md/.json`)
- `find tests -maxdepth 1 -type f -name 'test_*.py'`
- `pytest -q`
- `sed -n '1,220p' requirements.txt`
- `sed -n '1,260p' README.md`
- `sed -n '1,260p' src/run_all.py`

---

## STEP 1~3 진행 반영

- 완료: STEP 1 (툴링 표준화)
  - `pyproject.toml` 추가 (ruff/black/isort/mypy 설정)
  - `.pre-commit-config.yaml` 추가
  - 한 줄 Python 파일(`src/__init__.py`, `src/cli/__init__.py`, `src/utils/__init__.py`) 멀티라인 정리
- 완료: STEP 2 (CI)
  - `.github/workflows/ci.yml` 추가
  - Python 3.10/3.11 matrix + `ruff`, `mypy`, `pytest -q`
- 완료: STEP 3 (의존성 재현성)
  - `requirements.txt`를 최소 범위 지정 중심으로 재구성
  - `requirements.lock` 신설 (OS/CUDA 독립 의존성만 고정)
  - `docs/INSTALL.md` 신설 (CPU/CUDA/Colab 설치 레시피)

---

## A. 코드 스타일/포맷 문제 (특히 한 줄 파일)

### A-1) 포맷터/린터 설정 부재
- **진단**: `pyproject.toml`이 없어 `ruff/black/isort/mypy` 기준이 팀 차원에서 고정되어 있지 않다.
- **영향도**: 높음 (리뷰 비용 증가, 코드 일관성 붕괴)
- **난이도**: 낮음
- **권장 해결 순서**: 1
- **권장 조치**:
  1. `pyproject.toml`에 `black`, `ruff`, `isort`, `mypy` 설정 통합
  2. import order, line length, target-version 명시

### A-2) pre-commit 훅 부재
- **진단**: 커밋 전 자동 품질 게이트(`pre-commit`)가 없다.
- **영향도**: 높음 (스타일/정적분석 회귀 발생 가능)
- **난이도**: 낮음
- **권장 해결 순서**: 2
- **권장 조치**: `.pre-commit-config.yaml` 도입 + CI에서도 동일 훅 실행

### A-3) 한 줄 Python 파일 3개 존재
- **진단**: `src/__init__.py`, `src/cli/__init__.py`, `src/utils/__init__.py`가 한 줄 파일이다.
- **영향도**: 낮음 (기능 영향 작음) / 중간 (가독성·문서화 관점)
- **난이도**: 낮음
- **권장 해결 순서**: 8
- **권장 조치**: 멀티라인 docstring, `__all__` 선언, 패키지 설명 주석 추가

### A-4) 버전 suffix 파일 난립으로 스타일 드리프트 위험
- **진단**: `*_v1.py`, `*_v2.py`, `*_v3.py`가 공존해 포맷/규약이 분산될 가능성이 높다.
- **영향도**: 중간
- **난이도**: 중간
- **권장 해결 순서**: 9
- **권장 조치**: 활성 버전 명시 + 레거시 격리(`legacy/`) + 공통 모듈 추출

---

## B. import 구조

### B-1) `src.` 절대 import 고정으로 실행 경로 민감
- **진단**: `src/run_all.py` 등에서 `from src...` import를 폭넓게 사용한다.
- **영향도**: 중간 (환경에 따라 `PYTHONPATH`/실행 방식 이슈)
- **난이도**: 중간
- **권장 해결 순서**: 6
- **권장 조치**: 패키지 설치 전제(`pip install -e .`) + 내부 상대 import/엔트리포인트 정리

### B-2) 모듈 경계가 기능축보다 버전축(v1/v2/v3)에 치우침
- **진단**: import 대상이 버전별 파일로 분기되어 공통 API 계약이 약하다.
- **영향도**: 중간
- **난이도**: 중간
- **권장 해결 순서**: 10
- **권장 조치**: `src/data`, `src/models`에 stable interface 도입 (예: `current.py` façade)

### B-3) `__init__.py` export 정책 미흡
- **진단**: 패키지 경계에서 공개 API(`__all__`)가 사실상 정의되지 않았다.
- **영향도**: 중간
- **난이도**: 낮음
- **권장 해결 순서**: 11
- **권장 조치**: 패키지별 public symbol 명시, 내부 구현 import 금지 규약 수립

---

## C. 실행 진입점(entrypoint)

### C-1) CLI 진입점이 `python -m src.run_all`에 편중
- **진단**: README 기준 표준 실행이 모듈 경로 실행에 의존한다.
- **영향도**: 중간
- **난이도**: 중간
- **권장 해결 순서**: 7
- **권장 조치**: console script(`multiplex-gnn`) 제공, subcommand(`run-all`, `validate-ontology`) 통합

### C-2) 파이프라인 단일 파일(`src/run_all.py`) 비대화
- **진단**: 데이터 생성/검증/리포팅/설명 생성이 한 파일에 밀집되어 결합도가 높다.
- **영향도**: 높음 (변경 영향 범위 확대)
- **난이도**: 중간~높음
- **권장 해결 순서**: 12
- **권장 조치**: orchestration/service/helper 계층 분리 + 각 단계 함수 단위 테스트 보강

### C-3) 실행 모드 계약 문서화 부족
- **진단**: strict/constrained/report_only는 존재하지만 “언제 어떤 모드를 선택해야 하는지” 운영 가이드가 약하다.
- **영향도**: 중간
- **난이도**: 낮음
- **권장 해결 순서**: 13
- **권장 조치**: `docs/ONTOLOGY_CONTRACT.md`에 모드별 보장/트레이드오프 표준화

---

## D. 테스트 커버리지

### D-1) 테스트는 존재하나 커버리지 측정 미도입
- **진단**: `tests/`는 9개 파일로 구성되어 있고 `pytest -q`는 통과하지만, coverage gate가 없다.
- **영향도**: 높음 (사각지대 파악 불가)
- **난이도**: 낮음
- **권장 해결 순서**: 4
- **권장 조치**: `pytest --cov=src --cov-report=term-missing` 도입 + 최소 기준 설정

### D-2) CI 파이프라인 부재로 회귀 자동 탐지 불가
- **진단**: `.github/workflows/ci.yml`이 없다.
- **영향도**: 매우 높음
- **난이도**: 낮음
- **권장 해결 순서**: 3
- **권장 조치**: Python 3.10/3.11 matrix에서 `ruff + mypy + pytest` 실행

### D-3) 스키마 계약 테스트가 파일명상 분산/불명확
- **진단**: `test_manifest_validation.py`는 있으나, manifest contract를 체계적으로 문서-테스트 매핑한 흔적이 약하다.
- **영향도**: 중간
- **난이도**: 중간
- **권장 해결 순서**: 14
- **권장 조치**: `tests/test_manifest_schema.py`로 계약 테스트를 독립시키고 정상/실패 케이스 6+ 확보

### D-4) 경고(warning) 관리 정책 부재
- **진단**: 현재 pytest 실행 시 matplotlib/pyparsing deprecation warning 13건 발생.
- **영향도**: 중간 (노이즈 누적으로 실제 경고 묻힘)
- **난이도**: 낮음
- **권장 해결 순서**: 15
- **권장 조치**: `filterwarnings` 정책 수립 + 의존성 상향/핀 조정

---

## E. 의존성 충돌/재현성 위험

### E-1) README는 `requirements.lock` 설치를 안내하지만 파일 부재
- **진단**: 설치 문서와 실제 파일이 불일치한다.
- **영향도**: 높음 (온보딩 실패 가능)
- **난이도**: 낮음
- **권장 해결 순서**: 5
- **권장 조치**: `requirements.txt`/`requirements.lock` 전략 확정 + README 정합화

### E-2) `requirements.txt`가 과도한 핀 고정 + GPU 민감 패키지 포함
- **진단**: `torch==2.2.2`, `torch-geometric==2.5.3`가 환경별 wheel 차이를 고려하지 않고 단일 핀으로 명시돼 있다.
- **영향도**: 높음
- **난이도**: 중간
- **권장 해결 순서**: 6
- **권장 조치**: Torch/PyG는 문서 가이드 분리, 공통 최소 의존성만 lock

### E-3) 표준 라이브러리 `argparse`를 pip 의존성으로 선언
- **진단**: `argparse==1.4.0`는 Python stdlib와 충돌/혼동 가능성이 있다.
- **영향도**: 중간
- **난이도**: 낮음
- **권장 해결 순서**: 6
- **권장 조치**: requirements에서 제거

### E-4) 패키징 메타데이터 부재
- **진단**: `pip install -e .` 기반 개발 워크플로우를 위한 패키지 메타 설정이 없다.
- **영향도**: 중간
- **난이도**: 중간
- **권장 해결 순서**: 10
- **권장 조치**: `pyproject.toml`에 프로젝트 메타 + optional deps(cpu/cuda 문서연동) 정리

### E-5) 라이선스 상태 TBD
- **진단**: README에 라이선스가 미확정(TBD)으로 기재되어 배포/협업 리스크가 있다.
- **영향도**: 중간~높음 (법적/조달 이슈)
- **난이도**: 낮음
- **권장 해결 순서**: 16
- **권장 조치**: 명시적 LICENSE 추가 및 서드파티 라이선스 표기

---

## 우선순위 실행 로드맵 (권장)

1. **품질 게이트 기반 공사**: A-1, A-2, D-2, D-1
2. **설치/의존성 정합화**: E-1, E-2, E-3
3. **실행 인터페이스 안정화**: C-1, B-1
4. **계약 테스트 강화**: D-3, C-3
5. **구조적 리팩터링**: C-2, B-2, B-3, A-4
6. **운영/거버넌스 마감**: D-4, E-4, E-5, A-3

---

## 총평

- 현 상태는 **기능 구현은 상당히 진행**되었고 테스트도 통과한다.
- 다만, **표준화(포맷/린트/타입/CI) + 의존성 재현성 + 패키징/엔트리포인트**가 비어 있어, 확장 단계에서 기술부채가 급격히 커질 위험이 있다.
- 따라서 사용자 제안 로드맵 중 **STEP 1~3을 최우선**으로 착수하는 것이 가장 ROI가 높다.
