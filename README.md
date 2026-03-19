# WOW Emulsion Pipeline

`emulsion_bo_pipeline_v3.py`는 double emulsion 현미경 이미지를 분석하고, 누적 실험 결과를 바탕으로 다음 실험 조건을 추천하는 스크립트입니다.

## 주요 기능

- `experiments/` 아래의 실험 폴더를 자동으로 읽음
- 현미경 이미지에서 outer droplet과 inner droplet을 검출
- 실험별 droplet 측정값 CSV 저장
- 전체 실험 summary CSV 누적 관리
- 기존 결과를 바탕으로 BO(Bayesian Optimization) 추천 조건 계산
- 다음 추천 실험 조건과 BO 시각화 파일 저장

## 폴더 구조

각 실험은 `experiments/` 아래에 폴더 하나로 넣으면 됩니다.

```text
experiments/
  exp_001/
    condition.json
    microscopy/
      3_1.png
      3_2.png
      ...
    overlays/
      ...실행 후 자동 생성...
    droplets_v3.csv
```

선택 입력:

- `exp_001/` 폴더 바로 아래에 `vial.png`, `vial.jpg` 같은 vial 이미지 추가 가능

## 입력 규칙

- 폴더 이름(`exp_001`)이 자동으로 `batch_id`가 됩니다.
- `condition.json` 하나만 있으면 됩니다.
- 공정 관찰용 `process_observation.json`은 더 이상 사용하지 않습니다.
- `experiments/`에 넣은 모든 케이스는 기본적으로 `feasible = 1`로 처리됩니다.
- `um_per_pixel`을 넣지 않으면, 코드가 이미지의 빨간 scale bar를 보고 추정합니다.

## `condition.json` 예시

```json
{
  "w1_glycerol_wt": 10.0,
  "w1_tween20_wt": 2.0,
  "o_dextrin_palmitate_wt": 5.0,
  "o_span80_wt": 2.0,
  "o_adm_wt": 0.5,
  "w2_glycerol_wt": 10.0,
  "w2_pva_wt": 2.0,
  "w2_tween80_wt": 2.0,
  "q_w1_ul_min": 5.0,
  "q_o_ul_min": 10.0,
  "q_w2_ul_min": 130.0,
  "w1_pH": 7.0,
  "w2_pH": 7.0,
  "w1_conductivity_mScm": 0.05,
  "w2_conductivity_mScm": 0.05,
  "temperature_C": 25.0,
  "bath_stirring_speed_rpm": 300.0,
  "bath_carbomer_wt": 0.2,
  "time_h": 0.0,
  "operator": "BS"
}
```

## 실행 방법

```bash
source .venv/bin/activate
python emulsion_bo_pipeline_v3.py --experiments-root ./experiments
```

입력이 바뀌지 않았더라도 강제로 다시 분석하려면:

```bash
python emulsion_bo_pipeline_v3.py --experiments-root ./experiments --force-reprocess
```

기존 분석 결과만 지우고 `experiments/` 입력은 유지한 채 처음부터 다시 돌리려면:

```bash
python emulsion_bo_pipeline_v3.py --experiments-root ./experiments --reset-analysis
```

## 재실행 시 동작

코드는 각 실험 폴더에 대해 `input_fingerprint`를 계산합니다.

fingerprint 계산 대상:

- `condition.json`
- `microscopy/` 안의 이미지 파일들
- 선택적으로 존재하는 `vial.*`

동작 규칙:

- 같은 `batch_id`이고 입력도 같으면 해당 실험은 건너뜁니다.
- 같은 `batch_id`라도 `condition.json` 또는 이미지가 바뀌면 다시 분석하고 기존 row를 갱신합니다.
- 같은 입력으로 반복 실행해도 summary에 중복 행이 계속 쌓이지 않습니다.

`--reset-analysis`를 사용하면 아래 분석 산출물만 삭제한 뒤 전체 실험을 처음부터 다시 계산합니다.

- `experiment_summary_v3.csv`
- `bo_candidate_diagnostics_v3.csv`
- `bo_history_v3.csv`
- `bo_visualization_v3.png`
- 각 실험 폴더의 `droplets_v3.csv`
- 각 실험 폴더의 `overlays/`

이때 `condition.json`, `microscopy/`, 선택적인 `vial.*` 같은 입력 파일은 유지됩니다.

## 출력 파일

프로젝트 루트에 생성되는 파일:

- `experiment_summary_v3.csv`
- `bo_candidate_diagnostics_v3.csv`
- `bo_history_v3.csv`
- `bo_visualization_v3.png`

각 실험 폴더 안에 생성되는 파일:

- `droplets_v3.csv`
- `overlays/*.png`

## 파일 설명

- `experiment_summary_v3.csv`: 실험 1건당 1행으로 저장되는 전체 요약
- `droplets_v3.csv`: 검출된 droplet별 측정값
- `bo_candidate_diagnostics_v3.csv`: 현재 최종 추천된 다음 실험 조건 1행
- `bo_history_v3.csv`: BO 학습에 사용되는 이력 테이블
- `bo_visualization_v3.png`: BO 후보와 추천점을 보여주는 시각화

## Overlay 표시

- outer droplet 경계가 원본 현미경 이미지 위에 표시됩니다.
- inner droplet이 검출되면 같은 색으로 내부 경계도 함께 표시됩니다.
- inner droplet의 표시와 내부 물리량 계산은 ellipse fit 기반입니다.

## 주요 의존성

기존 `.venv` 환경 기준으로 아래 패키지가 필요합니다.

- `numpy`
- `pandas`
- `opencv-python`
- `scipy`
- `scikit-image`
- `scikit-learn`
- `scikit-optimize`
- `matplotlib`
