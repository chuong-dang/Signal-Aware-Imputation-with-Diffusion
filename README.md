# Reproducibility Package

This folder contains the code used for the imputation experiments reported in the paper draft. It is intended as a compact public release: public benchmark experiments can be reproduced directly after downloading the datasets, while vehicle experiments are documented and runnable only with access to the private vehicle telemetry data.

## Scope

Included:
- public dataset preprocessing for ETT, Electricity, Traffic, Weather, Exchange, Solar, and PEMS-style files
- vehicle preprocessing hooks for local/private parquet trip data
- S4 and score-based S4 imputation runners
- thesis-style classical baselines: LOCF/NOCB, linear interpolation, rolling mean, and temporal KNN
- scripts for dual-space and per-signal analysis
- lightweight summary of the preserved experiment results

Not included:
- private vehicle telemetry
- generated cutout arrays, checkpoints, masks, plots, or caches
- notebooks and exploratory local launch files

## Layout

```text
paper_reproducibility/
  ImputationMaster/SSSD/src/     # minimal SSSD/S4 fork used by the runner
  imputation_pipeline/           # reusable experiment, classical, and data helpers
  scripts/                       # preprocessing, training/evaluation, analysis
  configs/                       # command templates
  results/experiment_summary.json
  requirements.txt
```

## Installation

Use Python 3.11 or 3.12 with CUDA-enabled PyTorch if running diffusion models on GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements pin PyTorch to a CUDA 12.4-compatible build. If a machine exports an older CUDA path through `LD_LIBRARY_PATH`, clear or correct it before running the scripts; otherwise PyTorch may load incompatible system CUDA libraries.

Optional S4 CUDA extension:

```bash
cd ImputationMaster/SSSD/src/entensions/cauchy
pip install .
```

The extension is optional; the S4 code falls back to a slower PyTorch path if it is unavailable.

## Public Datasets

Public datasets are not redistributed in this repository. Download them from their upstream benchmark sources, then place the files under a local root such as:

- ETT: https://github.com/zhouhaoyi/ETDataset
- Electricity, Traffic, Weather, Exchange, Solar, PEMS-style benchmarks: https://github.com/thuml/Time-Series-Library

```text
data/time_series_datasets/
  ETT-small/ETTm1.csv
  ETT-small/ETTm2.csv
  ETT-small/ETTh1.csv
  ETT-small/ETTh2.csv
  electricity/electricity.csv
  traffic/traffic.csv
  weather/weather.csv
  exchange_rate/exchange_rate.csv
  Solar/solar_AL.txt
  PEMS/PEMS03.npz
  ...
```

Generate cutouts:

```bash
python scripts/generate_cutout_datasets.py \
  --source public \
  --public-root data/time_series_datasets \
  --output-root data/cutouts
```

Example ETTm1 run:

```bash
python scripts/run_imputation_experiment.py \
  --name ETTm1_sample500 \
  --train-path data/cutouts/ETTm1/sample_length_500/training_sets/ETTm1_train_data.npy \
  --test-path data/cutouts/ETTm1/sample_length_500/test_sets/ETTm1_test_data.npy \
  --feature-columns-path data/cutouts/ETTm1/sample_length_500/feature_columns.txt \
  --output-root runs \
  --mode full \
  --models standard_s4 score_s4 \
  --iterations 25000 \
  --batch-size 32 \
  --train-missing-ks 20 120 220 320 \
  --score-samplers em ode em_long \
  --scaling-mode train_standardize
```

For `sample_length=100`, use `--train-missing-ks 10 40 70`.

## Vehicle Datasets

Vehicle data is private and is not distributed. To reproduce the vehicle experiments, provide a local folder of trip-level parquet files with timestamp columns and numeric signals. The preprocessing script supports 1 Hz resampling:

```bash
python scripts/generate_cutout_datasets.py \
  --source vehicle \
  --vehicle-root /path/to/private/trips \
  --vehicle-resample-seconds 1 \
  --include-vehicle-prefixes V55 V56 V57 V58 V59 \
  --output-root data/vehicle_cutouts_1hz
```

Pool vehicle cutouts:

```bash
python scripts/build_pooled_vehicle_dataset.py \
  --datasets-root data/vehicle_cutouts_1hz \
  --output-dir data/vehicle_V55_V59_pool_sample500_10pct_1hz \
  --sample-length 500 \
  --subset-fraction 0.1 \
  --include-datasets vehicle_V55 vehicle_V56 vehicle_V57 vehicle_V58 vehicle_V59
```

The paper experiments used a thesis-compatible signal subset for selected vehicle analyses. Because private signal names vary by fleet generation, the feature mapping should be adapted to the local schema before rerunning those experiments.

## Metrics

The main runner reports RMSE on standardized data by default. Additional scripts support:

- inverse-transformed raw-space scoring
- normalized raw-space scoring
- per-signal RMSE
- small diagnostic plot bundles

Use `scripts/compare_scoring_spaces.py` and `scripts/export_per_signal_dualspace_rmse.py` after checkpoints have been produced.

## Results

`results/experiment_summary.json` contains lightweight summaries of the preserved runs used to guide the paper narrative. It does not contain model checkpoints or raw data.

## Notes

The public release focuses on the S4 and score-based S4 models used in the reported comparisons. Mamba experiments were exploratory and are not part of this reproducibility package.
