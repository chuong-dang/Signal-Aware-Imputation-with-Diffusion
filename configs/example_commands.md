# Example Commands

Public ETTm1, sample length 100:

```bash
python scripts/run_imputation_experiment.py \
  --name ETTm1_sample100 \
  --train-path data/cutouts/ETTm1/sample_length_100/training_sets/ETTm1_train_data.npy \
  --test-path data/cutouts/ETTm1/sample_length_100/test_sets/ETTm1_test_data.npy \
  --feature-columns-path data/cutouts/ETTm1/sample_length_100/feature_columns.txt \
  --output-root runs \
  --mode full \
  --models standard_s4 score_s4 \
  --iterations 25000 \
  --batch-size 32 \
  --train-missing-ks 10 40 70 \
  --score-samplers em ode em_long \
  --scaling-mode train_standardize \
  --plot-eval-samples 2 \
  --plot-eval-signals 4
```

Public Traffic, sample length 500:

```bash
python scripts/run_imputation_experiment.py \
  --name traffic_sample500 \
  --train-path data/cutouts/traffic/sample_length_500/training_sets/traffic_train_data.npy \
  --test-path data/cutouts/traffic/sample_length_500/test_sets/traffic_test_data.npy \
  --feature-columns-path data/cutouts/traffic/sample_length_500/feature_columns.txt \
  --output-root runs \
  --mode full \
  --models standard_s4 score_s4 \
  --iterations 25000 \
  --batch-size 32 \
  --train-missing-ks 20 120 220 320 \
  --score-samplers em ode em_long \
  --scaling-mode train_standardize
```

Vehicle runs require private data. The same runner is used after local preprocessing and pooling.
