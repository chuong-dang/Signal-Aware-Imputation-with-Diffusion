from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from imputation_pipeline.common import DatasetSpec
from imputation_pipeline.experiment import MODEL_SPECS, SCORE_SAMPLER_STEPS, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--feature-columns-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "eval", "full"), default="full")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=("standard_s4", "score_s4"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-fraction", type=float, default=1.0)
    parser.add_argument("--max-eval-samples", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--missing-k", type=int, default=20)
    parser.add_argument("--train-missing-ks", type=int, nargs="+", default=None)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--classical-workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--amp-mode", choices=("off", "bf16", "fp16"), default="off")
    parser.add_argument("--score-samplers", nargs="+", choices=tuple(SCORE_SAMPLER_STEPS), default=("em",))
    parser.add_argument("--scaling-mode", choices=("train_standardize", "prestandardized", "none"), default="train_standardize")
    parser.add_argument("--plot-eval-samples", type=int, default=0)
    parser.add_argument("--plot-eval-signals", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = DatasetSpec(
        name=args.name,
        train_path=args.train_path,
        test_path=args.test_path,
        feature_columns_path=args.feature_columns_path,
        subset_fraction=args.subset_fraction,
        max_eval_samples=(None if args.max_eval_samples < 0 else args.max_eval_samples),
    )
    run_pipeline(
        spec=spec,
        output_root=args.output_root,
        model_names=list(args.models),
        mode=args.mode,
        seed=args.seed,
        n_iters=args.iterations,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        missing_k=(args.train_missing_ks if args.train_missing_ks is not None else args.missing_k),
        save_every=args.save_every,
        log_every=args.log_every,
        resume=args.resume,
        classical_workers=args.classical_workers,
        amp_mode=args.amp_mode,
        score_samplers=list(args.score_samplers),
        scaling_mode=args.scaling_mode,
        plot_eval_samples=args.plot_eval_samples,
        plot_eval_signals=args.plot_eval_signals,
    )


if __name__ == "__main__":
    main()
