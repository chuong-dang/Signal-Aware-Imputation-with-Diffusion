from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_scoring_spaces import (
    diffusion_predictions,
    inverse_transform_ntf,
    standardized_classical_predictions,
)
from imputation_pipeline.common import get_mask_specs, load_feature_names
from imputation_pipeline.experiment import build_mask


def rmse_per_feature(pred_ntf: np.ndarray, truth_ntf: np.ndarray, missing_ntf: np.ndarray) -> list[float]:
    out: list[float] = []
    feature_count = truth_ntf.shape[2]
    for feat_idx in range(feature_count):
        feat_missing = missing_ntf[:, :, feat_idx]
        count = int(np.sum(feat_missing))
        if count == 0:
            out.append(float("nan"))
            continue
        diff = pred_ntf[:, :, feat_idx][feat_missing] - truth_ntf[:, :, feat_idx][feat_missing]
        out.append(float(np.sqrt(np.mean(diff * diff))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset-dir", type=Path, required=True)
    parser.add_argument("--standardized-dataset-dir", type=Path, required=True)
    parser.add_argument("--standardized-run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--diffusion-methods",
        nargs="+",
        default=("standard_s4", "score_s4_em", "score_s4_ode", "score_s4_em_long"),
    )
    args = parser.parse_args()

    raw_test = np.load(args.raw_dataset_dir / "test.npy").astype(np.float32)
    std_test = np.load(args.standardized_dataset_dir / "test.npy").astype(np.float32)
    mean = np.load(args.standardized_dataset_dir / "feature_mean.npy").astype(np.float32)
    std = np.load(args.standardized_dataset_dir / "feature_std.npy").astype(np.float32)
    feature_names = load_feature_names(SimpleNamespace(feature_columns_path=args.raw_dataset_dir / "feature_columns.txt"), raw_test.shape[2])

    keys = [f"{mode}_{k}" for mode, ks in get_mask_specs(std_test.shape[1]).items() for k in ks]
    diffusion_methods = list(args.diffusion_methods)
    output: dict[str, object] = {
        "samples": int(raw_test.shape[0]),
        "feature_names": feature_names,
        "keys": {},
    }

    for key in keys:
        mask_mode, k_str = key.split("_")
        missing_k = int(k_str)
        mask_list = [build_mask(mask_mode, torch.from_numpy(sample), missing_k).numpy() for sample in std_test]
        mask_ntf = np.stack(mask_list, axis=0).astype(np.float32)
        masked_std = std_test.copy()
        masked_std[mask_ntf == 0] = np.nan

        classical = standardized_classical_predictions(masked_std.copy(), std_test, key)
        diffusion = diffusion_predictions(std_test, mask_ntf, std_test.shape[1], args.standardized_run_dir, diffusion_methods)
        all_predictions = {**classical, **diffusion}
        missing = mask_ntf == 0

        key_payload: dict[str, object] = {
            "missing_counts_per_feature": [int(np.sum(missing[:, :, feat_idx])) for feat_idx in range(missing.shape[2])],
            "methods": {},
        }
        for method, pred_std in all_predictions.items():
            pred_raw = inverse_transform_ntf(pred_std, mean, std)
            std_per_feature = rmse_per_feature(pred_std, std_test, missing)
            raw_per_feature = rmse_per_feature(pred_raw, raw_test, missing)
            key_payload["methods"][method] = {
                "rmse_standardized_space_per_feature": std_per_feature,
                "rmse_inverse_transformed_raw_space_per_feature": raw_per_feature,
            }
        output["keys"][key] = key_payload

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
