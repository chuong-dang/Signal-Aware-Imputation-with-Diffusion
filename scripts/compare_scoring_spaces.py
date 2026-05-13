from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imputation_pipeline.classical import (
    classical_window_size,
    fill_knn_time,
    fill_linear_pandas,
    fill_locf_nocb,
    fill_rolling_mean,
)
from imputation_pipeline.common import get_mask_specs, load_feature_names
from imputation_pipeline.experiment import (
    MODEL_SPECS,
    build_mask,
    calc_diffusion_hyperparams,
    load_model_checkpoint,
    sample_score_model,
    sample_standard_model,
    select_plot_signal_indices,
)


def rmse(a: np.ndarray, b: np.ndarray, missing: np.ndarray) -> float:
    diff = a[missing] - b[missing]
    return float(np.sqrt(np.mean(diff * diff)))


def inverse_transform_ntf(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std[None, None, :] + mean[None, None, :]


def min_knn_prediction(masked_ntf: np.ndarray, truth_ntf: np.ndarray, missing: np.ndarray, key: str) -> np.ndarray:
    rolling_ws = classical_window_size(key)
    best_pred = None
    best_rmse = None
    for knn_k in (3, 5):
        pred = fill_knn_time(masked_ntf.copy(), k=knn_k, window_size=rolling_ws)
        cur = rmse(pred, truth_ntf, missing)
        if best_rmse is None or cur < best_rmse:
            best_rmse = cur
            best_pred = pred
    assert best_pred is not None
    return best_pred


def standardized_classical_predictions(masked_ntf: np.ndarray, truth_ntf: np.ndarray, key: str) -> dict[str, np.ndarray]:
    rolling_ws = classical_window_size(key)
    return {
        "locf_nocb": fill_locf_nocb(masked_ntf.copy()),
        "linear_interpolation_pandas": fill_linear_pandas(masked_ntf.copy()),
        "best_rolling_mean": fill_rolling_mean(masked_ntf.copy(), window_size=rolling_ws),
        "best_knn": min_knn_prediction(masked_ntf.copy(), truth_ntf, np.isnan(masked_ntf), key),
    }


def diffusion_predictions(
    standardized_ntf: np.ndarray,
    mask_ntf: np.ndarray,
    sample_length: int,
    std_run_dir: Path,
    methods: list[str],
) -> dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion_hyperparams = calc_diffusion_hyperparams(T=200, beta_0=1e-4, beta_T=0.02)
    for k, v in diffusion_hyperparams.items():
        if torch.is_tensor(v):
            diffusion_hyperparams[k] = v.to(device)
    cond = torch.from_numpy(np.transpose(standardized_ntf, (0, 2, 1))).float().to(device)
    mask = torch.from_numpy(np.transpose(mask_ntf, (0, 2, 1))).float().to(device)
    sigma_val = 3.0
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        if "standard_s4" in methods:
            model = load_model_checkpoint(
                MODEL_SPECS["standard_s4"],
                std_run_dir / "models" / "standard_s4" / "final.pt",
                standardized_ntf.shape[2],
                sample_length,
                device,
            )
            pred = sample_standard_model(model, cond, mask, diffusion_hyperparams)
            out["standard_s4"] = np.transpose(pred.detach().cpu().numpy(), (0, 2, 1))
            del model
        if any(m.startswith("score_s4_") for m in methods):
            model = load_model_checkpoint(
                MODEL_SPECS["score_s4"],
                std_run_dir / "models" / "score_s4" / "final.pt",
                standardized_ntf.shape[2],
                sample_length,
                device,
            )
            for method in methods:
                if not method.startswith("score_s4_"):
                    continue
                sampler = method.split("score_s4_", 1)[1]
                pred = sample_score_model(model, cond, mask, sigma_val, sampler_name=sampler)
                out[method] = np.transpose(pred.detach().cpu().numpy(), (0, 2, 1))
            del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def plot_method_order(diffusion_methods: list[str]) -> list[str]:
    ordered = [
        "locf_nocb",
        "linear_interpolation_pandas",
        "best_rolling_mean",
        "best_knn",
        "standard_s4",
        "score_s4_em",
        "score_s4_ode",
        "score_s4_em_long",
    ]
    return [name for name in ordered if name in {"locf_nocb", "linear_interpolation_pandas", "best_rolling_mean", "best_knn", *diffusion_methods}]


def create_plots(
    raw_test: np.ndarray,
    std_test: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    feature_names: list[str],
    all_predictions_std: dict[str, dict[str, np.ndarray]],
    plot_dir: Path,
    plot_samples: int,
    plot_signals: int,
) -> dict[str, dict]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    signal_indices = select_plot_signal_indices(std_test.shape[2], plot_signals)
    for key, preds in all_predictions_std.items():
        mask_mode, k_str = key.split("_")
        missing_k = int(k_str)
        mask_list = [build_mask(mask_mode, torch.from_numpy(sample), missing_k).numpy() for sample in std_test[:plot_samples]]
        mask_ntf = np.stack(mask_list, axis=0).astype(np.float32)
        method_names = plot_method_order(list(preds))
        summary[key] = {"samples": []}
        for sample_idx in range(min(plot_samples, len(std_test))):
            truth_std = std_test[sample_idx]
            truth_raw = raw_test[sample_idx]
            mask_tf = mask_ntf[sample_idx]
            observed_std = truth_std.copy()
            observed_std[mask_tf == 0] = np.nan
            observed_raw = truth_raw.copy()
            observed_raw[mask_tf == 0] = np.nan
            figure_payloads = [
                ("standardized", truth_std, observed_std, lambda arr: arr),
                ("inverse_raw", truth_raw, observed_raw, lambda arr: inverse_transform_ntf(arr[None, :, :], mean, std)[0]),
            ]
            for suffix, truth_tf, observed_tf, transform in figure_payloads:
                fig, axes = plt.subplots(len(signal_indices), 1, figsize=(14, 3.6 * len(signal_indices)), sharex=True)
                if len(signal_indices) == 1:
                    axes = [axes]
                time_axis = np.arange(truth_tf.shape[0])
                for ax, feat_idx in zip(axes, signal_indices):
                    ax.plot(time_axis, truth_tf[:, feat_idx], label="truth", color="black", linewidth=2.0)
                    ax.plot(time_axis, observed_tf[:, feat_idx], label="observed", color="gray", linewidth=1.2, alpha=0.9)
                    for method_name in method_names:
                        pred_tf = transform(preds[method_name][sample_idx])
                        ax.plot(time_axis, pred_tf[:, feat_idx], label=method_name, linewidth=1.0, alpha=0.9)
                    ax.set_title(f"{key} | sample {sample_idx} | {feature_names[feat_idx]} | {suffix}")
                    ax.grid(alpha=0.25)
                axes[0].legend(loc="upper right", ncol=3, fontsize=8)
                axes[-1].set_xlabel("time step")
                fig.tight_layout()
                rel_path = Path(suffix) / f"{key}_sample{sample_idx}.png"
                out_path = plot_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
                summary[key]["samples"].append(
                    {
                        "sample_idx": sample_idx,
                        "signal_indices": signal_indices,
                        "signal_names": [feature_names[i] for i in signal_indices],
                        "representation": suffix,
                        "plot_path": str(out_path),
                        "methods": method_names,
                    }
                )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset-dir", type=Path, required=True)
    parser.add_argument("--standardized-dataset-dir", type=Path, required=True)
    parser.add_argument("--standardized-run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=-1)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--plot-samples", type=int, default=2)
    parser.add_argument("--plot-signals", type=int, default=4)
    parser.add_argument(
        "--diffusion-methods",
        nargs="+",
        default=("standard_s4", "score_s4_em", "score_s4_ode", "score_s4_em_long"),
    )
    args = parser.parse_args()

    raw_test_all = np.load(args.raw_dataset_dir / "test.npy").astype(np.float32)
    std_test_all = np.load(args.standardized_dataset_dir / "test.npy").astype(np.float32)
    raw_test = raw_test_all if args.samples < 0 else raw_test_all[: args.samples]
    std_test = std_test_all if args.samples < 0 else std_test_all[: args.samples]
    mean = np.load(args.standardized_dataset_dir / "feature_mean.npy").astype(np.float32)
    std = np.load(args.standardized_dataset_dir / "feature_std.npy").astype(np.float32)
    feature_names = load_feature_names(SimpleNamespace(feature_columns_path=args.raw_dataset_dir / "feature_columns.txt"), raw_test.shape[2])

    keys = [f"{mode}_{k}" for mode, ks in get_mask_specs(std_test.shape[1]).items() for k in ks]
    diffusion_methods = list(args.diffusion_methods)
    results: dict[str, dict[str, dict[str, float]]] = {}
    all_predictions_std: dict[str, dict[str, np.ndarray]] = {}

    for key in keys:
        mask_mode, k_str = key.split("_")
        missing_k = int(k_str)
        mask_list = [build_mask(mask_mode, torch.from_numpy(sample), missing_k).numpy() for sample in std_test]
        mask_ntf = np.stack(mask_list, axis=0).astype(np.float32)
        masked_std = std_test.copy()
        masked_std[mask_ntf == 0] = np.nan

        key_results: dict[str, dict[str, float]] = {}
        classical = standardized_classical_predictions(masked_std.copy(), std_test, key)
        diffusion = diffusion_predictions(std_test, mask_ntf, std_test.shape[1], args.standardized_run_dir, diffusion_methods)
        all_predictions = {**classical, **diffusion}
        all_predictions_std[key] = all_predictions
        missing = mask_ntf == 0
        raw_truth = raw_test
        std_truth = std_test
        for method, pred_std in all_predictions.items():
            pred_raw = inverse_transform_ntf(pred_std, mean, std)
            key_results[method] = {
                "rmse_standardized_space": rmse(pred_std, std_truth, missing),
                "rmse_inverse_transformed_raw_space": rmse(pred_raw, raw_truth, missing),
            }
        results[key] = key_results

    summary = {
        "samples": int(raw_test.shape[0]),
        "keys": keys,
        "results": results,
    }
    if args.plot_dir is not None:
        plot_summary = create_plots(
            raw_test=raw_test,
            std_test=std_test,
            mean=mean,
            std=std,
            feature_names=feature_names,
            all_predictions_std=all_predictions_std,
            plot_dir=args.plot_dir,
            plot_samples=args.plot_samples,
            plot_signals=args.plot_signals,
        )
        (args.plot_dir / "plot_manifest.json").write_text(json.dumps(plot_summary, indent=2) + "\n", encoding="utf-8")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
