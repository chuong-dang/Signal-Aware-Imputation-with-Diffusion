from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
from typing import Callable, Sequence

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from imputation_pipeline.classical import (
    classical_window_size,
    fill_knn_time,
    fill_linear_pandas,
    fill_locf_nocb,
    fill_rolling_mean,
)
from imputation_pipeline.common import (
    DatasetSpec,
    cache_paths,
    get_mask_specs,
    load_dataset,
    load_feature_names,
    load_json,
    prepare_cuda_runtime,
    prepare_data_by_scaling_mode,
    rmse_from_sum,
    save_json,
    set_seed,
    stable_seed,
    update_progress,
)
from ImputationMaster.SSSD.src.imputers.SSSDS4Imputer import SSSDS4Imputer
from ImputationMaster.SSSD.src.utils.util import (
    calc_diffusion_hyperparams,
    diffusion_coeff,
    get_mask_bm,
    get_mask_mnr,
    get_mask_rm,
    marginal_prob_std,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    objective: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "standard_s4": ModelSpec(name="standard_s4", family="s4", objective="standard"),
    "score_s4": ModelSpec(name="score_s4", family="s4", objective="score"),
}

SCORE_SAMPLER_STEPS = {
    "em": 200,
    "ode": 200,
    "em_long": 400,
}


def make_s4_config(features: int, sample_length: int) -> dict:
    return {
        "in_channels": features,
        "out_channels": features,
        "num_res_layers": 36,
        "res_channels": 256,
        "skip_channels": 256,
        "diffusion_step_embed_dim_in": 128,
        "diffusion_step_embed_dim_mid": 512,
        "diffusion_step_embed_dim_out": 512,
        "s4_lmax": sample_length,
        "s4_d_state": 64,
        "s4_dropout": 0.0,
        "s4_bidirectional": 1,
        "s4_layernorm": 1,
    }


def instantiate_model(model_spec: ModelSpec, features: int, sample_length: int) -> nn.Module:
    if model_spec.family == "s4":
        return SSSDS4Imputer(**make_s4_config(features, sample_length))
    raise ValueError(model_spec.family)


def autocast_context(device: torch.device, amp_mode: str):
    if device.type != "cuda" or amp_mode == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_training_batch(
    train_gpu: torch.Tensor,
    batch_size: int,
    missing_k: int | Sequence[int],
    mask_mode: str = "rm",
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(train_gpu.shape[0], size=(batch_size,), device=train_gpu.device)
    batch_ntf = train_gpu.index_select(0, idx)
    sample = batch_ntf[0]
    if isinstance(missing_k, Sequence) and not isinstance(missing_k, (str, bytes)):
        if len(missing_k) == 0:
            raise ValueError("missing_k sequence must not be empty")
        selected_k = int(missing_k[torch.randint(len(missing_k), size=(1,), device=train_gpu.device).item()])
    else:
        selected_k = int(missing_k)
    if mask_mode == "rm":
        mask_tf = get_mask_rm(sample, selected_k).to(train_gpu.device)
    elif mask_mode == "mnr":
        mask_tf = get_mask_mnr(sample, selected_k).to(train_gpu.device)
    elif mask_mode == "bm":
        mask_tf = get_mask_bm(sample, selected_k).to(train_gpu.device)
    else:
        raise ValueError(mask_mode)
    batch = batch_ntf.permute(0, 2, 1).contiguous()
    mask = mask_tf.permute(1, 0).unsqueeze(0).expand(batch.shape[0], -1, -1).float().contiguous()
    return batch, mask


def standard_diffusion_loss(
    model: nn.Module,
    batch: torch.Tensor,
    mask: torch.Tensor,
    diffusion_hyperparams: dict,
) -> torch.Tensor:
    device = batch.device
    bsz = batch.shape[0]
    alpha = diffusion_hyperparams["Alpha"]
    sigma = diffusion_hyperparams["Sigma"]
    diffusion_steps = torch.randint(diffusion_hyperparams["T"], size=(bsz, 1, 1), device=device)
    z = torch.randn_like(batch)
    transformed = torch.sqrt(alpha[diffusion_steps]) * batch + sigma[diffusion_steps] * z
    predicted = model((transformed, batch, mask, diffusion_steps))
    return torch.mean((predicted - z) ** 2)


def score_matching_loss(
    model: nn.Module,
    batch: torch.Tensor,
    mask: torch.Tensor,
    sigma_value: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    random_t = torch.rand(batch.shape[0], 1, 1, device=batch.device) * (1.0 - eps) + eps
    z = torch.randn_like(batch)
    std = marginal_prob_std(random_t.squeeze(-1).squeeze(-1), sigma_value, device=batch.device)
    perturbed = batch + z * std[:, None, None]
    score = model((perturbed, batch, mask, random_t))
    return torch.mean(torch.sum((score * std[:, None, None] + z) ** 2, dim=(1, 2)))


def sample_standard_model(
    model: nn.Module,
    cond: torch.Tensor,
    mask: torch.Tensor,
    diffusion_hyperparams: dict,
) -> torch.Tensor:
    alpha = diffusion_hyperparams["Alpha"]
    alpha_bar = diffusion_hyperparams["Alpha_bar"]
    sigma = diffusion_hyperparams["Sigma"]
    x = torch.randn_like(cond)
    with torch.no_grad():
        for t in range(diffusion_hyperparams["T"] - 1, -1, -1):
            x = x * (1 - mask) + cond * mask
            diffusion_steps = torch.full((cond.shape[0], 1, 1), float(t), device=cond.device)
            eps_theta = model((x, cond, mask, diffusion_steps))
            x = (x - (1 - alpha[t]) / torch.sqrt(1 - alpha_bar[t]) * eps_theta) / torch.sqrt(alpha[t])
            if t > 0:
                x = x + sigma[t] * torch.randn_like(x)
    return x * (1 - mask) + cond * mask


def sample_score_model_em(
    model: nn.Module,
    cond: torch.Tensor,
    mask: torch.Tensor,
    sigma_value: float,
    num_steps: int = 200,
    eps: float = 1e-3,
) -> torch.Tensor:
    x = torch.randn_like(cond)
    x = x * (1 - mask) + cond * mask
    batch_size = cond.shape[0]
    time_steps = torch.linspace(1.0, eps, num_steps, device=cond.device)
    step_size = time_steps[0] - time_steps[1]
    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, 1, 1, device=cond.device) * time_step
            g = diffusion_coeff(time_step, sigma_value, device=cond.device)
            score = model((x, cond, mask, batch_time_step))
            x_mean = x + (g**2) * score * step_size
            x = x_mean + torch.sqrt(step_size) * g * torch.randn_like(x)
            x = x * (1 - mask) + cond * mask
    return x_mean * (1 - mask) + cond * mask


def sample_score_model_ode(
    model: nn.Module,
    cond: torch.Tensor,
    mask: torch.Tensor,
    sigma_value: float,
    num_steps: int = 200,
    eps: float = 1e-3,
) -> torch.Tensor:
    x = torch.randn_like(cond)
    x = x * (1 - mask) + cond * mask
    batch_size = cond.shape[0]
    time_steps = torch.linspace(1.0, eps, num_steps, device=cond.device)
    step_size = time_steps[0] - time_steps[1]
    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, 1, 1, device=cond.device) * time_step
            g = diffusion_coeff(time_step, sigma_value, device=cond.device)
            score = model((x, cond, mask, batch_time_step))
            x_mean = x + 0.5 * (g**2) * score * step_size
            x = x_mean * (1 - mask) + cond * mask
    return x * (1 - mask) + cond * mask


def sample_score_model(
    model: nn.Module,
    cond: torch.Tensor,
    mask: torch.Tensor,
    sigma_value: float,
    sampler_name: str = "em",
) -> torch.Tensor:
    if sampler_name == "em":
        return sample_score_model_em(model, cond, mask, sigma_value, num_steps=SCORE_SAMPLER_STEPS["em"])
    if sampler_name == "em_long":
        return sample_score_model_em(model, cond, mask, sigma_value, num_steps=SCORE_SAMPLER_STEPS["em_long"])
    if sampler_name == "ode":
        return sample_score_model_ode(model, cond, mask, sigma_value, num_steps=SCORE_SAMPLER_STEPS["ode"])
    raise ValueError(f"Unknown score sampler: {sampler_name}")


def build_mask(mask_mode: str, sample_ntf: torch.Tensor, missing_k: int) -> torch.Tensor:
    if mask_mode == "rm":
        return get_mask_rm(sample_ntf, missing_k)
    if mask_mode == "mnr":
        return get_mask_mnr(sample_ntf, missing_k)
    if mask_mode == "bm":
        return get_mask_bm(sample_ntf, missing_k)
    raise ValueError(mask_mode)


def ensure_mask_cache(
    test_x: np.ndarray,
    sample_length: int,
    seed: int,
    cache_dir: Path,
) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    mask_paths: dict[str, Path] = {}
    test_tensor = torch.from_numpy(test_x)
    for mask_mode, ks in get_mask_specs(sample_length).items():
        for k in ks:
            key = f"{mask_mode}_{k}"
            out_path = cache_dir / f"{key}.npy"
            mask_paths[key] = out_path
            if out_path.exists():
                continue
            key_seed = stable_seed(seed, key)
            set_seed(key_seed)
            print(f"[mask-cache] building {key}", flush=True)
            mask_list = [build_mask(mask_mode, sample, k).numpy() for sample in test_tensor]
            np.save(out_path, np.stack(mask_list, axis=0).astype(np.float32))
    return mask_paths


def prepare_standardized_tensors(test_x: np.ndarray, cache_dir: Path) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ntf_path = cache_dir / "test_ntf.npy"
    nft_path = cache_dir / "test_nft.npy"
    if not ntf_path.exists():
        np.save(ntf_path, test_x.astype(np.float32))
    if not nft_path.exists():
        np.save(nft_path, np.transpose(test_x, (0, 2, 1)).astype(np.float32))
    return ntf_path, nft_path


def evaluate_classical_cached(
    test_x: np.ndarray,
    mask_paths: dict[str, Path],
    cache_dir: Path,
    workers: int,
) -> dict[str, dict[str, float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}
    for key in tqdm(sorted(mask_paths), desc="classical", ascii=True, mininterval=2.0):
        result_path = cache_dir / f"{key}.json"
        cached = load_json(result_path)
        if cached is not None:
            print(f"[classical] cache hit {key}", flush=True)
            results[key] = cached
            continue
        print(f"[classical] evaluating {key}", flush=True)
        mask_ntf = np.load(mask_paths[key]).astype(np.float32)
        masked = test_x.copy()
        masked[mask_ntf == 0] = np.nan
        results[key] = {}
        rolling_ws = classical_window_size(key)
        knn_ks = (3, 5)
        methods: dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "locf_nocb": fill_locf_nocb,
            "linear_interpolation_pandas": fill_linear_pandas,
        }
        for name, fn in methods.items():
            t0 = time.time()
            pred = fn(masked.copy())
            missing = mask_ntf == 0
            sse = float(np.sum((pred[missing] - test_x[missing]) ** 2))
            count = int(np.sum(missing))
            results[key][name] = rmse_from_sum(sse, count)
            print(f"[classical] {key} {name} rmse={results[key][name]:.6f} elapsed={time.time()-t0:.1f}s", flush=True)
        rolling_candidates: dict[str, float] = {}
        for agg_name in ("mean",):
            t0 = time.time()
            pred = fill_rolling_mean(masked.copy(), window_size=rolling_ws)
            missing = mask_ntf == 0
            sse = float(np.sum((pred[missing] - test_x[missing]) ** 2))
            count = int(np.sum(missing))
            rolling_candidates[f"rolling_{agg_name}_{rolling_ws}"] = rmse_from_sum(sse, count)
            print(
                f"[classical] {key} rolling_{agg_name}_{rolling_ws} rmse={rolling_candidates[f'rolling_{agg_name}_{rolling_ws}']:.6f} elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
        results[key]["best_rolling_mean"] = min(rolling_candidates.values())
        knn_candidates: dict[str, float] = {}
        for knn_k in knn_ks:
            t0 = time.time()
            pred = fill_knn_time(masked.copy(), k=knn_k, window_size=rolling_ws)
            missing = mask_ntf == 0
            sse = float(np.sum((pred[missing] - test_x[missing]) ** 2))
            count = int(np.sum(missing))
            knn_candidates[f"knn_{knn_k}_{rolling_ws}"] = rmse_from_sum(sse, count)
            print(
                f"[classical] {key} knn_{knn_k}_{rolling_ws} rmse={knn_candidates[f'knn_{knn_k}_{rolling_ws}']:.6f} elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
        results[key]["best_knn"] = min(knn_candidates.values())
        save_json(result_path, results[key])
    return results


def method_order_for_plots(score_samplers: list[str]) -> list[str]:
    ordered = [
        "locf_nocb",
        "linear_interpolation_pandas",
        "best_rolling_mean",
        "best_knn",
        "standard_s4",
    ]
    ordered.extend([f"score_s4_{sampler}" for sampler in score_samplers])
    return ordered


def compute_classical_predictions_for_sample(truth_tf: np.ndarray, masked_tf: np.ndarray, mask_tf: np.ndarray, mask_key: str) -> dict[str, np.ndarray]:
    rolling_ws = classical_window_size(mask_key)
    rolling_pred = fill_rolling_mean(masked_tf[None, :, :].copy(), window_size=rolling_ws)[0]
    knn_candidates = [
        fill_knn_time(masked_tf[None, :, :].copy(), k=knn_k, window_size=rolling_ws)[0]
        for knn_k in (3, 5)
    ]
    missing = mask_tf == 0
    knn_errors = [float(np.mean((pred[missing] - truth_tf[missing]) ** 2)) for pred in knn_candidates]
    best_knn_pred = knn_candidates[int(np.argmin(knn_errors))]
    return {
        "locf_nocb": fill_locf_nocb(masked_tf[None, :, :].copy())[0],
        "linear_interpolation_pandas": fill_linear_pandas(masked_tf[None, :, :].copy())[0],
        "best_rolling_mean": rolling_pred,
        "best_knn": best_knn_pred,
    }


def sample_diffusion_prediction_single(
    model_spec: ModelSpec,
    ckpt_path: Path,
    sample_nft: np.ndarray,
    mask_nft: np.ndarray,
    sample_length: int,
    score_sampler: str,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = sample_nft.shape[0]
    sigma_val = 3.0
    diffusion_hyperparams = calc_diffusion_hyperparams(T=200, beta_0=1e-4, beta_T=0.02)
    for k, v in diffusion_hyperparams.items():
        if torch.is_tensor(v):
            diffusion_hyperparams[k] = v.to(device)
    model = load_model_checkpoint(model_spec, ckpt_path, features, sample_length, device)
    cond = torch.from_numpy(sample_nft[None, :, :]).float().to(device)
    mask = torch.from_numpy(mask_nft[None, :, :]).float().to(device)
    with torch.no_grad():
        if model_spec.objective == "standard":
            pred = sample_standard_model(model, cond, mask, diffusion_hyperparams)
        else:
            pred = sample_score_model(model, cond, mask, sigma_val, sampler_name=score_sampler)
    out = pred[0].detach().cpu().numpy()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def select_plot_keys(sample_length: int) -> list[str]:
    if sample_length == 500:
        return ["rm_20", "mnr_120", "bm_320"]
    return ["rm_10", "mnr_40", "bm_70"]


def select_plot_signal_indices(feature_count: int, signal_count: int) -> list[int]:
    if signal_count >= feature_count:
        return list(range(feature_count))
    return np.linspace(0, feature_count - 1, num=signal_count, dtype=int).tolist()


def create_eval_plots(
    eval_x: np.ndarray,
    feature_names: list[str],
    sample_length: int,
    mask_paths: dict[str, Path],
    model_specs: list[ModelSpec],
    model_ckpts: dict[str, Path],
    out_dir: Path,
    score_samplers: list[str],
    plot_samples: int,
    plot_signals: int,
) -> None:
    plots_dir = out_dir / "plots" / "eval"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_keys = [key for key in select_plot_keys(sample_length) if key in mask_paths]
    signal_indices = select_plot_signal_indices(eval_x.shape[2], plot_signals)
    summary: dict[str, dict] = {}
    for key in plot_keys:
        mask_ntf = np.load(mask_paths[key]).astype(np.float32)
        summary[key] = {"samples": []}
        for sample_idx in range(min(plot_samples, len(eval_x))):
            truth_tf = eval_x[sample_idx]
            mask_tf = mask_ntf[sample_idx]
            masked_tf = truth_tf.copy()
            masked_tf[mask_tf == 0] = np.nan
            predictions = compute_classical_predictions_for_sample(truth_tf, masked_tf, mask_tf, key)
            for model_spec in model_specs:
                if model_spec.name not in model_ckpts:
                    continue
                if model_spec.objective == "standard":
                    predictions[model_spec.name] = sample_diffusion_prediction_single(
                        model_spec, model_ckpts[model_spec.name], truth_tf.T, mask_tf.T, sample_length, score_sampler="em"
                    ).T
                else:
                    for sampler in score_samplers:
                        predictions[f"{model_spec.name}_{sampler}"] = sample_diffusion_prediction_single(
                            model_spec, model_ckpts[model_spec.name], truth_tf.T, mask_tf.T, sample_length, score_sampler=sampler
                        ).T
            fig, axes = plt.subplots(len(signal_indices), 1, figsize=(14, 3.6 * len(signal_indices)), sharex=True)
            if len(signal_indices) == 1:
                axes = [axes]
            time_axis = np.arange(sample_length)
            method_names = [name for name in method_order_for_plots(score_samplers) if name in predictions]
            for ax, feat_idx in zip(axes, signal_indices):
                ax.plot(time_axis, truth_tf[:, feat_idx], label="truth", color="black", linewidth=2.0)
                observed = np.where(mask_tf[:, feat_idx] > 0, truth_tf[:, feat_idx], np.nan)
                ax.plot(time_axis, observed, label="observed", color="gray", linewidth=1.2, alpha=0.9)
                for method_name in method_names:
                    ax.plot(time_axis, predictions[method_name][:, feat_idx], label=method_name, linewidth=1.0, alpha=0.9)
                ax.set_title(f"{key} | sample {sample_idx} | {feature_names[feat_idx]}")
                ax.grid(alpha=0.25)
            axes[0].legend(loc="upper right", ncol=3, fontsize=8)
            axes[-1].set_xlabel("time step")
            fig.tight_layout()
            plot_path = plots_dir / f"{key}_sample{sample_idx}.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            summary[key]["samples"].append(
                {
                    "sample_idx": sample_idx,
                    "signal_indices": signal_indices,
                    "signal_names": [feature_names[i] for i in signal_indices],
                    "plot_path": str(plot_path),
                    "methods": method_names,
                }
            )
    save_json(plots_dir / "plot_manifest.json", summary)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(checkpoint_dir.glob("iter_*.pt"))
    return checkpoints[-1] if checkpoints else None


def load_model_checkpoint(
    model_spec: ModelSpec,
    ckpt_path: Path,
    features: int,
    sample_length: int,
    device: torch.device,
) -> nn.Module:
    model = instantiate_model(model_spec, features, sample_length).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model_state = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in state["model_state_dict"].items()
    }
    model.load_state_dict(model_state, assign=True)
    model.eval()
    return model


def resolve_eval_batch_size(
    requested: int,
    model_spec: ModelSpec,
    model: nn.Module,
    cond_nft: torch.Tensor,
    mask_nft: torch.Tensor,
    diffusion_hyperparams: dict,
    sigma_value: float,
    batch_size_key: str,
    score_sampler: str,
) -> int:
    batch_size = max(1, requested)
    best = None
    while batch_size >= 1:
        try:
            cond = cond_nft[:batch_size]
            mask = mask_nft[:batch_size]
            with torch.no_grad():
                if model_spec.objective == "standard":
                    _ = sample_standard_model(model, cond, mask, diffusion_hyperparams)
                else:
                    _ = sample_score_model(model, cond, mask, sigma_value, sampler_name=score_sampler)
            best = batch_size
            if cond_nft.device.type != "cuda":
                return best
            next_batch = batch_size * 2
            if next_batch > len(cond_nft):
                return best
            batch_size = next_batch
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if best is not None:
                print(f"[eval] selected batch size for {batch_size_key}: {best}", flush=True)
                return best
            batch_size //= 2
            print(f"[eval] reduced batch size for {batch_size_key} to {batch_size}", flush=True)
    raise RuntimeError(f"Could not find a valid eval batch size for {batch_size_key}")


def evaluate_diffusion_cached(
    test_x: np.ndarray,
    sample_length: int,
    model_specs: list[ModelSpec],
    model_ckpts: dict[str, Path],
    mask_paths: dict[str, Path],
    cache_dir: Path,
    eval_batch_size: int,
    score_samplers: list[str],
) -> dict[str, dict[str, float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress_path = cache_dir.parent.parent / "progress.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = test_x.shape[2]
    sigma_val = 3.0
    diffusion_hyperparams = calc_diffusion_hyperparams(T=200, beta_0=1e-4, beta_T=0.02)
    for k, v in diffusion_hyperparams.items():
        if torch.is_tensor(v):
            diffusion_hyperparams[k] = v.to(device)

    test_nft = torch.from_numpy(np.transpose(test_x, (0, 2, 1))).float().to(device, non_blocking=True)
    results: dict[str, dict[str, float]] = {key: {} for key in sorted(mask_paths)}
    for model_spec in tqdm(model_specs, desc="models", ascii=True, mininterval=2.0):
        if model_spec.name not in model_ckpts:
            continue
        print(f"[diffusion] loading {model_spec.name}", flush=True)
        update_progress(progress_path, stage="eval_diffusion", current_model=model_spec.name)
        model = load_model_checkpoint(model_spec, model_ckpts[model_spec.name], features, sample_length, device)
        eval_names = [model_spec.name] if model_spec.objective == "standard" else [f"{model_spec.name}_{sampler}" for sampler in score_samplers]
        for eval_name in tqdm(eval_names, desc=f"{model_spec.name}:samplers", ascii=True, mininterval=2.0, leave=False):
            score_sampler = eval_name.split(f"{model_spec.name}_", 1)[1] if model_spec.objective == "score" else "em"
            for key in tqdm(sorted(mask_paths), desc=f"{eval_name}", ascii=True, mininterval=2.0):
                result_path = cache_dir / eval_name / f"{key}.json"
                cached = load_json(result_path)
                if cached is not None:
                    print(f"[diffusion] cache hit {eval_name} {key}", flush=True)
                    results[key][eval_name] = cached["rmse"]
                    update_progress(progress_path, stage="eval_diffusion", current_model=eval_name, current_key=key, cached=True)
                    continue
                mask_ntf = np.load(mask_paths[key]).astype(np.float32)
                mask_nft = torch.from_numpy(np.transpose(mask_ntf, (0, 2, 1))).float().to(device, non_blocking=True)
                local_batch_size = resolve_eval_batch_size(
                    eval_batch_size, model_spec, model, test_nft, mask_nft, diffusion_hyperparams, sigma_val, f"{eval_name}:{key}", score_sampler
                )
                print(f"[diffusion] evaluating {eval_name} {key} batch_size={local_batch_size}", flush=True)
                update_progress(progress_path, stage="eval_diffusion", current_model=eval_name, current_key=key, batch_size=local_batch_size)
                sse = 0.0
                count = 0
                t0 = time.time()
                total_batches = math.ceil(len(test_x) / local_batch_size)
                with torch.no_grad():
                    for start in tqdm(
                        range(0, len(test_x), local_batch_size),
                        total=total_batches,
                        desc=f"{eval_name}:{key}",
                        ascii=True,
                        mininterval=2.0,
                        leave=False,
                    ):
                        end = min(start + local_batch_size, len(test_x))
                        cond = test_nft[start:end]
                        mask = mask_nft[start:end]
                        if model_spec.objective == "standard":
                            pred = sample_standard_model(model, cond, mask, diffusion_hyperparams)
                        else:
                            pred = sample_score_model(model, cond, mask, sigma_val, sampler_name=score_sampler)
                        missing = mask == 0
                        diff = pred[missing] - cond[missing]
                        sse += float(torch.sum(diff * diff).item())
                        count += int(torch.sum(missing).item())
                        batch_idx = start // local_batch_size + 1
                        if batch_idx == 1 or batch_idx == total_batches or batch_idx % 20 == 0:
                            print(
                                f"[diffusion] {eval_name} {key} batch {batch_idx}/{total_batches}",
                                flush=True,
                            )
                            update_progress(progress_path, stage="eval_diffusion", current_model=eval_name, current_key=key, batch=batch_idx, total_batches=total_batches)
                rmse = rmse_from_sum(sse, count)
                results[key][eval_name] = rmse
                save_json(result_path, {"rmse": rmse, "elapsed_seconds": time.time() - t0, "batch_size": local_batch_size, "score_sampler": score_sampler})
                print(f"[diffusion] {eval_name} {key} rmse={rmse:.6f} elapsed={time.time()-t0:.1f}s", flush=True)
                update_progress(progress_path, stage="eval_diffusion", current_model=eval_name, current_key=key, rmse=rmse, completed_key=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def resolve_train_batch_size(
    requested: int,
    train_gpu: torch.Tensor,
    model_spec: ModelSpec,
    model: nn.Module,
    diffusion_hyperparams: dict,
    sigma_value: float,
    missing_k: int | Sequence[int],
    amp_mode: str,
) -> int:
    if train_gpu.device.type != "cuda":
        return requested
    batch_size = requested
    while batch_size >= 1:
        try:
            batch, mask = make_training_batch(train_gpu, batch_size, missing_k, mask_mode="rm")
            with autocast_context(train_gpu.device, amp_mode):
                if model_spec.objective == "standard":
                    loss = standard_diffusion_loss(model, batch, mask, diffusion_hyperparams)
                else:
                    loss = score_matching_loss(model, batch, mask, sigma_value)
            loss.backward()
            model.zero_grad(set_to_none=True)
            return batch_size
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)
            batch_size //= 2
            print(f"[train] reduced batch size for {model_spec.name} to {batch_size}", flush=True)
    raise RuntimeError(f"Could not find a valid training batch size for {model_spec.name}")


def train_model(
    train_x: np.ndarray,
    out_dir: Path,
    sample_length: int,
    model_spec: ModelSpec,
    seed: int,
    n_iters: int,
    batch_size: int,
    learning_rate: float,
    missing_k: int | Sequence[int],
    save_every: int,
    log_every: int,
    resume: bool,
    amp_mode: str,
) -> tuple[Path, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_cuda_runtime()
    set_seed(seed)
    model_dir = out_dir / "models" / model_spec.name
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = model_dir / "training_state.json"
    progress_path = out_dir / "progress.json"

    features = train_x.shape[2]
    model = instantiate_model(model_spec, features, sample_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    diffusion_hyperparams = calc_diffusion_hyperparams(T=200, beta_0=1e-4, beta_T=0.02)
    for k, v in diffusion_hyperparams.items():
        if torch.is_tensor(v):
            diffusion_hyperparams[k] = v.to(device)
    sigma_val = 3.0

    train_gpu = torch.from_numpy(train_x).to(device, non_blocking=True)
    effective_batch_size = resolve_train_batch_size(
        batch_size, train_gpu, model_spec, model, diffusion_hyperparams, sigma_val, missing_k, amp_mode
    )
    start_iter = 0
    loss_sum = 0.0
    loss_count = 0
    loss_last = None

    if resume:
        latest = find_latest_checkpoint(checkpoint_dir)
        if latest is not None:
            checkpoint = torch.load(latest, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_iter = int(checkpoint["iteration"])
            cached_state = load_json(state_path) or {}
            loss_sum = float(cached_state.get("loss_sum", 0.0))
            loss_count = int(cached_state.get("loss_count", 0))
            loss_last = cached_state.get("loss_last")
            print(f"[train] resumed {model_spec.name} from iter={start_iter}", flush=True)

    train_bar = tqdm(range(start_iter, n_iters), desc=f"train:{model_spec.name}", ascii=True, mininterval=2.0)
    for step in train_bar:
        batch, mask = make_training_batch(train_gpu, effective_batch_size, missing_k, mask_mode="rm")
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_mode):
            if model_spec.objective == "standard":
                loss = standard_diffusion_loss(model, batch, mask, diffusion_hyperparams)
            else:
                loss = score_matching_loss(model, batch, mask, sigma_val)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        loss_sum += loss_value
        loss_count += 1
        loss_last = loss_value

        if (step + 1) % log_every == 0 or step == 0 or step + 1 == n_iters:
            print(
                f"[train] {model_spec.name} iter={step+1}/{n_iters} loss={loss_value:.6f} batch_size={effective_batch_size}",
                flush=True,
            )
            train_bar.set_postfix(loss=f"{loss_value:.4f}", bs=effective_batch_size)
            update_progress(
                progress_path,
                stage="train",
                current_model=model_spec.name,
                iteration=step + 1,
                total_iterations=n_iters,
                loss=loss_value,
                batch_size=effective_batch_size,
            )
        if (step + 1) % save_every == 0 or step + 1 == n_iters:
            ckpt_path = checkpoint_dir / f"iter_{step+1:06d}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iteration": step + 1,
                },
                ckpt_path,
            )
            save_json(
                state_path,
                {
                    "iteration": step + 1,
                    "loss_sum": loss_sum,
                    "loss_count": loss_count,
                    "loss_last": loss_last,
                    "effective_batch_size": effective_batch_size,
                    "learning_rate": learning_rate,
                    "missing_k": missing_k,
                },
            )

    final_path = model_dir / "final.pt"
    torch.save({"model_state_dict": model.state_dict(), "iteration": n_iters}, final_path)
    return final_path, {
        "iterations": n_iters,
        "batch_size": effective_batch_size,
        "learning_rate": learning_rate,
        "missing_k": list(missing_k) if isinstance(missing_k, Sequence) and not isinstance(missing_k, (str, bytes)) else missing_k,
        "loss_last": loss_last,
        "loss_mean": (loss_sum / loss_count) if loss_count else None,
        "diffusion_steps": 200,
        "sigma_scorebased": sigma_val if model_spec.objective == "score" else None,
    }


def merge_rmse(
    classical_results: dict[str, dict[str, float]] | None,
    diffusion_results: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]] | None:
    if classical_results is None and diffusion_results is None:
        return None
    keys = set()
    if classical_results is not None:
        keys.update(classical_results.keys())
    if diffusion_results is not None:
        keys.update(diffusion_results.keys())
    merged: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        merged[key] = {}
        if classical_results is not None and key in classical_results:
            merged[key].update(classical_results[key])
        if diffusion_results is not None and key in diffusion_results:
            merged[key].update(diffusion_results[key])
    return merged


def run_pipeline(
    spec: DatasetSpec,
    output_root: Path,
    model_names: list[str],
    mode: str,
    seed: int,
    n_iters: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    missing_k: int | Sequence[int],
    save_every: int,
    log_every: int,
    resume: bool,
    classical_workers: int,
    amp_mode: str,
    score_samplers: list[str],
    scaling_mode: str,
    plot_eval_samples: int,
    plot_eval_signals: int,
) -> None:
    out_dir = output_root / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    caches = cache_paths(out_dir)
    for path in caches.values():
        path.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw = load_dataset(spec)
    eval_raw = test_raw if spec.max_eval_samples is None else test_raw[: spec.max_eval_samples]
    sample_length = train_raw.shape[1]
    feature_names = load_feature_names(spec, train_raw.shape[2])
    print(f"[{spec.name}] loaded train={train_raw.shape} test={test_raw.shape} eval={eval_raw.shape}", flush=True)
    update_progress(progress_path, stage="setup", dataset=spec.name, train_shape=list(train_raw.shape), eval_shape=list(eval_raw.shape))

    train_x, test_x, mean, std = prepare_data_by_scaling_mode(train_raw, eval_raw, scaling_mode)
    np.save(out_dir / "feature_mean.npy", mean)
    np.save(out_dir / "feature_std.npy", std)
    np.save(out_dir / "train_subset.npy", train_raw)
    np.save(out_dir / "test_subset.npy", test_raw)

    summary = load_json(out_dir / "results.json") or {
        "dataset": {
            "name": spec.name,
            "train_path": str(spec.train_path),
            "test_path": str(spec.test_path),
            "feature_columns_path": str(spec.feature_columns_path) if spec.feature_columns_path is not None else None,
            "subset_fraction": spec.subset_fraction,
            "max_eval_samples": spec.max_eval_samples,
            "scaling_mode": scaling_mode,
        },
        "train_shape": list(train_raw.shape),
        "test_shape": list(test_raw.shape),
        "eval_shape": list(eval_raw.shape),
        "standardization": {
            "mean_path": str(out_dir / "feature_mean.npy"),
            "std_path": str(out_dir / "feature_std.npy"),
        },
        "training": {},
        "rmse": None,
    }

    model_specs = [MODEL_SPECS[name] for name in model_names]
    model_ckpts: dict[str, Path] = {}
    if mode in {"train", "full"}:
        for idx, model_spec in enumerate(model_specs):
            ckpt, meta = train_model(
                train_x=train_x,
                out_dir=out_dir,
                sample_length=sample_length,
                model_spec=model_spec,
                seed=seed + idx,
                n_iters=n_iters,
                batch_size=batch_size,
                learning_rate=learning_rate,
                missing_k=missing_k,
                save_every=save_every,
                log_every=log_every,
                resume=resume,
                amp_mode=amp_mode,
            )
            model_ckpts[model_spec.name] = ckpt
            summary["training"][model_spec.name] = meta
            save_json(out_dir / "results.partial.json", summary)
        update_progress(progress_path, stage="train_complete", training_models=model_names)
    else:
        for model_spec in model_specs:
            final_path = out_dir / "models" / model_spec.name / "final.pt"
            if not final_path.exists():
                raise FileNotFoundError(f"Missing model checkpoint for {model_spec.name}: {final_path}")
            model_ckpts[model_spec.name] = final_path

    if mode in {"eval", "full"}:
        prepare_standardized_tensors(test_x, caches["tensors"])
        mask_paths = ensure_mask_cache(test_x, sample_length, seed, caches["masks"])
        classical_results = evaluate_classical_cached(test_x, mask_paths, caches["classical"], workers=classical_workers)
        print(f"[{spec.name}] classical baselines evaluated", flush=True)
        update_progress(progress_path, stage="eval_classical_complete")
        diffusion_results = evaluate_diffusion_cached(
            test_x=test_x,
            sample_length=sample_length,
            model_specs=model_specs,
            model_ckpts=model_ckpts,
            mask_paths=mask_paths,
            cache_dir=caches["diffusion"],
            eval_batch_size=eval_batch_size,
            score_samplers=score_samplers,
        )
        print(f"[{spec.name}] diffusion models evaluated", flush=True)
        update_progress(progress_path, stage="eval_diffusion_complete")
        summary["rmse"] = merge_rmse(classical_results, diffusion_results)
        if plot_eval_samples > 0 and plot_eval_signals > 0:
            print(f"[{spec.name}] generating evaluation plots", flush=True)
            create_eval_plots(
                eval_x=test_x,
                feature_names=feature_names,
                sample_length=sample_length,
                mask_paths=mask_paths,
                model_specs=model_specs,
                model_ckpts=model_ckpts,
                out_dir=out_dir,
                score_samplers=score_samplers,
                plot_samples=plot_eval_samples,
                plot_signals=plot_eval_signals,
            )
            update_progress(progress_path, stage="eval_plots_complete")
    save_json(out_dir / "results.json", summary)
    update_progress(progress_path, stage="done", results_path=str(out_dir / "results.json"))
