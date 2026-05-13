from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


MASK_SPECS_BY_SAMPLE_LENGTH = {
    100: {
        "rm": (10, 40, 70),
        "mnr": (10, 40, 70),
        "bm": (10, 40, 70),
    },
    500: {
        "rm": (20, 120, 220, 320),
        "mnr": (20, 120, 220, 320),
        "bm": (20, 120, 220, 320),
    },
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    train_path: Path
    test_path: Path
    feature_columns_path: Path | None = None
    subset_fraction: float = 1.0
    max_eval_samples: int | None = None


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_cuda_runtime() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def get_mask_specs(sample_length: int) -> dict[str, tuple[int, ...]]:
    try:
        return MASK_SPECS_BY_SAMPLE_LENGTH[sample_length]
    except KeyError as exc:
        raise ValueError(f"Unsupported sample_length for mask specs: {sample_length}") from exc


def load_dataset(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    train = np.load(spec.train_path)
    test = np.load(spec.test_path)
    if spec.subset_fraction < 1.0:
        train_n = max(1, int(len(train) * spec.subset_fraction))
        test_n = max(1, int(len(test) * spec.subset_fraction))
        train = train[:train_n]
        test = test[:test_n]
    return train.astype(np.float32), test.astype(np.float32)


def load_feature_names(spec: DatasetSpec, feature_count: int) -> list[str]:
    if spec.feature_columns_path is not None and spec.feature_columns_path.exists():
        names = spec.feature_columns_path.read_text(encoding="utf-8").splitlines()
        if len(names) == feature_count:
            return names
    return [f"f{i}" for i in range(feature_count)]


def fit_standardizer(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = train.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean[None, None, :]) / std[None, None, :]


def prepare_data_by_scaling_mode(
    train_raw: np.ndarray,
    eval_raw: np.ndarray,
    scaling_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if scaling_mode == "train_standardize":
        mean, std = fit_standardizer(train_raw)
        return standardize(train_raw, mean, std), standardize(eval_raw, mean, std), mean, std
    if scaling_mode == "prestandardized":
        features = train_raw.shape[2]
        mean = np.zeros(features, dtype=np.float32)
        std = np.ones(features, dtype=np.float32)
        return train_raw.astype(np.float32), eval_raw.astype(np.float32), mean, std
    if scaling_mode == "none":
        features = train_raw.shape[2]
        mean = np.zeros(features, dtype=np.float32)
        std = np.ones(features, dtype=np.float32)
        return train_raw.astype(np.float32), eval_raw.astype(np.float32), mean, std
    raise ValueError(f"Unknown scaling_mode: {scaling_mode}")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_progress(progress_path: Path, **fields) -> None:
    payload = load_json(progress_path) or {}
    payload.update(fields)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_json(progress_path, payload)


def stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def rmse_from_sum(sse: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return float(math.sqrt(sse / count))


def cache_paths(out_dir: Path) -> dict[str, Path]:
    base = out_dir / "cache"
    return {
        "base": base,
        "masks": base / "masks",
        "classical": base / "classical",
        "diffusion": base / "diffusion",
        "tensors": base / "tensors",
        "training": base / "training",
    }
