from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-length", type=int, choices=(100, 500), required=True)
    parser.add_argument("--subset-fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-datasets", nargs="+", default=None)
    return parser.parse_args()


def subset_indices(length: int, fraction: float, rng: np.random.Generator) -> np.ndarray:
    count = max(1, int(length * fraction))
    if count >= length:
        return np.arange(length)
    return np.sort(rng.choice(length, size=count, replace=False))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    vehicles = sorted(
        path for path in args.datasets_root.iterdir() if path.is_dir() and path.name.startswith("vehicle_")
    )
    if args.include_datasets is not None:
        include = set(args.include_datasets)
        vehicles = [path for path in vehicles if path.name in include]
    if not vehicles:
        raise FileNotFoundError(f"No vehicle_* datasets found under {args.datasets_root}")

    pooled_train = []
    pooled_test = []
    sources = []
    feature_columns = None

    for vehicle_dir in vehicles:
        base = vehicle_dir / f"sample_length_{args.sample_length}"
        train_path = base / "training_sets" / f"{vehicle_dir.name}_train_data.npy"
        test_path = base / "test_sets" / f"{vehicle_dir.name}_test_data.npy"
        feature_path = base / "feature_columns.txt"
        if not train_path.exists() or not test_path.exists():
            continue

        train = np.load(train_path)
        test = np.load(test_path)
        train_idx = subset_indices(len(train), args.subset_fraction, rng)
        test_idx = subset_indices(len(test), args.subset_fraction, rng)
        pooled_train.append(train[train_idx].astype(np.float32))
        pooled_test.append(test[test_idx].astype(np.float32))

        current_columns = feature_path.read_text(encoding="utf-8").splitlines() if feature_path.exists() else None
        if feature_columns is None:
            feature_columns = current_columns
        elif current_columns is not None and feature_columns != current_columns:
            raise ValueError(f"Feature column mismatch for {vehicle_dir.name}")

        sources.append(
            {
                "dataset": vehicle_dir.name,
                "train_total": int(train.shape[0]),
                "test_total": int(test.shape[0]),
                "train_selected": int(len(train_idx)),
                "test_selected": int(len(test_idx)),
            }
        )

    if not pooled_train or not pooled_test:
        raise RuntimeError("No train/test arrays were pooled")

    train_combined = np.concatenate(pooled_train, axis=0)
    test_combined = np.concatenate(pooled_test, axis=0)

    train_order = rng.permutation(len(train_combined))
    test_order = rng.permutation(len(test_combined))
    train_combined = train_combined[train_order]
    test_combined = test_combined[test_order]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "train.npy", train_combined)
    np.save(args.output_dir / "test.npy", test_combined)

    if feature_columns is not None:
        (args.output_dir / "feature_columns.txt").write_text("\n".join(feature_columns) + "\n", encoding="utf-8")

    metadata = {
        "datasets_root": str(args.datasets_root),
        "sample_length": args.sample_length,
        "subset_fraction": args.subset_fraction,
        "seed": args.seed,
        "include_datasets": args.include_datasets,
        "train_shape": list(train_combined.shape),
        "test_shape": list(test_combined.shape),
        "sources": sources,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
