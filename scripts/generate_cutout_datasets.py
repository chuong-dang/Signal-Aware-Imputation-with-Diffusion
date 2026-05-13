from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


VEHICLE_ROOT = Path("/mnt/nvme/full/trips")
PUBLIC_ROOT = Path("/home/stdUser001/datasets/time_series_datasets")
DEFAULT_OUTPUT_ROOT = Path("/mnt/nvme/missingvalueimputation/datasets")
SAMPLE_LENGTHS = (100, 500)
RNG_SEED = 42
VEHICLE_DROP_COLUMNS = {
    "epto_sw_stat",
    "hv_batisores_cval",
    "hvdc_maxdischrgcurr",
    "hvdc_maxdischrgpwr",
}


@dataclass(frozen=True)
class PublicDatasetSpec:
    name: str
    path: Path
    kind: str


def chunk_valid_windows(rows: np.ndarray, sample_length: int) -> Iterator[np.ndarray]:
    buffer: list[np.ndarray] = []
    for row in rows:
        if np.isfinite(row).all():
            buffer.append(row.astype(np.float32, copy=False))
            if len(buffer) == sample_length:
                yield np.stack(buffer, axis=0)
                buffer = []
        else:
            buffer = []


def load_public_matrix(spec: PublicDatasetSpec) -> tuple[np.ndarray, list[str]]:
    if spec.kind == "csv":
        df = pd.read_csv(spec.path)
        if "date" in df.columns:
            df = df.drop(columns=["date"])
        df = df.apply(pd.to_numeric, errors="coerce")
        return df.to_numpy(dtype=np.float32, copy=False), [str(c) for c in df.columns]

    if spec.kind == "txt":
        df = pd.read_csv(spec.path, header=None)
        df = df.apply(pd.to_numeric, errors="coerce")
        return df.to_numpy(dtype=np.float32, copy=False), [f"f{i}" for i in range(df.shape[1])]

    if spec.kind == "npz":
        with np.load(spec.path) as data:
            arr = data[data.files[0]]
        if arr.ndim == 2:
            matrix = arr
            feature_names = [f"f{i}" for i in range(matrix.shape[1])]
        elif arr.ndim == 3:
            t, d1, d2 = arr.shape
            matrix = arr.reshape(t, d1 * d2)
            feature_names = [f"n{i}_c{j}" for i in range(d1) for j in range(d2)]
        else:
            raise ValueError(f"Unsupported array shape for {spec.path}: {arr.shape}")
        return matrix.astype(np.float32, copy=False), feature_names

    raise ValueError(f"Unsupported dataset kind: {spec.kind}")


def vehicle_prefixes(vehicle_root: Path) -> list[str]:
    return sorted({p.name.split("_")[0] for p in vehicle_root.glob("*.parquet")})


def vehicle_common_numeric_columns(vehicle_root: Path) -> list[str]:
    schemas = [set(pq.ParquetFile(p).schema.names) for p in vehicle_root.glob("*.parquet")]
    common_cols = set.intersection(*schemas)
    metadata_cols = {"AbsTime_DIADEM", "signal_ts", "vehicle_id"}
    numeric_cols = sorted(common_cols - metadata_cols - VEHICLE_DROP_COLUMNS)
    return numeric_cols


def vehicle_files_for_prefix(vehicle_root: Path, prefix: str) -> list[Path]:
    return sorted(vehicle_root.glob(f"{prefix}_*.parquet"))


def resample_vehicle_frame(df: pd.DataFrame, numeric_cols: list[str], resample_seconds: int | None) -> pd.DataFrame:
    if resample_seconds is None:
        return df[numeric_cols]

    if "signal_ts" not in df.columns:
        raise ValueError("signal_ts column is required for vehicle resampling")

    ts = pd.to_datetime(df["signal_ts"], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("Failed to parse signal_ts values during resampling")

    freq = f"{resample_seconds}s"
    rounded = ts.dt.round(freq)
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    grouped = numeric.groupby(rounded, sort=True).mean()
    return grouped.reset_index(drop=True)


def iter_vehicle_windows(
    files: Iterable[Path],
    numeric_cols: list[str],
    sample_length: int,
    resample_seconds: int | None = None,
) -> Iterator[np.ndarray]:
    for file_path in files:
        columns = list(numeric_cols)
        if resample_seconds is not None:
            columns = ["signal_ts", *columns]
        df = pd.read_parquet(file_path, columns=columns)
        if resample_seconds is not None:
            df = resample_vehicle_frame(df, numeric_cols, resample_seconds)
        matrix = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=False)
        yield from chunk_valid_windows(matrix, sample_length)


def iter_public_windows(matrix: np.ndarray, sample_length: int) -> Iterator[np.ndarray]:
    yield from chunk_valid_windows(matrix, sample_length)


def build_dataset(
    output_root: Path,
    dataset_name: str,
    feature_names: list[str],
    sample_length: int,
    windows_factory: Callable[[], Iterator[np.ndarray]],
    extra_metadata: dict | None = None,
) -> dict[str, int]:
    dataset_dir = output_root / dataset_name / f"sample_length_{sample_length}"
    training_dir = dataset_dir / "training_sets"
    test_dir = dataset_dir / "test_sets"
    training_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    total = sum(1 for _ in windows_factory())
    n_features = len(feature_names)
    train_count = int(total * 0.8)
    test_count = total - train_count

    train_path = training_dir / f"{dataset_name}_train_data.npy"
    test_path = test_dir / f"{dataset_name}_test_data.npy"
    feature_path = dataset_dir / "feature_columns.txt"
    metadata_path = dataset_dir / "metadata.json"

    with feature_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(feature_names) + "\n")

    if total == 0:
        np.save(train_path, np.empty((0, sample_length, n_features), dtype=np.float32))
        np.save(test_path, np.empty((0, sample_length, n_features), dtype=np.float32))
    else:
        rng = np.random.default_rng(RNG_SEED)
        assignments = np.zeros(total, dtype=np.bool_)
        assignments[:train_count] = True
        rng.shuffle(assignments)

        train_arr = np.lib.format.open_memmap(
            train_path, mode="w+", dtype=np.float32, shape=(train_count, sample_length, n_features)
        )
        test_arr = np.lib.format.open_memmap(
            test_path, mode="w+", dtype=np.float32, shape=(test_count, sample_length, n_features)
        )
        train_i = 0
        test_i = 0
        for idx, window in enumerate(windows_factory()):
            if assignments[idx]:
                train_arr[train_i] = window
                train_i += 1
            else:
                test_arr[test_i] = window
                test_i += 1
        del train_arr
        del test_arr

    metadata = {
        "dataset_name": dataset_name,
        "sample_length": sample_length,
        "feature_count": n_features,
        "total_windows": total,
        "train_windows": train_count,
        "test_windows": test_count,
        "layout": "N,T,F",
        "dtype": "float32",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    return metadata


def generate_vehicle_datasets(
    output_root: Path,
    vehicle_root: Path,
    include_prefixes: list[str] | None = None,
    resample_seconds: int | None = None,
) -> list[dict[str, int]]:
    prefixes = vehicle_prefixes(vehicle_root)
    if include_prefixes is not None:
        wanted = set(include_prefixes)
        prefixes = [prefix for prefix in prefixes if prefix in wanted]
    numeric_cols = vehicle_common_numeric_columns(vehicle_root)
    results: list[dict[str, int]] = []
    for prefix in prefixes:
        dataset_name = f"vehicle_{prefix}"
        files = vehicle_files_for_prefix(vehicle_root, prefix)
        print(
            f"[vehicle] {dataset_name}: {len(files)} parquet files, {len(numeric_cols)} numeric columns"
            + (f", resample={resample_seconds}s" if resample_seconds is not None else ""),
            flush=True,
        )
        for sample_length in SAMPLE_LENGTHS:
            metadata = build_dataset(
                output_root=output_root,
                dataset_name=dataset_name,
                feature_names=numeric_cols,
                sample_length=sample_length,
                windows_factory=lambda files=files, numeric_cols=numeric_cols, sample_length=sample_length, resample_seconds=resample_seconds: iter_vehicle_windows(
                    files, numeric_cols, sample_length, resample_seconds=resample_seconds
                ),
                extra_metadata={"resample_seconds": resample_seconds},
            )
            print(
                f"  sample_length_{sample_length}: total={metadata['total_windows']} "
                f"train={metadata['train_windows']} test={metadata['test_windows']}",
                flush=True,
            )
            results.append(metadata)
    return results


def public_dataset_specs(public_root: Path) -> tuple[PublicDatasetSpec, ...]:
    return (
        PublicDatasetSpec("electricity", public_root / "electricity/electricity.csv", "csv"),
        PublicDatasetSpec("traffic", public_root / "traffic/traffic.csv", "csv"),
        PublicDatasetSpec("ETTh1", public_root / "ETT-small/ETTh1.csv", "csv"),
        PublicDatasetSpec("ETTh2", public_root / "ETT-small/ETTh2.csv", "csv"),
        PublicDatasetSpec("ETTm1", public_root / "ETT-small/ETTm1.csv", "csv"),
        PublicDatasetSpec("ETTm2", public_root / "ETT-small/ETTm2.csv", "csv"),
        PublicDatasetSpec("exchange_rate", public_root / "exchange_rate/exchange_rate.csv", "csv"),
        PublicDatasetSpec("weather", public_root / "weather/weather.csv", "csv"),
        PublicDatasetSpec("solar_AL", public_root / "Solar/solar_AL.txt", "txt"),
        PublicDatasetSpec("PEMS03", public_root / "PEMS/PEMS03.npz", "npz"),
        PublicDatasetSpec("PEMS04", public_root / "PEMS/PEMS04.npz", "npz"),
        PublicDatasetSpec("PEMS07", public_root / "PEMS/PEMS07.npz", "npz"),
        PublicDatasetSpec("PEMS08", public_root / "PEMS/PEMS08.npz", "npz"),
    )


def generate_public_datasets(output_root: Path, public_root: Path) -> list[dict[str, int]]:
    results: list[dict[str, int]] = []
    for spec in public_dataset_specs(public_root):
        matrix, feature_names = load_public_matrix(spec)
        print(f"[public] {spec.name}: rows={matrix.shape[0]} features={matrix.shape[1]}", flush=True)
        for sample_length in SAMPLE_LENGTHS:
            metadata = build_dataset(
                output_root=output_root,
                dataset_name=spec.name,
                feature_names=feature_names,
                sample_length=sample_length,
                windows_factory=lambda matrix=matrix, sample_length=sample_length: iter_public_windows(
                    matrix, sample_length
                ),
            )
            print(
                f"  sample_length_{sample_length}: total={metadata['total_windows']} "
                f"train={metadata['train_windows']} test={metadata['test_windows']}",
                flush=True,
            )
            results.append(metadata)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vehicle-root", type=Path, default=VEHICLE_ROOT)
    parser.add_argument("--public-root", type=Path, default=PUBLIC_ROOT)
    parser.add_argument(
        "--source",
        choices=("all", "vehicle", "public"),
        default="all",
        help="Which source datasets to generate.",
    )
    parser.add_argument("--include-vehicle-prefixes", nargs="+", default=None)
    parser.add_argument("--vehicle-resample-seconds", type=int, default=None)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, int]] = []

    if args.source in {"all", "vehicle"}:
        all_results.extend(
            generate_vehicle_datasets(
                args.output_root,
                vehicle_root=args.vehicle_root,
                include_prefixes=args.include_vehicle_prefixes,
                resample_seconds=args.vehicle_resample_seconds,
            )
        )
    if args.source in {"all", "public"}:
        all_results.extend(generate_public_datasets(args.output_root, args.public_root))

    summary_path = args.output_root / "generation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        f.write("\n")
    print(f"Wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
