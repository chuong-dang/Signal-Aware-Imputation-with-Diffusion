from __future__ import annotations

import numpy as np
import pandas as pd


def fill_forward_backward(masked_ntf: np.ndarray) -> np.ndarray:
    n, t, f = masked_ntf.shape
    series = masked_ntf.transpose(0, 2, 1).reshape(-1, t)
    time_idx = np.arange(t)

    forward_idx = np.where(~np.isnan(series), time_idx, 0)
    np.maximum.accumulate(forward_idx, axis=1, out=forward_idx)
    ffilled = np.take_along_axis(series, forward_idx, axis=1)

    rev = series[:, ::-1]
    reverse_idx = np.where(~np.isnan(rev), time_idx, 0)
    np.maximum.accumulate(reverse_idx, axis=1, out=reverse_idx)
    bfilled = np.take_along_axis(rev, reverse_idx, axis=1)[:, ::-1]

    filled = np.where(np.isnan(ffilled), bfilled, ffilled)
    filled = np.nan_to_num(filled, nan=0.0)
    return filled.reshape(n, f, t).transpose(0, 2, 1)


def fill_locf_nocb(masked_ntf: np.ndarray) -> np.ndarray:
    return fill_forward_backward(masked_ntf)


def fill_linear_pandas(masked_ntf: np.ndarray) -> np.ndarray:
    n, t, _ = masked_ntf.shape
    out = masked_ntf.copy()
    time_idx = np.arange(t)
    for sample_idx in range(n):
        frame = pd.DataFrame(out[sample_idx], index=time_idx)
        frame = frame.interpolate(method="linear", axis=0).ffill().bfill()
        out[sample_idx] = frame.to_numpy(dtype=np.float32, copy=False)
    return out


def _rolling_mean_single(masked_tf: np.ndarray, window_size: int) -> np.ndarray:
    t, _ = masked_tf.shape
    locf_fallback = fill_locf_nocb(masked_tf[None, :, :])[0]
    out = masked_tf.copy()
    for ti in range(t):
        left = max(0, ti - window_size)
        right = min(t, ti + window_size + 1)
        window = masked_tf[left:right]
        means = np.nanmean(window, axis=0)
        means = np.where(np.isnan(means), locf_fallback[ti], means)
        missing = np.isnan(out[ti])
        if np.any(missing):
            out[ti, missing] = means[missing]
    return out


def fill_rolling_mean(masked_ntf: np.ndarray, window_size: int) -> np.ndarray:
    out = masked_ntf.copy()
    for sample_idx in range(out.shape[0]):
        out[sample_idx] = _rolling_mean_single(out[sample_idx], window_size).astype(np.float32, copy=False)
    return out


def _knn_time_single(masked_tf: np.ndarray, k: int, window_size: int) -> np.ndarray:
    t, f = masked_tf.shape
    locf_fallback = fill_locf_nocb(masked_tf[None, :, :])[0]
    out = masked_tf.copy()
    for ti in range(t):
        left = max(0, ti - window_size)
        right = min(t, ti + window_size + 1)
        window = masked_tf[left:right]
        time_candidates = np.arange(left, right)
        distances = np.abs(time_candidates - ti)
        for fi in range(f):
            if not np.isnan(out[ti, fi]):
                continue
            valid = ~np.isnan(window[:, fi])
            if np.any(valid):
                valid_values = window[valid, fi]
                valid_distances = distances[valid]
                order = np.argsort(valid_distances, kind="stable")[:k]
                out[ti, fi] = float(np.mean(valid_values[order]))
            else:
                out[ti, fi] = locf_fallback[ti, fi]
    return out


def fill_knn_time(masked_ntf: np.ndarray, k: int, window_size: int) -> np.ndarray:
    out = masked_ntf.copy()
    for sample_idx in range(out.shape[0]):
        out[sample_idx] = _knn_time_single(out[sample_idx], k=k, window_size=window_size).astype(np.float32, copy=False)
    return out


def classical_window_size(mask_key: str) -> int:
    mask_mode, missing_k = mask_key.split("_")
    missing_k_int = int(missing_k)
    if mask_mode in {"bm", "mnr"}:
        return int(missing_k_int * 2 / 3) + 1
    return 15

