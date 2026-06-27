from __future__ import annotations

import csv
from pathlib import Path
from statistics import median

IGNORED_PREVIEW_COLUMNS = {"tick", "t_ns", "quality_flags"}


def read_numeric_preview_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, str]] = []
    numeric_columns: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            for column, value in row.items():
                if column is None or column in IGNORED_PREVIEW_COLUMNS:
                    continue
                if _parse_float(value) is not None:
                    numeric_columns.add(column)
    ordered_columns = sorted(numeric_columns)
    parsed_rows: list[dict[str, float]] = []
    for row in rows:
        parsed = {}
        for column in ordered_columns:
            parsed[column] = _parse_float(row.get(column)) or 0.0
        parsed_rows.append(parsed)
    return parsed_rows


def discover_numeric_preview_columns(path: Path) -> list[str]:
    rows = read_numeric_preview_rows(path)
    return sorted({column for row in rows for column in row})


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    features: dict[str, float] = {}
    channels = sorted({channel for row in rows for channel in row})
    for channel in channels:
        values = [abs(float(row.get(channel, 0.0))) for row in rows]
        profile = robust_profile(values)
        features[f"{channel}_median"] = profile["center"]
        features[f"{channel}_mad"] = profile["mad"]
        features[f"{channel}_p95"] = percentile(values, 0.95)
    return features


def summarize_preview(path: Path) -> dict[str, float]:
    rows = read_numeric_preview_rows(path)
    if not rows:
        return {}
    return summarize_rows(rows)


def robust_profile(values: list[float]) -> dict[str, float]:
    if not values:
        return {"center": 0.0, "mad": 1.0}
    center = median(values)
    deviations = [abs(value - center) for value in values]
    mad = median(deviations) or 1.0
    return {"center": float(center), "mad": float(mad)}


def full_profile(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    center = median(ordered) if ordered else 0.0
    deviations = [abs(value - center) for value in ordered]
    mad = median(deviations) if deviations else 1.0
    mad = mad or 1.0
    return {
        "center": float(center),
        "mad": float(mad),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "min": float(ordered[0]) if ordered else 0.0,
        "max": float(ordered[-1]) if ordered else 0.0,
    }


def mean_profile(features: list[dict[str, float]]) -> dict[str, float]:
    if not features:
        return {}
    keys = sorted({key for feature in features for key in feature})
    return {
        key: sum(feature.get(key, 0.0) for feature in features) / len(features)
        for key in keys
    }


def feature_distance(left: dict[str, float], right: dict[str, float]) -> tuple[float, dict[str, float]]:
    shared = sorted(set(left) & set(right))
    if not shared:
        return 0.0, {}
    contributions = {
        key: abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) / max(abs(float(right.get(key, 0.0))), 1.0)
        for key in shared
    }
    return sum(contributions.values()) / len(contributions), contributions


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return float(ordered[idx])


def _parse_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

