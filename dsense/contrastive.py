from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from .manifest import project_path
from .models.features import feature_distance, feature_manifest, mean_profile, read_numeric_preview_rows, slope, variance
from .models.scene_store import SceneFeatureStore, build_or_load_feature_store, feature_manifest_from_store
from .utils.files import ensure_dir, read_json, write_json
from .utils.timebase import utc_now_iso

CONTRASTIVE_FORMAT = "dsense-contrastive-model-v1"
CONTRASTIVE_MODEL_VERSION = "contrastive-profile-v2"
TCN_WEIGHTS_NAME = "contrastive_tcn.pt"
TCN_BACKENDS = {"torch_tcn", "tcn"}


@dataclass(frozen=True)
class ContrastiveTemporalModel:
    project_name: str
    trained_utc: str
    scene_count: int
    backend: str
    label_counts: dict[str, int]
    family_counts: dict[str, int]
    label_profiles: dict[str, dict[str, float]]
    family_profiles: dict[str, dict[str, float]]
    sequence_channels: list[str]
    feature_manifest: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": CONTRASTIVE_FORMAT,
            "project_name": self.project_name,
            "trained_utc": self.trained_utc,
            "scene_count": self.scene_count,
            "backend": self.backend,
            "label_counts": self.label_counts,
            "family_counts": self.family_counts,
            "label_profiles": self.label_profiles,
            "family_profiles": self.family_profiles,
            "sequence_channels": self.sequence_channels,
            "feature_manifest": self.feature_manifest,
        }


@dataclass(frozen=True)
class _SceneExample:
    scene_id: str
    label: str
    family: str
    rows: list[dict[str, float]]
    features: dict[str, float]


def contrastive_path(project_name: str) -> Path:
    return project_path(project_name) / "exports" / "contrastive_model.json"


def train_project_contrastive(project_name: str, backend: str = "profile") -> ContrastiveTemporalModel:
    normalized_backend = _normalize_backend(backend)
    if normalized_backend in TCN_BACKENDS:
        return _train_torch_tcn_or_fallback(project_name, normalized_backend)
    if normalized_backend != "profile":
        raise ValueError(f"Unknown contrastive backend: {backend}")
    store = build_or_load_feature_store(project_name, workers=1, model_options={"contrastive_backend": "profile"})
    return train_project_contrastive_from_store(store, backend="profile")


def train_project_contrastive_from_store(store: SceneFeatureStore, backend: str = "profile") -> ContrastiveTemporalModel:
    normalized_backend = _normalize_backend(backend)
    if normalized_backend != "profile":
        return train_project_contrastive(store.project_name, backend=backend)
    examples = [
        _SceneExample(scene.scene_id, scene.label, scene_family(scene.label), scene.preview_rows, scene.contrastive_features or extract_contrastive_features(scene.preview_rows))
        for scene in store.accepted_scenes
        if scene.preview_rows
    ]
    return _train_profile_from_examples(store.project_name, examples, backend="profile", store=store)


def train_and_save_project_contrastive(project_name: str, backend: str = "profile") -> ContrastiveTemporalModel:
    model = train_project_contrastive(project_name, backend=backend)
    out = contrastive_path(project_name)
    ensure_dir(out.parent)
    write_json(out, model.to_dict())
    return model


def load_project_contrastive(project_name: str) -> ContrastiveTemporalModel | None:
    path = contrastive_path(project_name)
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return None
    return ContrastiveTemporalModel(
        project_name=str(data.get("project_name", project_name)),
        trained_utc=str(data.get("trained_utc", "")),
        scene_count=int(data.get("scene_count", 0)),
        backend=str(data.get("backend", "profile")),
        label_counts={str(k): int(v) for k, v in dict(data.get("label_counts", {})).items()},
        family_counts={str(k): int(v) for k, v in dict(data.get("family_counts", {})).items()},
        label_profiles=_float_profile_map(data.get("label_profiles", {})),
        family_profiles=_float_profile_map(data.get("family_profiles", {})),
        sequence_channels=[str(channel) for channel in list(data.get("sequence_channels", []))],
        feature_manifest=dict(data.get("feature_manifest", {})),
    )


def predict_scene_contrastive(model: ContrastiveTemporalModel | None, preview_path: Path) -> dict[str, object]:
    if model is None:
        return _unknown_prediction("missing contrastive model")
    if model.backend in TCN_BACKENDS:
        return _predict_scene_tcn(model, preview_path)
    rows = read_numeric_preview_rows(preview_path) if preview_path.exists() else []
    if not rows or not model.family_profiles:
        return _unknown_prediction("missing preview rows or profiles", backend=model.backend, channels=model.sequence_channels)
    features = extract_contrastive_features(rows)
    family_prediction = _predict_profile(model.family_profiles, features, model.family_counts)
    label_prediction = _predict_profile(model.label_profiles, features, model.label_counts)
    confidence = _combined_confidence(family_prediction, label_prediction, model.scene_count)
    return {
        "family": family_prediction["name"],
        "label": label_prediction["name"],
        "confidence": confidence,
        "family_distance": family_prediction["distance"],
        "label_distance": label_prediction["distance"],
        "nearest_family_distances": family_prediction["nearest_distances"],
        "nearest_label_distances": label_prediction["nearest_distances"],
        "contributions": label_prediction["contributions"] or family_prediction["contributions"],
        "sequence_channels": model.sequence_channels,
        "backend": model.backend,
    }


def extract_contrastive_features(rows: list[dict[str, float]], windows: int = 4) -> dict[str, float]:
    features: dict[str, float] = {}
    channels = sorted({channel for row in rows for channel in row})
    channel_values: dict[str, list[float]] = {}
    for channel in channels:
        values = [float(row.get(channel, 0.0)) for row in rows]
        if not values:
            continue
        channel_values[channel] = values
        deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
        abs_deltas = [abs(delta) for delta in deltas]
        avg = sum(values) / len(values)
        windows_values = _fixed_windows(values, windows)
        window_medians = [float(median(window)) if window else 0.0 for window in windows_values]
        window_variances = [variance(window) if window else 0.0 for window in windows_values]
        features[f"{channel}_first"] = values[0]
        features[f"{channel}_last"] = values[-1]
        features[f"{channel}_median"] = float(median(values))
        features[f"{channel}_mean"] = avg
        features[f"{channel}_min"] = min(values)
        features[f"{channel}_max"] = max(values)
        features[f"{channel}_variance"] = variance(values)
        features[f"{channel}_slope"] = slope(values)
        features[f"{channel}_roughness"] = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0
        features[f"{channel}_max_abs_delta"] = max(abs_deltas) if abs_deltas else 0.0
        features[f"{channel}_peak_count"] = float(_peak_count(values))
        features[f"{channel}_rolling_variance"] = _rolling_variance(values)
        features[f"{channel}_energy"] = sum(value * value for value in values) / len(values)
        features[f"{channel}_abs_energy"] = sum(abs(value) for value in values) / len(values)
        for index, value in enumerate(window_medians):
            features[f"{channel}_window_{index}_median"] = value
        for index, value in enumerate(window_variances):
            features[f"{channel}_window_{index}_variance"] = value
        for index in range(1, len(window_medians)):
            features[f"{channel}_window_{index}_median_delta"] = window_medians[index] - window_medians[index - 1]
            features[f"{channel}_window_{index}_variance_delta"] = window_variances[index] - window_variances[index - 1]

    for left, right in _safe_channel_pairs(sorted(channel_values)):
        left_mean = sum(channel_values[left]) / len(channel_values[left])
        right_mean = sum(channel_values[right]) / len(channel_values[right])
        denom = max(abs(right_mean), 1.0)
        features[f"{left}__over__{right}_mean_ratio"] = left_mean / denom
        features[f"{left}__minus__{right}_mean_delta"] = left_mean - right_mean
    return features


def scene_family(label: str) -> str:
    normalized = _normalize_label(label)
    if normalized.startswith("baseline_"):
        return "baseline"
    if normalized.startswith("activity_"):
        return "machine_activity"
    if normalized.startswith(("user_", "person_", "walk_", "phone_", "door_")):
        return "user_presence"
    if normalized.startswith("typing") or normalized.startswith("mouse"):
        return "user_presence"
    return "unknown"


def label_to_scene_family(label: str) -> str:
    return scene_family(label)


def _train_profile(project_name: str, *, backend: str, warnings: list[str] | None = None) -> ContrastiveTemporalModel:
    examples = _load_examples(project_name)
    return _train_profile_from_examples(project_name, examples, backend=backend, warnings=warnings)


def _train_profile_from_examples(
    project_name: str,
    examples: list[_SceneExample],
    *,
    backend: str,
    warnings: list[str] | None = None,
    store: SceneFeatureStore | None = None,
) -> ContrastiveTemporalModel:
    label_features: dict[str, list[dict[str, float]]] = {}
    family_features: dict[str, list[dict[str, float]]] = {}
    label_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    all_features: list[dict[str, float]] = []
    all_rows: list[dict[str, float]] = []
    sequence_channels: set[str] = set()

    for example in examples:
        label_counts[example.label] = label_counts.get(example.label, 0) + 1
        family_counts[example.family] = family_counts.get(example.family, 0) + 1
        label_features.setdefault(example.label, []).append(example.features)
        family_features.setdefault(example.family, []).append(example.features)
        all_features.append(example.features)
        all_rows.extend(example.rows)
        sequence_channels.update({channel for row in example.rows for channel in row})

    manifest = feature_manifest_from_store(store, "contrastive_features") if store is not None else feature_manifest(all_features, all_rows)
    if store is not None:
        manifest["dataset_fingerprint"] = store.fingerprint
    manifest["model_version"] = CONTRASTIVE_MODEL_VERSION if backend == "profile" else f"contrastive-{backend}-v2"
    manifest["contrastive_stats"] = [
        "first",
        "last",
        "median",
        "mean",
        "min",
        "max",
        "variance",
        "slope",
        "roughness",
        "max_abs_delta",
        "peak_count",
        "rolling_variance",
        "window_median",
        "window_variance",
        "window_delta",
        "energy",
        "cross_channel_ratio",
    ]
    manifest["training_warnings"] = list(warnings or [])
    return ContrastiveTemporalModel(
        project_name=project_name,
        trained_utc=utc_now_iso(),
        scene_count=len(examples),
        backend=backend,
        label_counts=label_counts,
        family_counts=family_counts,
        label_profiles={label: mean_profile(features) for label, features in label_features.items()},
        family_profiles={family: mean_profile(features) for family, features in family_features.items()},
        sequence_channels=sorted(sequence_channels),
        feature_manifest=manifest,
    )


def _load_examples(project_name: str) -> list[_SceneExample]:
    root = project_path(project_name)
    examples: list[_SceneExample] = []
    for scene_path in sorted((root / "scenes").glob("scene_*/scene.json")):
        try:
            scene = read_json(scene_path)
        except (OSError, ValueError):
            continue
        if scene.get("accepted") is False:
            continue
        preview_path = scene_path.parent / "preview.csv"
        if not preview_path.exists():
            continue
        rows = read_numeric_preview_rows(preview_path)
        if not rows:
            continue
        features = extract_contrastive_features(rows)
        if not features:
            continue
        label = str(scene.get("label", "unknown"))
        examples.append(_SceneExample(str(scene.get("scene_id", scene_path.parent.name)), label, scene_family(label), rows, features))
    return examples


def _predict_profile(profiles: dict[str, dict[str, float]], features: dict[str, float], counts: dict[str, int]) -> dict[str, object]:
    distances: list[tuple[float, str, dict[str, float]]] = []
    for name, profile in profiles.items():
        distance, contributions = feature_distance(features, profile)
        if contributions:
            distances.append((distance, name, contributions))
    if not distances:
        return {"name": "unknown", "confidence": 0.0, "distance": 0.0, "nearest_distances": {}, "contributions": {}}
    distances.sort(key=lambda item: (item[0], item[1]))
    distance, name, contributions = distances[0]
    second = distances[1][0] if len(distances) > 1 else None
    nearest = {label: round(value, 6) for value, label, _ in distances[:5]}
    confidence = _distance_gap_confidence(distance, second, counts.get(name, 0), sum(counts.values()))
    return {
        "name": name,
        "confidence": confidence,
        "distance": round(distance, 6),
        "nearest_distances": nearest,
        "contributions": dict(sorted(contributions.items(), key=lambda item: item[1], reverse=True)[:8]),
    }


def _combined_confidence(family_prediction: dict[str, object], label_prediction: dict[str, object], scene_count: int) -> float:
    family_confidence = float(family_prediction.get("confidence", 0.0) or 0.0)
    label_confidence = float(label_prediction.get("confidence", 0.0) or 0.0)
    confidence = family_confidence * 0.62 + label_confidence * 0.38
    if scene_count < 3:
        confidence = min(confidence, 0.25)
    elif scene_count < 6:
        confidence = min(confidence, 0.55)
    return round(max(0.0, min(1.0, confidence)), 3)


def _distance_gap_confidence(distance: float, second: float | None, count: int, total: int) -> float:
    base = 1.0 / (1.0 + max(distance, 0.0))
    if second is not None:
        gap = max(0.0, second - distance)
        base *= min(1.0, gap / max(second, 1.0))
    else:
        base *= 0.35
    if count < 2:
        base *= 0.55
    if total < 4:
        base *= 0.65
    return round(max(0.0, min(0.85, base)), 3)


def _train_torch_tcn_or_fallback(project_name: str, backend: str) -> ContrastiveTemporalModel:
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ImportError:
        return _train_profile(project_name, backend="profile", warnings=[f"requested {backend} but PyTorch is not installed; install with python -m pip install -e \".[ml]\""])

    examples = _load_examples(project_name)
    family_count = len({example.family for example in examples})
    if len(examples) < 3 or family_count < 2:
        return _train_profile(project_name, backend="profile", warnings=[f"requested {backend} but at least 3 scenes across 2 families are needed for TCN training"])

    torch.manual_seed(13)
    random.seed(13)
    channels = sorted({channel for example in examples for row in example.rows for channel in row})
    length = 64
    tensors = [_rows_to_tensor(torch, example.rows, channels, length) for example in examples]
    encoder = _build_tcn_encoder(nn, len(channels), hidden=32, embedding=32)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.003, weight_decay=1e-4)
    pairs = _contrastive_pairs(examples, limit=256)
    for _ in range(8):
        random.shuffle(pairs)
        for left_index, right_index, target in pairs:
            left_embedding = encoder(tensors[left_index].unsqueeze(0))
            right_embedding = encoder(tensors[right_index].unsqueeze(0))
            distance = functional.pairwise_distance(left_embedding, right_embedding)
            target_tensor = torch.tensor([float(target)])
            loss = target_tensor * distance.pow(2) + (1.0 - target_tensor) * torch.clamp(1.2 - distance, min=0.0).pow(2)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

    with torch.no_grad():
        embeddings = [encoder(tensor.unsqueeze(0)).squeeze(0).detach() for tensor in tensors]

    label_embeddings: dict[str, list[object]] = {}
    family_embeddings: dict[str, list[object]] = {}
    label_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for example, embedding in zip(examples, embeddings):
        label_counts[example.label] = label_counts.get(example.label, 0) + 1
        family_counts[example.family] = family_counts.get(example.family, 0) + 1
        label_embeddings.setdefault(example.label, []).append(embedding)
        family_embeddings.setdefault(example.family, []).append(embedding)

    def prototype(items: list[object]) -> dict[str, float]:
        stacked = torch.stack(items)
        mean = stacked.mean(dim=0)
        return {f"e{index}": float(value) for index, value in enumerate(mean.tolist())}

    weights_path = project_path(project_name) / "exports" / TCN_WEIGHTS_NAME
    ensure_dir(weights_path.parent)
    torch.save({"state_dict": encoder.state_dict(), "channels": channels, "length": length, "hidden": 32, "embedding": 32}, weights_path)

    all_features = [example.features for example in examples]
    all_rows = [row for example in examples for row in example.rows]
    manifest = feature_manifest(all_features, all_rows)
    manifest.update({
        "requested_backend": backend,
        "tcn_weights": TCN_WEIGHTS_NAME,
        "embedding_size": 32,
        "sequence_length": length,
        "hidden_size": 32,
        "training_warnings": [],
    })
    return ContrastiveTemporalModel(
        project_name=project_name,
        trained_utc=utc_now_iso(),
        scene_count=len(examples),
        backend="torch_tcn",
        label_counts=label_counts,
        family_counts=family_counts,
        label_profiles={label: prototype(items) for label, items in label_embeddings.items()},
        family_profiles={family: prototype(items) for family, items in family_embeddings.items()},
        sequence_channels=channels,
        feature_manifest=manifest,
    )


def _predict_scene_tcn(model: ContrastiveTemporalModel, preview_path: Path) -> dict[str, object]:
    try:
        import torch
        from torch import nn
    except ImportError:
        return _unknown_prediction("PyTorch is unavailable for torch_tcn prediction", backend=model.backend, channels=model.sequence_channels)
    if not preview_path.exists():
        return _unknown_prediction("missing preview rows", backend=model.backend, channels=model.sequence_channels)
    rows = read_numeric_preview_rows(preview_path)
    if not rows:
        return _unknown_prediction("missing preview rows", backend=model.backend, channels=model.sequence_channels)
    weights_name = str(model.feature_manifest.get("tcn_weights", TCN_WEIGHTS_NAME))
    weights_path = project_path(model.project_name) / "exports" / weights_name
    if not weights_path.exists():
        return _unknown_prediction(f"missing TCN weights: {weights_name}", backend=model.backend, channels=model.sequence_channels)
    try:
        state = torch.load(weights_path, map_location="cpu")
        channels = [str(channel) for channel in list(state.get("channels", model.sequence_channels))]
        length = int(state.get("length", model.feature_manifest.get("sequence_length", 64)) or 64)
        hidden = int(state.get("hidden", model.feature_manifest.get("hidden_size", 32)) or 32)
        embedding = int(state.get("embedding", model.feature_manifest.get("embedding_size", 32)) or 32)
        encoder = _build_tcn_encoder(nn, len(channels), hidden=hidden, embedding=embedding)
        encoder.load_state_dict(state["state_dict"])
        encoder.eval()
        with torch.no_grad():
            vector = encoder(_rows_to_tensor(torch, rows, channels, length).unsqueeze(0)).squeeze(0).tolist()
    except Exception as exc:
        return _unknown_prediction(f"TCN prediction failed: {exc}", backend=model.backend, channels=model.sequence_channels)

    features = {f"e{index}": float(value) for index, value in enumerate(vector)}
    family_prediction = _predict_profile(model.family_profiles, features, model.family_counts)
    label_prediction = _predict_profile(model.label_profiles, features, model.label_counts)
    return {
        "family": family_prediction["name"],
        "label": label_prediction["name"],
        "confidence": _combined_confidence(family_prediction, label_prediction, model.scene_count),
        "family_distance": family_prediction["distance"],
        "label_distance": label_prediction["distance"],
        "nearest_family_distances": family_prediction["nearest_distances"],
        "nearest_label_distances": label_prediction["nearest_distances"],
        "contributions": label_prediction["contributions"] or family_prediction["contributions"],
        "sequence_channels": model.sequence_channels,
        "backend": model.backend,
    }


def _build_tcn_encoder(nn, input_channels: int, hidden: int, embedding: int):
    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int):
            super().__init__()
            padding = dilation
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            )
            self.activation = nn.ReLU()

        def forward(self, value):
            return self.activation(value + self.net(value))

    class TCNEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(max(1, input_channels), hidden, kernel_size=1),
                nn.ReLU(),
                ResidualBlock(hidden, 1),
                ResidualBlock(hidden, 2),
                ResidualBlock(hidden, 4),
                ResidualBlock(hidden, 8),
            )
            self.projection = nn.Linear(hidden, embedding)

        def forward(self, value):
            encoded = self.net(value)
            pooled = encoded.mean(dim=-1)
            return self.projection(pooled)

    return TCNEncoder()


def _rows_to_tensor(torch, rows: list[dict[str, float]], channels: list[str], length: int):
    matrix = []
    for channel in channels:
        values = [float(row.get(channel, 0.0)) for row in rows]
        values = _resample(values, length)
        avg = sum(values) / len(values) if values else 0.0
        centered = [value - avg for value in values]
        scale = math.sqrt(sum(value * value for value in centered) / len(centered)) if centered else 1.0
        scale = scale or 1.0
        matrix.append([value / scale for value in centered])
    if not matrix:
        matrix = [[0.0 for _ in range(length)]]
    return torch.tensor(matrix, dtype=torch.float32)


def _contrastive_pairs(examples: list[_SceneExample], limit: int) -> list[tuple[int, int, int]]:
    positives: list[tuple[int, int, int]] = []
    negatives: list[tuple[int, int, int]] = []
    for left in range(len(examples)):
        for right in range(left + 1, len(examples)):
            if examples[left].label == examples[right].label or examples[left].family == examples[right].family:
                positives.append((left, right, 1))
            elif examples[left].family != examples[right].family:
                negatives.append((left, right, 0))
    random.shuffle(positives)
    random.shuffle(negatives)
    count = min(limit // 2, len(positives), len(negatives))
    if count == 0:
        return positives[:limit] + negatives[:limit]
    return positives[:count] + negatives[:count]


def _resample(values: list[float], length: int) -> list[float]:
    if not values:
        return [0.0 for _ in range(length)]
    if len(values) == length:
        return values
    if length <= 1:
        return [values[0]]
    out = []
    for index in range(length):
        position = index * (len(values) - 1) / (length - 1)
        left = int(math.floor(position))
        right = min(len(values) - 1, left + 1)
        weight = position - left
        out.append(values[left] * (1.0 - weight) + values[right] * weight)
    return out


def _unknown_prediction(reason: str = "", *, backend: str = "profile", channels: list[str] | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "family": "unknown",
        "label": "unknown",
        "confidence": 0.0,
        "family_distance": 0.0,
        "label_distance": 0.0,
        "nearest_family_distances": {},
        "nearest_label_distances": {},
        "contributions": {},
        "sequence_channels": list(channels or []),
        "backend": backend,
    }
    if reason:
        result["warning"] = reason
    return result


def _fixed_windows(values: list[float], windows: int) -> list[list[float]]:
    count = max(1, windows)
    if len(values) <= count:
        return [[value] for value in values]
    out: list[list[float]] = []
    for index in range(count):
        start = int(round(index * len(values) / count))
        end = int(round((index + 1) * len(values) / count))
        out.append(values[start:max(start + 1, end)])
    return out


def _rolling_variance(values: list[float], window: int = 5) -> float:
    if not values:
        return 0.0
    width = max(2, min(window, len(values)))
    variances = [variance(values[index:index + width]) for index in range(0, len(values) - width + 1)]
    return float(sum(variances) / len(variances)) if variances else 0.0


def _peak_count(values: list[float]) -> int:
    if len(values) < 3:
        return 0
    count = 0
    for previous, current, following in zip(values, values[1:], values[2:]):
        if current > previous and current > following:
            count += 1
    return count


def _safe_channel_pairs(channels: list[str]) -> list[tuple[str, str]]:
    priority = ["sleep_drift_ns", "process_ns_estimate", "cpu_load_ppm", "dt_ns"]
    selected = [channel for channel in priority if channel in channels]
    selected.extend(channel for channel in channels if channel not in selected)
    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(selected[:4]):
        for right in selected[index + 1:4]:
            pairs.append((left, right))
    return pairs[:6]


def _float_profile_map(value: object) -> dict[str, dict[str, float]]:
    return {
        str(label): {str(k): float(v) for k, v in dict(profile).items()}
        for label, profile in dict(value).items()
    }


def _normalize_backend(backend: str) -> str:
    normalized = str(backend).strip().lower().replace("-", "_")
    return "torch_tcn" if normalized == "tcn" else normalized


def _normalize_label(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_") or "unknown"
