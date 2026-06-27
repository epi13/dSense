from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from dsense.manifest import project_path
from dsense.models.features import feature_distance, mean_profile, summarize_preview
from dsense.utils.files import ensure_dir, read_json, write_json
from dsense.utils.timebase import utc_now_iso


@dataclass(frozen=True)
class SceneSample:
    scene_id: str
    label: str
    created_utc: str
    scene_dir: Path
    features: dict[str, float]


def evaluation_report_path(project_name: str) -> Path:
    return project_path(project_name) / "exports" / "evaluation_report.json"


def evaluate_project_scenes(project_name: str) -> dict[str, object]:
    samples = load_scene_samples(project_name)
    label_counts = _label_counts(samples)
    report = {
        "format": "dsense-evaluation-v1",
        "project_name": project_name,
        "created_utc": utc_now_iso(),
        "scene_count": len(samples),
        "label_counts": label_counts,
        "within_label_similarity": _within_label_similarity(samples),
        "between_label_distance": _between_label_distance(samples),
        "confusion_matrix": _leave_one_out_confusion(samples),
        "baseline_drift": _baseline_drift(samples),
        "channel_usefulness_ranking": _channel_usefulness(samples),
    }
    out = evaluation_report_path(project_name)
    ensure_dir(out.parent)
    write_json(out, report)
    return report


def load_scene_samples(project_name: str) -> list[SceneSample]:
    samples: list[SceneSample] = []
    root = project_path(project_name)
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
        features = summarize_preview(preview_path)
        if not features:
            continue
        samples.append(SceneSample(
            scene_id=str(scene.get("scene_id", scene_path.parent.name)),
            label=str(scene.get("label", "unknown")),
            created_utc=str(scene.get("created_utc", "")),
            scene_dir=scene_path.parent,
            features=features,
        ))
    return samples


def predict_from_profiles(label_profiles: dict[str, dict[str, float]], features: dict[str, float]) -> dict[str, object]:
    distances = []
    for label, profile in label_profiles.items():
        distance, contributions = feature_distance(features, profile)
        if contributions:
            distances.append((distance, label, contributions))
    if not distances:
        return {"label": "unknown", "confidence": 0.0, "distance": 0.0, "contributions": {}}
    distance, label, contributions = min(distances, key=lambda item: (item[0], item[1]))
    return {
        "label": label,
        "confidence": round(1.0 / (1.0 + distance), 3),
        "distance": round(distance, 6),
        "contributions": dict(sorted(contributions.items(), key=lambda item: item[1], reverse=True)[:5]),
    }


def print_evaluation_report(report: dict[str, object]) -> None:
    print(f"Evaluation report: {report['project_name']}")
    print(f"Scenes: {report['scene_count']}")
    print(f"Labels: {report['label_counts']}")
    print("")
    print(f"{'Metric':<28} {'Value':<14} Why it matters")
    print(f"{'-' * 28} {'-' * 14} {'-' * 42}")
    print(f"{'within-label similarity':<28} {_fmt(report.get('within_label_similarity')):<14} repeated takes should agree")
    print(f"{'between-label distance':<28} {_fmt(report.get('between_label_distance')):<14} labels should separate")
    drift = dict(report.get("baseline_drift", {})).get("max_drift", 0.0)
    print(f"{'baseline drift over time':<28} {float(drift):<14.3f} machine stability check")
    ranking = report.get("channel_usefulness_ranking", [])
    top = ", ".join(str(item.get("channel")) for item in list(ranking)[:3]) if isinstance(ranking, list) else "none"
    print(f"{'channel usefulness ranking':<28} {top or 'none':<14} channels that separate labels")
    print("")
    print("Confusion matrix:")
    matrix = dict(report.get("confusion_matrix", {})).get("matrix", {})
    labels = sorted(matrix)
    if not labels:
        print("  none")
        return
    print(f"{'actual':<24} predicted counts")
    for actual in labels:
        print(f"{actual:<24} {matrix[actual]}")


def _label_counts(samples: list[SceneSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.label] = counts.get(sample.label, 0) + 1
    return counts


def _within_label_similarity(samples: list[SceneSample]) -> dict[str, object]:
    by_label = _samples_by_label(samples)
    label_scores: dict[str, float] = {}
    all_scores: list[float] = []
    for label, label_samples in by_label.items():
        distances = [_distance(a, b) for a, b in combinations(label_samples, 2)]
        if distances:
            similarity = sum(1.0 / (1.0 + distance) for distance in distances) / len(distances)
            label_scores[label] = round(similarity, 6)
            all_scores.append(similarity)
    return {"overall": round(sum(all_scores) / len(all_scores), 6) if all_scores else 0.0, "labels": label_scores}


def _between_label_distance(samples: list[SceneSample]) -> dict[str, object]:
    distances = [
        _distance(left, right)
        for left, right in combinations(samples, 2)
        if left.label != right.label
    ]
    return {
        "average": round(sum(distances) / len(distances), 6) if distances else 0.0,
        "pair_count": len(distances),
    }


def _leave_one_out_confusion(samples: list[SceneSample]) -> dict[str, object]:
    matrix: dict[str, dict[str, int]] = {}
    correct = 0
    evaluated = 0
    for holdout in samples:
        training = [sample for sample in samples if sample.scene_id != holdout.scene_id]
        profiles = {
            label: mean_profile([sample.features for sample in label_samples])
            for label, label_samples in _samples_by_label(training).items()
        }
        prediction = predict_from_profiles(profiles, holdout.features)
        predicted = str(prediction["label"])
        matrix.setdefault(holdout.label, {})
        matrix[holdout.label][predicted] = matrix[holdout.label].get(predicted, 0) + 1
        evaluated += 1
        if predicted == holdout.label:
            correct += 1
    return {
        "accuracy": round(correct / evaluated, 6) if evaluated else 0.0,
        "evaluated": evaluated,
        "matrix": matrix,
    }


def _baseline_drift(samples: list[SceneSample]) -> dict[str, object]:
    baseline = [sample for sample in samples if sample.label.startswith("baseline_")]
    baseline.sort(key=lambda sample: (sample.created_utc, sample.scene_id))
    drifts: list[float] = []
    pairs: list[dict[str, object]] = []
    for previous, current in zip(baseline, baseline[1:]):
        distance = _distance(previous, current)
        drifts.append(distance)
        pairs.append({"from": previous.scene_id, "to": current.scene_id, "drift": round(distance, 6)})
    return {
        "baseline_scene_count": len(baseline),
        "max_drift": round(max(drifts), 6) if drifts else 0.0,
        "average_drift": round(sum(drifts) / len(drifts), 6) if drifts else 0.0,
        "pairs": pairs,
    }


def _channel_usefulness(samples: list[SceneSample]) -> list[dict[str, object]]:
    channels = sorted({
        key.removesuffix("_median")
        for sample in samples
        for key in sample.features
        if key.endswith("_median")
    })
    by_label = _samples_by_label(samples)
    ranking = []
    for channel in channels:
        key = f"{channel}_median"
        label_means = []
        within_values = []
        for label_samples in by_label.values():
            values = [sample.features.get(key, 0.0) for sample in label_samples]
            if not values:
                continue
            mean = sum(values) / len(values)
            label_means.append(mean)
            within_values.extend(abs(value - mean) for value in values)
        if len(label_means) < 2:
            score = 0.0
        else:
            overall = sum(label_means) / len(label_means)
            between = sum(abs(value - overall) for value in label_means) / len(label_means)
            within = (sum(within_values) / len(within_values)) if within_values else 0.0
            score = between / max(within, 1.0)
        ranking.append({"channel": channel, "score": round(score, 6)})
    return sorted(ranking, key=lambda item: (-float(item["score"]), str(item["channel"])))


def _samples_by_label(samples: list[SceneSample]) -> dict[str, list[SceneSample]]:
    by_label: dict[str, list[SceneSample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)
    return by_label


def _distance(left: SceneSample, right: SceneSample) -> float:
    distance, _ = feature_distance(left.features, right.features)
    return distance


def _fmt(value: object) -> str:
    if isinstance(value, dict):
        for key in ("overall", "average"):
            if key in value:
                return f"{float(value[key]):.3f}"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "n/a"
