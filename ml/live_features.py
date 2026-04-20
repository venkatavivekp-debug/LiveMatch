from __future__ import annotations

from typing import Any


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def blend_cricket_features(
    historical_features: dict[str, float],
    live_features: dict[str, Any] | None,
    recency_weight: float = 0.65,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Blend historical feature baselines with live match-day context.

    The model feature space remains unchanged; this function only updates
    feature values before TimeMCL inference.
    """
    if not isinstance(historical_features, dict):
        raise ValueError("historical_features must be a dictionary")

    live_features = live_features or {}
    clamped_weight = max(0.0, min(1.0, float(recency_weight)))

    blended = dict(historical_features)
    live_used = 0
    contributions: list[dict[str, Any]] = []

    for feature_name, historical_value in historical_features.items():
        live_value = live_features.get(feature_name)
        if not _is_number(live_value):
            continue
        live_used += 1
        updated = ((1.0 - clamped_weight) * float(historical_value)) + (clamped_weight * float(live_value))
        blended[feature_name] = float(updated)
        contributions.append(
            {
                "feature": feature_name,
                "historical": round(float(historical_value), 4),
                "live": round(float(live_value), 4),
                "blended": round(float(updated), 4),
            }
        )

    data_mode = "HISTORICAL"
    if live_used > 0:
        data_mode = "HYBRID"

    meta = {
        "data_mode": data_mode,
        "recency_weight": round(clamped_weight, 3),
        "live_feature_count": int(live_used),
        "historical_feature_count": int(len(historical_features)),
        "contributions": contributions[:8],
    }
    return blended, meta
