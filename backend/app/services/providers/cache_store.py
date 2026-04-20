from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_seconds(updated_at: str) -> float:
    try:
        ts = datetime.fromisoformat(updated_at)
    except ValueError:
        return float("inf")
    return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())


@dataclass
class CacheHit:
    key: str
    payload: Any
    updated_at: str
    age_seconds: float


class JSONFileCacheStore:
    def __init__(self, path: Path, max_entries: int = 300) -> None:
        self.path = path
        self.max_entries = max_entries
        self._lock = RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    def get(self, key: str, max_age_seconds: Optional[int] = None) -> CacheHit | None:
        with self._lock:
            payload = self._load()
            row = payload.get(key)
            if not isinstance(row, dict):
                return None

            updated_at = str(row.get("updated_at") or "")
            if not updated_at:
                return None
            age = _age_seconds(updated_at)
            if max_age_seconds is not None and age > float(max_age_seconds):
                return None

            return CacheHit(
                key=key,
                payload=row.get("payload"),
                updated_at=updated_at,
                age_seconds=age,
            )

    def set(self, key: str, value: Any) -> CacheHit:
        with self._lock:
            payload = self._load()
            payload[key] = {"updated_at": _utc_now_iso(), "payload": value}
            if len(payload) > self.max_entries:
                sorted_keys = sorted(
                    payload.keys(),
                    key=lambda item: str(payload[item].get("updated_at", "")),
                )
                prune_count = len(payload) - self.max_entries
                for prune_key in sorted_keys[:prune_count]:
                    payload.pop(prune_key, None)
            self._save(payload)

            hit = payload[key]
            return CacheHit(
                key=key,
                payload=hit.get("payload"),
                updated_at=str(hit.get("updated_at")),
                age_seconds=0.0,
            )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            payload = self._load()
            if not payload:
                return {"entries": 0, "latest_updated_at": None}
            latest = max(str(row.get("updated_at", "")) for row in payload.values() if isinstance(row, dict))
            return {"entries": len(payload), "latest_updated_at": latest or None}
