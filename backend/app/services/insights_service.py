from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.services.catalog_service import CatalogService
from app.services.name_resolver_service import get_player_name_resolver
from app.services.providers.factory import get_live_provider


class InsightsService:
    @staticmethod
    def _resolver():
        from app.core.config import get_settings

        settings = get_settings()
        return get_player_name_resolver(str(settings.data_processed_dir))

    @staticmethod
    def live_insights(
        sport: str,
        tournament: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 5,
    ) -> dict:
        sport_key = str(sport or "cricket").lower()
        if sport_key != "cricket":
            return {
                "provider": "not-supported",
                "as_of": datetime.now(timezone.utc).isoformat(),
                "cards": [],
            }

        provider = get_live_provider()
        chosen_state = str(state or "").strip().lower()
        states = [chosen_state] if chosen_state in {"live", "upcoming"} else ["live", "upcoming"]

        rows: list[dict] = []
        try:
            if "live" in states:
                rows.extend(provider.fetch_live_matches(tournament=tournament, limit=max(1, min(limit, 12))))
            if "upcoming" in states:
                rows.extend(provider.fetch_upcoming_matches(tournament=tournament, limit=max(1, min(limit, 12))))
        except Exception:  # noqa: BLE001
            rows = []

        deduped: dict[str, dict] = {}
        for row in rows:
            match_id = str(row.get("match_id") or "")
            if not match_id:
                continue
            deduped[match_id] = row
        selected = list(deduped.values())[: max(1, min(limit, 12))]

        cards = []
        for row in selected:
            score_rows = row.get("score") if isinstance(row.get("score"), list) else []
            first_score = score_rows[0] if score_rows else {}
            runs = first_score.get("r")
            wickets = first_score.get("w")
            overs = first_score.get("o")
            state_text = str(row.get("state") or "upcoming")
            status_text = str(row.get("status_text") or "").strip()
            if isinstance(runs, (int, float)):
                summary = f"{row.get('team_a')} vs {row.get('team_b')} | {int(runs)}/{int(wickets or 0)} in {overs or 0} overs"
            elif status_text:
                summary = f"{row.get('team_a')} vs {row.get('team_b')} | {status_text}"
            else:
                summary = f"{row.get('team_a')} vs {row.get('team_b')} | {state_text}"

            reasons = [
                {
                    "feature": "state",
                    "value": state_text,
                    "impact": "neutral",
                    "explanation": f"state: {state_text}",
                },
            ]
            if isinstance(runs, (int, float)):
                reasons.append(
                    {
                        "feature": "runs",
                        "value": float(runs),
                        "impact": "positive" if float(runs) >= 165 else "neutral",
                        "explanation": f"live_runs: {float(runs):.0f}",
                    }
                )
            freshness = row.get("freshness_seconds")
            if isinstance(freshness, (int, float)):
                reasons.append(
                    {
                        "feature": "freshness_seconds",
                        "value": float(freshness),
                        "impact": "positive" if float(freshness) <= 600 else "neutral",
                        "explanation": f"freshness_seconds: {float(freshness):.0f}",
                    }
                )

            cards.append(
                {
                    "match_id": str(row.get("match_id")),
                    "sport": "cricket",
                    "tournament": str(row.get("tournament") or ""),
                    "team_a": str(row.get("team_a") or ""),
                    "team_b": str(row.get("team_b") or ""),
                    "state": state_text,
                    "summary": summary,
                    "reasons": reasons,
                }
            )

        return {
            "provider": getattr(provider, "provider_name", "unknown"),
            "as_of": datetime.now(timezone.utc).isoformat(),
            "cards": cards,
        }

    @staticmethod
    def top_players(
        sport: str,
        tournament: Optional[str] = None,
        team: Optional[str] = None,
        role: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict]:
        rows = CatalogService.top_players(
            sport=sport,
            tournament=tournament,
            team=team,
            role=role,
            limit=limit,
        )

        top: list[dict] = []
        resolver = InsightsService._resolver()
        for row in rows:
            sport_code = sport.lower()
            if sport_code == "football":
                reasons = [
                    {
                        "feature": "goals_last5",
                        "value": float(row.get("goals_last5", 0.0)),
                        "impact": "positive",
                        "explanation": "goals_last5 trend is strong.",
                    },
                    {
                        "feature": "xg_per90",
                        "value": float(row.get("xg_per90", 0.0)),
                        "impact": "positive",
                        "explanation": "xg_per90 supports repeatability.",
                    },
                    {
                        "feature": "impact_score",
                        "value": float(row.get("impact_score", 0.0)),
                        "impact": "positive",
                        "explanation": "impact_score above baseline.",
                    },
                ]
            else:
                reasons = [
                    {
                        "feature": "batting_form",
                        "value": float(row.get("batting_form", 0.0)),
                        "impact": "positive",
                        "explanation": "batting_form trend is strong.",
                    },
                    {
                        "feature": "bowling_form",
                        "value": float(row.get("bowling_form", 0.0)),
                        "impact": "positive" if float(row.get("bowling_form", 0.0)) >= 20 else "neutral",
                        "explanation": "bowling_form adds all-round value.",
                    },
                    {
                        "feature": "impact_score",
                        "value": float(row.get("impact_score", 0.0)),
                        "impact": "positive",
                        "explanation": "impact_score above baseline.",
                    },
                ]

            impact = float(row.get("rank_score", row.get("impact_score", 0.0)))
            confidence = max(0.35, min(0.92, impact / 100.0))
            resolved_name, replaced = resolver.resolve(
                name=str(row.get("player", row.get("name", ""))),
                sport=sport,
                team=str(row.get("team", team or "")),
                role=str(row.get("role", role or "standout")),
                seed=f"insights|{sport}|{row.get('team')}|{row.get('role')}|{row.get('player')}",
            )
            top.append(
                {
                    "name": resolved_name,
                    "team": str(row.get("team", "Unknown Team")),
                    "sport": sport,
                    "tournament": str(row.get("tournament", tournament or "")),
                    "role": str(row.get("role", role or "standout")),
                    "impact_score": round(impact, 3),
                    "confidence": round(confidence, 3),
                    "reasons": reasons,
                }
            )

        return top
