from __future__ import annotations

from app.core.config import get_settings
from app.services.name_resolver_service import get_player_name_resolver


def test_name_resolver_replaces_placeholder_with_real_name() -> None:
    resolver = get_player_name_resolver(str(get_settings().data_processed_dir))
    resolved, replaced = resolver.resolve(
        name="Mumbai Player 1",
        sport="cricket",
        team="Mumbai Indians",
        role="batsman",
        seed="pytest|name-resolution",
    )

    assert replaced is True
    assert resolved
    assert "player 1" not in resolved.lower()
    assert resolved != "Unavailable"


def test_name_resolver_avoids_placeholder_pattern_on_unknown_team() -> None:
    resolver = get_player_name_resolver(str(get_settings().data_processed_dir))
    resolved, replaced = resolver.resolve(
        name="Top Batter",
        sport="cricket",
        team="Unknown XI",
        role="batsman",
        seed="pytest|unknown-team",
    )

    assert replaced is True
    assert resolved
    assert "player" not in resolved.lower()


def test_name_resolver_football_unknown_team_uses_real_player_pool() -> None:
    resolver = get_player_name_resolver(str(get_settings().data_processed_dir))
    resolved, replaced = resolver.resolve(
        name="Goal Scorer",
        sport="football",
        team="Unknown FC",
        role="goal_scorer",
        seed="pytest|football-unknown",
    )

    assert replaced is True
    assert resolved
    assert resolved != "Unavailable"
