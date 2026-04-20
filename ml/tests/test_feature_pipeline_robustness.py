from __future__ import annotations

import pandas as pd

from ml.config import FEATURE_COLUMNS
from ml.features import build_feature_table


def test_build_feature_table_filters_invalid_rows_and_keeps_shape() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "match_date": "2024-04-01",
                "tournament": "IPL",
                "team_a": "Mumbai Indians",
                "team_b": "Chennai Super Kings",
                "venue": "Wankhede Stadium",
                "winner": "Mumbai Indians",
                "first_innings_team": "Mumbai Indians",
                "first_innings_total": 178,
                "first_innings_wickets": 6,
                "second_innings_team": "Chennai Super Kings",
                "second_innings_total": 171,
                "second_innings_wickets": 8,
            },
            {
                "match_id": "m2",
                "match_date": "2024-04-03",
                "tournament": "IPL",
                "team_a": "Mumbai Indians",
                "team_b": "Mumbai Indians",
                "venue": "Wankhede Stadium",
                "winner": "Mumbai Indians",
                "first_innings_team": "Mumbai Indians",
                "first_innings_total": 165,
                "first_innings_wickets": 7,
                "second_innings_team": "Mumbai Indians",
                "second_innings_total": 160,
                "second_innings_wickets": 8,
            },
        ]
    )
    players = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "match_date": "2024-04-01",
                "team": "Mumbai Indians",
                "opponent": "Chennai Super Kings",
                "venue": "Wankhede Stadium",
                "player": "Rohit Sharma",
                "runs": 52,
                "balls": 35,
                "wickets": 0,
                "balls_bowled": 0,
                "runs_conceded": 0,
                "is_winner": 1,
            },
            {
                "match_id": "m1",
                "match_date": "2024-04-01",
                "team": "Chennai Super Kings",
                "opponent": "Mumbai Indians",
                "venue": "Wankhede Stadium",
                "player": "Ruturaj Gaikwad",
                "runs": 48,
                "balls": 34,
                "wickets": 0,
                "balls_bowled": 0,
                "runs_conceded": 0,
                "is_winner": 0,
            },
        ]
    )

    feature_table = build_feature_table(matches_df=matches, players_df=players, history_window=5)
    assert not feature_table.empty
    assert set(FEATURE_COLUMNS).issubset(set(feature_table.columns))
    assert {"target_team_a_score", "target_team_b_score", "target_winner_team_a"}.issubset(
        set(feature_table.columns)
    )
    assert "m2" not in set(feature_table["match_id"].tolist())
