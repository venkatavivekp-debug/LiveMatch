from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None  # type: ignore[assignment]

from ml.config import FEATURE_COLUMNS, MLConfig, ensure_directories

IPL_TEAM_ROSTERS: dict[str, list[str]] = {
    "Mumbai Indians": [
        "Rohit Sharma",
        "Ishan Kishan",
        "Suryakumar Yadav",
        "Hardik Pandya",
        "Tilak Varma",
        "Tim David",
        "Jasprit Bumrah",
        "Piyush Chawla",
        "Gerald Coetzee",
        "Nuwan Thushara",
        "Akash Madhwal",
    ],
    "Chennai Super Kings": [
        "Ruturaj Gaikwad",
        "Devon Conway",
        "Ajinkya Rahane",
        "Shivam Dube",
        "Ravindra Jadeja",
        "MS Dhoni",
        "Daryl Mitchell",
        "Moeen Ali",
        "Deepak Chahar",
        "Matheesha Pathirana",
        "Tushar Deshpande",
    ],
    "Royal Challengers Bengaluru": [
        "Virat Kohli",
        "Faf du Plessis",
        "Glenn Maxwell",
        "Rajat Patidar",
        "Cameron Green",
        "Dinesh Karthik",
        "Mohammed Siraj",
        "Yash Dayal",
        "Karn Sharma",
        "Reece Topley",
        "Mayank Dagar",
    ],
    "Kolkata Knight Riders": [
        "Shreyas Iyer",
        "Sunil Narine",
        "Andre Russell",
        "Rinku Singh",
        "Venkatesh Iyer",
        "Phil Salt",
        "Nitish Rana",
        "Varun Chakaravarthy",
        "Mitchell Starc",
        "Harshit Rana",
        "Anukul Roy",
    ],
    "Rajasthan Royals": [
        "Sanju Samson",
        "Yashasvi Jaiswal",
        "Jos Buttler",
        "Riyan Parag",
        "Shimron Hetmyer",
        "Ravichandran Ashwin",
        "Trent Boult",
        "Yuzvendra Chahal",
        "Sandeep Sharma",
        "Avesh Khan",
        "Dhruv Jurel",
    ],
    "Sunrisers Hyderabad": [
        "Pat Cummins",
        "Travis Head",
        "Abhishek Sharma",
        "Aiden Markram",
        "Heinrich Klaasen",
        "Nitish Kumar Reddy",
        "Bhuvneshwar Kumar",
        "T Natarajan",
        "Mayank Markande",
        "Abdul Samad",
        "Rahul Tripathi",
    ],
    "Delhi Capitals": [
        "Rishabh Pant",
        "David Warner",
        "Prithvi Shaw",
        "Jake Fraser-McGurk",
        "Tristan Stubbs",
        "Axar Patel",
        "Kuldeep Yadav",
        "Khaleel Ahmed",
        "Anrich Nortje",
        "Mukesh Kumar",
        "Mitchell Marsh",
    ],
    "Punjab Kings": [
        "Shikhar Dhawan",
        "Jonny Bairstow",
        "Liam Livingstone",
        "Sam Curran",
        "Jitesh Sharma",
        "Shashank Singh",
        "Arshdeep Singh",
        "Kagiso Rabada",
        "Rahul Chahar",
        "Harpreet Brar",
        "Prabhsimran Singh",
    ],
    "Lucknow Super Giants": [
        "KL Rahul",
        "Quinton de Kock",
        "Nicholas Pooran",
        "Marcus Stoinis",
        "Deepak Hooda",
        "Krunal Pandya",
        "Ravi Bishnoi",
        "Naveen-ul-Haq",
        "Mohsin Khan",
        "Ayush Badoni",
        "Yash Thakur",
    ],
    "Gujarat Titans": [
        "Shubman Gill",
        "Wriddhiman Saha",
        "Sai Sudharsan",
        "David Miller",
        "Rahul Tewatia",
        "Rashid Khan",
        "Noor Ahmad",
        "Mohit Sharma",
        "Umesh Yadav",
        "Azmatullah Omarzai",
        "Kane Williamson",
    ],
}


@dataclass
class TeamMetrics:
    avg_runs_scored: float
    avg_wickets_lost: float
    avg_runs_conceded: float
    avg_wickets_taken: float
    win_rate: float


def _team_score_from_row(row: pd.Series, team: str) -> float | None:
    if row.get("first_innings_team") == team:
        value = row.get("first_innings_total")
    elif row.get("second_innings_team") == team:
        value = row.get("second_innings_total")
    else:
        return None
    if pd.isna(value):
        return None
    return float(value)


def _core_dependencies_ready() -> bool:
    return np is not None and pd is not None


def _require_core_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if pd is None:
        missing.append("pandas")
    if missing:
        raise RuntimeError(
            "Missing ML dependencies: "
            + ", ".join(missing)
            + ". Run scripts/setup_ml.sh to install requirements. "
            "Running in fallback mode (no trained model available)."
        )


def _clean_text(value: Any, default: str = "") -> str:
    text = " ".join(str(value or "").strip().split())
    return text or default


def _normalize_matches_input(matches_df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "match_id",
        "match_date",
        "tournament",
        "team_a",
        "team_b",
        "venue",
        "winner",
        "first_innings_team",
        "first_innings_total",
        "first_innings_wickets",
        "second_innings_team",
        "second_innings_total",
        "second_innings_wickets",
    ]
    missing = [col for col in required if col not in matches_df.columns]
    if missing:
        raise ValueError(f"matches.csv missing required columns: {', '.join(missing)}")

    frame = matches_df.copy()
    for col in [
        "match_id",
        "tournament",
        "team_a",
        "team_b",
        "venue",
        "winner",
        "first_innings_team",
        "second_innings_team",
    ]:
        frame[col] = frame[col].map(_clean_text)

    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    for col in ["first_innings_total", "first_innings_wickets", "second_innings_total", "second_innings_wickets"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame[
        frame["match_id"].ne("")
        & frame["team_a"].ne("")
        & frame["team_b"].ne("")
        & frame["venue"].ne("")
        & frame["tournament"].ne("")
        & frame["match_date"].notna()
        & frame["first_innings_total"].notna()
        & (
            frame["first_innings_team"].str.lower().eq(frame["team_a"].str.lower())
            | frame["first_innings_team"].str.lower().eq(frame["team_b"].str.lower())
        )
        & (
            frame["second_innings_team"].str.lower().eq(frame["team_a"].str.lower())
            | frame["second_innings_team"].str.lower().eq(frame["team_b"].str.lower())
        )
        & frame["team_a"].str.lower().ne(frame["team_b"].str.lower())
    ]
    frame = frame.sort_values("match_date").drop_duplicates(subset=["match_id"], keep="last")
    return frame.reset_index(drop=True)


def _normalize_players_input(players_df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "match_id",
        "match_date",
        "team",
        "opponent",
        "venue",
        "player",
        "runs",
        "balls",
        "wickets",
        "balls_bowled",
        "runs_conceded",
        "is_winner",
    ]
    missing = [col for col in required if col not in players_df.columns]
    if missing:
        raise ValueError(f"player_match_stats.csv missing required columns: {', '.join(missing)}")

    frame = players_df.copy()
    for col in ["match_id", "team", "opponent", "venue", "player"]:
        frame[col] = frame[col].map(_clean_text)
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    for col in ["runs", "balls", "wickets", "balls_bowled", "runs_conceded", "is_winner"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame[
        frame["match_id"].ne("")
        & frame["team"].ne("")
        & frame["opponent"].ne("")
        & frame["player"].ne("")
        & frame["match_date"].notna()
    ]
    for col in ["runs", "balls", "wickets", "balls_bowled", "runs_conceded", "is_winner"]:
        frame[col] = frame[col].fillna(0.0)
    return frame.sort_values("match_date").reset_index(drop=True)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_minimal_fallback_artifacts(processed_dir: Path, seed: int = 42) -> None:
    """
    Pure-Python fallback that does not depend on numpy/pandas.
    Used only when core ML dependencies are unavailable.
    """
    rng = random.Random(seed)
    teams = ["Mumbai Indians", "Royal Challengers Bengaluru", "Chennai Super Kings", "Kolkata Knight Riders"]
    venues = ["Wankhede Stadium", "M Chinnaswamy Stadium"]

    matches: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    players = {
        team: IPL_TEAM_ROSTERS.get(team, IPL_TEAM_ROSTERS["Mumbai Indians"])[:6]
        for team in teams
    }

    start = datetime(2024, 4, 1)
    for idx in range(12):
        team_a, team_b = rng.sample(teams, 2)
        venue = rng.choice(venues)
        match_date = (start + timedelta(days=idx)).date().isoformat()
        first_total = 165 + rng.randint(-20, 25)
        second_total = 160 + rng.randint(-20, 25)
        winner = team_a if first_total > second_total else team_b
        first_team = team_a
        second_team = team_b
        match_id = f"fallback_{idx:04d}"

        matches.append(
            {
                "match_id": match_id,
                "match_date": match_date,
                "tournament": "IPL",
                "team_a": team_a,
                "team_b": team_b,
                "venue": venue,
                "winner": winner,
                "toss_winner": team_a,
                "toss_decision": "bat",
                "first_innings_team": first_team,
                "first_innings_total": first_total,
                "first_innings_wickets": 6 + rng.randint(-2, 2),
                "second_innings_team": second_team,
                "second_innings_total": second_total,
                "second_innings_wickets": 7 + rng.randint(-2, 2),
            }
        )

        for team in [team_a, team_b]:
            for player in players[team]:
                player_rows.append(
                    {
                        "match_id": match_id,
                        "match_date": match_date,
                        "team": team,
                        "opponent": team_b if team == team_a else team_a,
                        "venue": venue,
                        "player": player,
                        "runs": float(max(0, rng.randint(8, 55))),
                        "balls": float(rng.randint(8, 40)),
                        "wickets": float(rng.choice([0, 0, 1, 2, 3])),
                        "balls_bowled": float(rng.choice([0, 12, 18, 24])),
                        "runs_conceded": float(rng.randint(0, 45)),
                        "is_winner": 1 if team == winner else 0,
                    }
                )

        feature_rows.append(
            {
                "match_id": match_id,
                "match_date": match_date,
                "tournament": "IPL",
                "team_a": team_a,
                "team_b": team_b,
                "venue": venue,
                "batting_team": first_team,
                "bowling_team": second_team,
                "target_score": float(first_total),
                "actual_first_innings_score": float(first_total),
                "target_team_a_score": float(first_total if first_team == team_a else second_total),
                "target_team_b_score": float(first_total if first_team == team_b else second_total),
                "target_winner_team_a": 1.0 if winner == team_a else 0.0,
                "team_a_avg_runs_last_5": float(first_total - 4),
                "team_a_avg_runs_last_10": float(first_total - 6),
                "team_a_avg_wickets_last_5": 6.8,
                "team_a_run_rate_trend": 0.22,
                "team_b_avg_runs_last_5": float(second_total - 3),
                "team_b_avg_runs_last_10": float(second_total - 5),
                "team_b_avg_wickets_last_5": 7.1,
                "team_b_run_rate_trend": -0.08,
                "team_a_win_rate_vs_b": 0.52,
                "avg_score_team_a_vs_b": 171.0,
                "avg_score_team_b_vs_a": 168.0,
                "venue_avg_score": 169.0,
                "venue_chase_success_rate": 0.47,
                "venue_defend_bias": 0.06,
                "team_a_runs_vs_opponent_avg": 3.0,
                "team_b_runs_vs_opponent_avg": -2.0,
                "batting_first": 1.0 if first_team == team_a else 0.0,
                "team_a_bats_first": 1.0 if first_team == team_a else 0.0,
                "team_b_bats_first": 0.0 if first_team == team_a else 1.0,
                "team_a_chase_success_rate": 0.52,
                "team_b_chase_success_rate": 0.48,
                "team_a_defend_success_rate": 0.51,
                "team_b_defend_success_rate": 0.49,
                "chase_defend_edge_team_a_first": -0.03,
                "chase_defend_edge_team_b_first": 0.03,
                "venue_batting_first_advantage": 0.06,
                "recent_form_diff": 4.0,
                "recent_run_rate_diff": 0.3,
                "head_to_head_win_diff": 0.04,
                "wickets_taken_diff": 0.2,
            }
        )

    team_profiles = []
    for team in teams:
        team_profiles.append(
            {
                "team": team,
                "avg_runs_scored": 171.0,
                "avg_wickets_lost": 6.9,
                "avg_runs_conceded": 169.0,
                "avg_wickets_taken": 6.8,
                "win_rate": 0.5,
                "batting_strength_index": 33.0,
                "bowling_strength_index": 26.0,
                "matches_played": 6,
            }
        )

    venue_profiles = [
        {"venue": venue, "avg_first_innings": 171.0, "avg_second_innings": 167.0, "matches_played": 6}
        for venue in venues
    ]

    player_form = []
    for team, roster in players.items():
        for player in roster:
            player_form.append(
                {
                    "team": team,
                    "player": player,
                    "matches_played": 6,
                    "avg_runs": 29.0,
                    "recent_runs": 31.0,
                    "total_runs": 174.0,
                    "total_balls": 120.0,
                    "avg_wickets": 1.1,
                    "recent_wickets": 1.0,
                    "total_runs_conceded": 110.0,
                    "total_balls_bowled": 72.0,
                    "win_rate": 0.5,
                    "last_match_date": "2024-04-30",
                    "strike_rate": 145.0,
                    "economy": 9.1,
                    "batting_form": 34.0,
                    "bowling_form": 22.0,
                    "impact_score": 29.0,
                }
            )

    _write_csv(
        processed_dir / "matches.csv",
        list(matches[0].keys()),
        matches,
    )
    _write_csv(
        processed_dir / "player_match_stats.csv",
        list(player_rows[0].keys()),
        player_rows,
    )
    _write_csv(
        processed_dir / "model_features.csv",
        list(feature_rows[0].keys()),
        feature_rows,
    )
    _write_csv(
        processed_dir / "match_feature_lookup.csv",
        list(feature_rows[0].keys()),
        feature_rows,
    )
    _write_csv(
        processed_dir / "team_profiles.csv",
        list(team_profiles[0].keys()),
        team_profiles,
    )
    _write_csv(
        processed_dir / "venue_profiles.csv",
        list(venue_profiles[0].keys()),
        venue_profiles,
    )
    _write_csv(
        processed_dir / "player_form_latest.csv",
        list(player_form[0].keys()),
        player_form,
    )

    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "feature_medians": {
            feature: float(median([row[feature] for row in feature_rows])) for feature in FEATURE_COLUMNS
        },
        "feature_means": {
            feature: float(mean([row[feature] for row in feature_rows])) for feature in FEATURE_COLUMNS
        },
        "feature_stds": {feature: 0.0 for feature in FEATURE_COLUMNS},
        "baseline_first_innings_mean": float(mean([row["target_score"] for row in feature_rows])),
        "baseline_first_innings_median": float(median([row["target_score"] for row in feature_rows])),
        "num_matches": len(matches),
        "num_rows": len(feature_rows),
        "num_teams": len(teams),
        "num_venues": len(venues),
        "num_players": len(player_form),
        "scenario_labels": ["Low", "Baseline", "High", "Aggressive"],
    }
    (processed_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def bootstrap_sample_datasets(processed_dir: Path, seed: int = 42, matches_count: int = 140) -> None:
    """Creates deterministic synthetic IPL-style data as fallback when raw ingestion is unavailable."""
    _require_core_dependencies()
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    teams = [
        "Mumbai Indians",
        "Chennai Super Kings",
        "Royal Challengers Bengaluru",
        "Kolkata Knight Riders",
        "Rajasthan Royals",
        "Sunrisers Hyderabad",
        "Delhi Capitals",
        "Punjab Kings",
        "Lucknow Super Giants",
        "Gujarat Titans",
    ]
    venues = [
        "Wankhede Stadium",
        "MA Chidambaram Stadium",
        "M Chinnaswamy Stadium",
        "Eden Gardens",
        "Narendra Modi Stadium",
        "Arun Jaitley Stadium",
    ]

    team_attack = {team: rng.uniform(-10, 12) for team in teams}
    team_defense = {team: rng.uniform(-8, 9) for team in teams}

    players_by_team = {
        team: IPL_TEAM_ROSTERS.get(team, IPL_TEAM_ROSTERS["Mumbai Indians"])[:11]
        for team in teams
    }

    base_date = datetime(2020, 9, 20)
    match_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []

    for index in range(matches_count):
        team_a, team_b = rng.sample(teams, 2)
        venue = rng.choice(venues)
        match_date = base_date + timedelta(days=index)

        toss_winner = rng.choice([team_a, team_b])
        toss_decision = rng.choice(["bat", "field"])

        if toss_decision == "bat":
            first_innings_team = toss_winner
        else:
            first_innings_team = team_b if toss_winner == team_a else team_a

        second_innings_team = team_b if first_innings_team == team_a else team_a

        venue_bias = {
            "Wankhede Stadium": 6,
            "MA Chidambaram Stadium": -8,
            "M Chinnaswamy Stadium": 10,
            "Eden Gardens": 4,
            "Narendra Modi Stadium": 3,
            "Arun Jaitley Stadium": 1,
        }[venue]

        first_mu = (
            170
            + team_attack[first_innings_team]
            - team_defense[second_innings_team]
            + venue_bias
            + np_rng.normal(0, 4)
        )
        first_total = int(np.clip(np_rng.normal(first_mu, 17), 120, 240))
        first_wickets = int(np.clip(np_rng.normal(6.8, 1.8), 2, 10))

        chase_noise = np_rng.normal(-2, 15)
        second_total = int(np.clip(first_total + chase_noise, 110, 245))
        second_wickets = int(np.clip(np_rng.normal(6.9, 1.9), 2, 10))

        winner = second_innings_team if second_total > first_total else first_innings_team

        match_id = f"sample_{match_date:%Y%m%d}_{index:04d}"
        match_rows.append(
            {
                "match_id": match_id,
                "match_date": match_date.date().isoformat(),
                "tournament": "IPL",
                "team_a": team_a,
                "team_b": team_b,
                "venue": venue,
                "winner": winner,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "first_innings_team": first_innings_team,
                "first_innings_total": first_total,
                "first_innings_wickets": first_wickets,
                "second_innings_team": second_innings_team,
                "second_innings_total": second_total,
                "second_innings_wickets": second_wickets,
            }
        )

        for team in [team_a, team_b]:
            opponent = team_b if team == team_a else team_a
            batting_total = first_total if first_innings_team == team else second_total
            wickets_taken = second_wickets if first_innings_team == team else first_wickets
            team_players = players_by_team[team]

            batting_weights = np_rng.dirichlet(np.ones(6))
            bowling_weights = np_rng.dirichlet(np.ones(5))

            for idx, player in enumerate(team_players[:6]):
                player_runs = float(max(0.0, batting_total * batting_weights[idx] + np_rng.normal(0, 4)))
                balls = float(max(1.0, player_runs / rng.uniform(0.9, 1.8)))
                wickets = 0.0
                balls_bowled = 0.0
                runs_conceded = 0.0
                if idx < 5:
                    wickets = float(max(0.0, wickets_taken * bowling_weights[idx] + np_rng.normal(0, 0.3)))
                    balls_bowled = float(max(0.0, 24 + np_rng.normal(0, 6)))
                    runs_conceded = float(max(0.0, (batting_total / 5) + np_rng.normal(0, 10)))

                player_rows.append(
                    {
                        "match_id": match_id,
                        "match_date": match_date.date().isoformat(),
                        "team": team,
                        "opponent": opponent,
                        "venue": venue,
                        "player": player,
                        "runs": round(player_runs, 2),
                        "balls": round(balls, 2),
                        "wickets": round(wickets, 2),
                        "balls_bowled": round(balls_bowled, 2),
                        "runs_conceded": round(runs_conceded, 2),
                        "is_winner": 1 if winner == team else 0,
                    }
                )

    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(match_rows).to_csv(processed_dir / "matches.csv", index=False)
    pd.DataFrame(player_rows).to_csv(processed_dir / "player_match_stats.csv", index=False)


def _team_history_matches(matches_df: pd.DataFrame, team: str, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    hist = matches_df[(matches_df["match_date"] < cutoff_date)]
    return hist[(hist["team_a"] == team) | (hist["team_b"] == team)]


def _team_recent_metrics(
    matches_df: pd.DataFrame,
    team: str,
    cutoff_date: pd.Timestamp,
    window: int,
    defaults: TeamMetrics,
) -> TeamMetrics:
    hist = _team_history_matches(matches_df, team, cutoff_date).tail(window)
    if hist.empty:
        return defaults

    runs_scored: list[float] = []
    wickets_lost: list[float] = []
    runs_conceded: list[float] = []
    wickets_taken: list[float] = []
    wins: list[int] = []

    for _, row in hist.iterrows():
        if row["first_innings_team"] == team:
            runs_scored.append(float(row["first_innings_total"]))
            wickets_lost.append(float(row["first_innings_wickets"]))
            if not pd.isna(row.get("second_innings_total")):
                runs_conceded.append(float(row["second_innings_total"]))
            if not pd.isna(row.get("second_innings_wickets")):
                wickets_taken.append(float(row["second_innings_wickets"]))
        else:
            if not pd.isna(row.get("second_innings_total")):
                runs_scored.append(float(row["second_innings_total"]))
            if not pd.isna(row.get("second_innings_wickets")):
                wickets_lost.append(float(row["second_innings_wickets"]))
            runs_conceded.append(float(row["first_innings_total"]))
            wickets_taken.append(float(row["first_innings_wickets"]))

        wins.append(1 if row.get("winner") == team else 0)

    return TeamMetrics(
        avg_runs_scored=float(np.mean(runs_scored)) if runs_scored else defaults.avg_runs_scored,
        avg_wickets_lost=float(np.mean(wickets_lost)) if wickets_lost else defaults.avg_wickets_lost,
        avg_runs_conceded=float(np.mean(runs_conceded)) if runs_conceded else defaults.avg_runs_conceded,
        avg_wickets_taken=float(np.mean(wickets_taken)) if wickets_taken else defaults.avg_wickets_taken,
        win_rate=float(np.mean(wins)) if wins else defaults.win_rate,
    )


def _team_chase_defend_rates(
    matches_df: pd.DataFrame,
    team: str,
    cutoff_date: pd.Timestamp,
) -> tuple[float, float]:
    hist = _team_history_matches(matches_df, team, cutoff_date)
    if hist.empty:
        return 0.5, 0.5

    chase_attempts = 0
    chase_wins = 0
    defend_attempts = 0
    defend_wins = 0
    for _, row in hist.iterrows():
        first_team = str(row.get("first_innings_team", ""))
        second_team = str(row.get("second_innings_team", ""))
        winner = str(row.get("winner", ""))
        if second_team == team:
            chase_attempts += 1
            if winner == team:
                chase_wins += 1
        elif first_team == team:
            defend_attempts += 1
            if winner == team:
                defend_wins += 1
    chase_rate = float(chase_wins / chase_attempts) if chase_attempts else 0.5
    defend_rate = float(defend_wins / defend_attempts) if defend_attempts else 0.5
    return float(np.clip(chase_rate, 0.0, 1.0)), float(np.clip(defend_rate, 0.0, 1.0))


def _head_to_head_stats(
    matches_df: pd.DataFrame,
    team_a: str,
    team_b: str,
    cutoff_date: pd.Timestamp,
    default_score: float,
) -> tuple[float, float, float]:
    hist = matches_df[matches_df["match_date"] < cutoff_date]
    subset = hist[
        ((hist["team_a"] == team_a) & (hist["team_b"] == team_b))
        | ((hist["team_a"] == team_b) & (hist["team_b"] == team_a))
    ]
    if subset.empty:
        return 0.5, default_score, default_score

    wins_a = 0
    score_a: list[float] = []
    score_b: list[float] = []
    for _, match_row in subset.iterrows():
        winner = str(match_row.get("winner", ""))
        if winner == team_a:
            wins_a += 1
        team_a_score = _team_score_from_row(match_row, team_a)
        team_b_score = _team_score_from_row(match_row, team_b)
        if team_a_score is not None:
            score_a.append(team_a_score)
        if team_b_score is not None:
            score_b.append(team_b_score)

    avg_a = float(np.mean(score_a)) if score_a else default_score
    avg_b = float(np.mean(score_b)) if score_b else default_score
    return float(wins_a / len(subset)), avg_a, avg_b


def _venue_context_metrics(
    matches_df: pd.DataFrame,
    venue: str,
    cutoff_date: pd.Timestamp,
    default_score: float,
) -> tuple[float, float, float]:
    hist = matches_df[(matches_df["match_date"] < cutoff_date) & (matches_df["venue"] == venue)]
    if hist.empty:
        return default_score, 0.5, 0.0

    first = pd.to_numeric(hist["first_innings_total"], errors="coerce")
    second = pd.to_numeric(hist["second_innings_total"], errors="coerce")
    innings_values = pd.concat([first, second], axis=0).dropna()
    venue_avg_score = float(innings_values.mean()) if not innings_values.empty else default_score

    chase_success = float(np.mean(second > first)) if len(hist) else 0.5
    batting_first_advantage = float(np.mean(first > second) - 0.5) * 2.0
    return venue_avg_score, float(np.clip(chase_success, 0.0, 1.0)), float(
        np.clip(batting_first_advantage, -1.0, 1.0)
    )


def _team_runs_vs_opponent_avg(
    team_metrics: TeamMetrics,
    opponent_metrics: TeamMetrics,
) -> float:
    return float(team_metrics.avg_runs_scored - opponent_metrics.avg_runs_conceded)


def _player_strength_indices(
    players_df: pd.DataFrame,
    team: str,
    cutoff_date: pd.Timestamp,
    window: int,
    default_batting: float,
    default_bowling: float,
) -> tuple[float, float]:
    team_hist = players_df[(players_df["match_date"] < cutoff_date) & (players_df["team"] == team)]
    if team_hist.empty:
        return default_batting, default_bowling

    recent = team_hist.sort_values("match_date").groupby("player", sort=False).tail(window)
    agg = recent.groupby("player", as_index=False).agg(
        avg_runs=("runs", "mean"),
        avg_wickets=("wickets", "mean"),
        total_runs_conceded=("runs_conceded", "sum"),
        total_balls_bowled=("balls_bowled", "sum"),
    )
    if agg.empty:
        return default_batting, default_bowling

    agg["economy"] = np.where(
        agg["total_balls_bowled"] > 0,
        agg["total_runs_conceded"] / (agg["total_balls_bowled"] / 6.0),
        np.nan,
    )

    batting_strength = float(agg["avg_runs"].nlargest(min(3, len(agg))).mean())

    top_wkts = agg["avg_wickets"].nlargest(min(3, len(agg))).mean()
    best_economy = agg["economy"].nsmallest(min(3, len(agg.dropna(subset=["economy"])))).mean()
    if np.isnan(best_economy):
        best_economy = 8.0

    bowling_strength = float((top_wkts * 20.0) + max(0.0, (8.5 - float(best_economy)) * 4.0))
    return batting_strength if not np.isnan(batting_strength) else default_batting, bowling_strength


def build_feature_table(
    matches_df: pd.DataFrame,
    players_df: pd.DataFrame,
    history_window: int,
) -> pd.DataFrame:
    matches_df = _normalize_matches_input(matches_df)
    players_df = _normalize_players_input(players_df)
    if matches_df.empty:
        raise ValueError("No valid match rows available after normalization.")

    global_first = float(matches_df["first_innings_total"].astype(float).mean())
    global_second = float(matches_df["second_innings_total"].astype(float).mean())
    global_score = float(np.nanmean([global_first, global_second]))
    defaults = TeamMetrics(
        avg_runs_scored=global_score,
        avg_wickets_lost=7.0,
        avg_runs_conceded=global_score,
        avg_wickets_taken=7.0,
        win_rate=0.5,
    )

    feature_rows: list[dict[str, Any]] = []

    for _, row in matches_df.iterrows():
        team_a = str(row.get("team_a"))
        team_b = str(row.get("team_b"))
        batting_team = row.get("first_innings_team")
        if batting_team not in {team_a, team_b}:
            continue
        bowling_team = team_b if batting_team == team_a else team_a
        cutoff_date = row["match_date"]

        team_a_last5 = _team_recent_metrics(matches_df, team_a, cutoff_date, 5, defaults)
        team_a_last10 = _team_recent_metrics(matches_df, team_a, cutoff_date, 10, defaults)
        team_b_last5 = _team_recent_metrics(matches_df, team_b, cutoff_date, 5, defaults)
        team_b_last10 = _team_recent_metrics(matches_df, team_b, cutoff_date, 10, defaults)

        team_a_win_rate_vs_b, avg_score_a_vs_b, avg_score_b_vs_a = _head_to_head_stats(
            matches_df,
            team_a=team_a,
            team_b=team_b,
            cutoff_date=cutoff_date,
            default_score=global_score,
        )
        team_a_chase_rate, team_a_defend_rate = _team_chase_defend_rates(matches_df, team_a, cutoff_date)
        team_b_chase_rate, team_b_defend_rate = _team_chase_defend_rates(matches_df, team_b, cutoff_date)
        venue_avg_score, venue_chase_success_rate, venue_batting_first_advantage = _venue_context_metrics(
            matches_df,
            venue=str(row.get("venue")),
            cutoff_date=cutoff_date,
            default_score=global_score,
        )

        team_a_score = _team_score_from_row(row, team_a)
        team_b_score = _team_score_from_row(row, team_b)
        if team_a_score is None or team_b_score is None:
            continue
        winner = _clean_text(row.get("winner"))
        if not winner:
            if team_a_score > team_b_score:
                winner = team_a
            elif team_b_score > team_a_score:
                winner = team_b

        feature_row = {
            "match_id": row["match_id"],
            "match_date": row["match_date"].date().isoformat(),
            "tournament": row.get("tournament", "IPL"),
            "team_a": team_a,
            "team_b": team_b,
            "venue": row["venue"],
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "target_score": float(row["first_innings_total"]),
            "actual_first_innings_score": float(row["first_innings_total"]),
            "target_team_a_score": float(team_a_score),
            "target_team_b_score": float(team_b_score),
            "target_winner_team_a": 1.0 if winner == team_a else 0.0,
            "team_a_avg_runs_last_5": team_a_last5.avg_runs_scored,
            "team_a_avg_runs_last_10": team_a_last10.avg_runs_scored,
            "team_a_avg_wickets_last_5": team_a_last5.avg_wickets_lost,
            "team_a_run_rate_trend": team_a_last5.avg_runs_scored - team_a_last10.avg_runs_scored,
            "team_b_avg_runs_last_5": team_b_last5.avg_runs_scored,
            "team_b_avg_runs_last_10": team_b_last10.avg_runs_scored,
            "team_b_avg_wickets_last_5": team_b_last5.avg_wickets_lost,
            "team_b_run_rate_trend": team_b_last5.avg_runs_scored - team_b_last10.avg_runs_scored,
            "team_a_win_rate_vs_b": team_a_win_rate_vs_b,
            "avg_score_team_a_vs_b": avg_score_a_vs_b,
            "avg_score_team_b_vs_a": avg_score_b_vs_a,
            "venue_avg_score": venue_avg_score,
            "venue_chase_success_rate": venue_chase_success_rate,
            "venue_defend_bias": -venue_batting_first_advantage,
            "team_a_runs_vs_opponent_avg": _team_runs_vs_opponent_avg(team_a_last5, team_b_last5),
            "team_b_runs_vs_opponent_avg": _team_runs_vs_opponent_avg(team_b_last5, team_a_last5),
            "batting_first": 1.0 if batting_team == team_a else 0.0,
            "team_a_bats_first": 1.0 if batting_team == team_a else 0.0,
            "team_b_bats_first": 0.0 if batting_team == team_a else 1.0,
            "team_a_chase_success_rate": team_a_chase_rate,
            "team_b_chase_success_rate": team_b_chase_rate,
            "team_a_defend_success_rate": team_a_defend_rate,
            "team_b_defend_success_rate": team_b_defend_rate,
            "chase_defend_edge_team_a_first": team_b_chase_rate - team_a_defend_rate,
            "chase_defend_edge_team_b_first": team_a_chase_rate - team_b_defend_rate,
            "venue_batting_first_advantage": venue_batting_first_advantage,
            "recent_form_diff": team_a_last5.avg_runs_scored - team_b_last5.avg_runs_scored,
            "recent_run_rate_diff": (team_a_last5.avg_runs_scored - team_a_last10.avg_runs_scored)
            - (team_b_last5.avg_runs_scored - team_b_last10.avg_runs_scored),
            "head_to_head_win_diff": (2.0 * team_a_win_rate_vs_b) - 1.0,
            "wickets_taken_diff": team_a_last5.avg_wickets_taken - team_b_last5.avg_wickets_taken,
        }
        feature_rows.append(feature_row)

    output = pd.DataFrame(feature_rows)
    if output.empty:
        raise ValueError("No feature rows could be generated from normalized matches.")
    for col in FEATURE_COLUMNS + ["target_team_a_score", "target_team_b_score", "target_winner_team_a"]:
        output[col] = pd.to_numeric(output[col], errors="coerce")
    output = output.dropna(subset=FEATURE_COLUMNS + ["target_team_a_score", "target_team_b_score"])
    return output.reset_index(drop=True)


def build_team_profiles(
    matches_df: pd.DataFrame,
    players_df: pd.DataFrame,
    history_window: int,
) -> pd.DataFrame:
    matches_df = _normalize_matches_input(matches_df)
    players_df = _normalize_players_input(players_df)
    if matches_df.empty:
        return pd.DataFrame(
            columns=[
                "team",
                "avg_runs_scored",
                "avg_wickets_lost",
                "avg_runs_conceded",
                "avg_wickets_taken",
                "win_rate",
                "batting_strength_index",
                "bowling_strength_index",
                "matches_played",
            ]
        )

    latest_date = matches_df["match_date"].max() + pd.Timedelta(days=1)
    global_default = TeamMetrics(
        avg_runs_scored=float(matches_df["first_innings_total"].astype(float).mean()),
        avg_wickets_lost=7.0,
        avg_runs_conceded=float(matches_df["first_innings_total"].astype(float).mean()),
        avg_wickets_taken=7.0,
        win_rate=0.5,
    )

    rows = []
    all_teams = sorted(set(matches_df["team_a"]).union(set(matches_df["team_b"])))
    for team in all_teams:
        metrics = _team_recent_metrics(matches_df, team, latest_date, history_window * 6, global_default)
        batting_strength, bowling_strength = _player_strength_indices(
            players_df,
            team=team,
            cutoff_date=latest_date,
            window=history_window * 3,
            default_batting=32.0,
            default_bowling=25.0,
        )
        matches_played = len(_team_history_matches(matches_df, team, latest_date))
        rows.append(
            {
                "team": team,
                "avg_runs_scored": metrics.avg_runs_scored,
                "avg_wickets_lost": metrics.avg_wickets_lost,
                "avg_runs_conceded": metrics.avg_runs_conceded,
                "avg_wickets_taken": metrics.avg_wickets_taken,
                "win_rate": metrics.win_rate,
                "batting_strength_index": batting_strength,
                "bowling_strength_index": bowling_strength,
                "matches_played": matches_played,
            }
        )

    return pd.DataFrame(rows)


def build_venue_profiles(matches_df: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_matches_input(matches_df)
    if frame.empty:
        return pd.DataFrame(columns=["venue", "avg_first_innings", "avg_second_innings", "matches_played"])
    grouped = frame.groupby("venue", as_index=False).agg(
        avg_first_innings=("first_innings_total", "mean"),
        avg_second_innings=("second_innings_total", "mean"),
        matches_played=("match_id", "count"),
    )
    return grouped.sort_values("matches_played", ascending=False)


def build_player_form(players_df: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_players_input(players_df)
    if frame.empty:
        return pd.DataFrame(columns=["team", "player", "impact_score"])

    grouped = frame.groupby(["team", "player"], as_index=False).agg(
        matches_played=("match_id", "nunique"),
        avg_runs=("runs", "mean"),
        recent_runs=("runs", lambda s: float(s.tail(5).mean())),
        total_runs=("runs", "sum"),
        total_balls=("balls", "sum"),
        avg_wickets=("wickets", "mean"),
        recent_wickets=("wickets", lambda s: float(s.tail(5).mean())),
        total_runs_conceded=("runs_conceded", "sum"),
        total_balls_bowled=("balls_bowled", "sum"),
        win_rate=("is_winner", "mean"),
        last_match_date=("match_date", "max"),
    )

    grouped["strike_rate"] = np.where(
        grouped["total_balls"] > 0,
        (grouped["total_runs"] / grouped["total_balls"]) * 100.0,
        110.0,
    )
    grouped["economy"] = np.where(
        grouped["total_balls_bowled"] > 0,
        grouped["total_runs_conceded"] / (grouped["total_balls_bowled"] / 6.0),
        8.2,
    )

    grouped["batting_form"] = (
        grouped["recent_runs"] * 0.55
        + grouped["avg_runs"] * 0.30
        + (grouped["strike_rate"] / 12.0) * 0.15
    )
    grouped["bowling_form"] = (
        grouped["recent_wickets"] * 15.0
        + grouped["avg_wickets"] * 10.0
        + np.maximum(0.0, 8.5 - grouped["economy"]) * 4.0
    )
    grouped["impact_score"] = grouped["batting_form"] * 0.6 + grouped["bowling_form"] * 0.4

    return grouped.sort_values(["team", "impact_score"], ascending=[True, False])


def build_manifest(
    feature_table: pd.DataFrame,
    team_profiles: pd.DataFrame,
    venue_profiles: pd.DataFrame,
    player_form: pd.DataFrame,
) -> dict[str, Any]:
    for feature in FEATURE_COLUMNS:
        if feature not in feature_table.columns:
            feature_table[feature] = np.nan

    medians = {feature: float(feature_table[feature].median()) for feature in FEATURE_COLUMNS}
    means = {feature: float(feature_table[feature].mean()) for feature in FEATURE_COLUMNS}
    stds = {feature: float(feature_table[feature].std()) for feature in FEATURE_COLUMNS}

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "feature_medians": medians,
        "feature_means": means,
        "feature_stds": stds,
        "baseline_first_innings_mean": float(feature_table["target_score"].mean()),
        "baseline_first_innings_median": float(feature_table["target_score"].median()),
        "num_matches": int(feature_table["match_id"].nunique()),
        "num_rows": int(len(feature_table)),
        "num_teams": int(team_profiles["team"].nunique()),
        "num_venues": int(venue_profiles["venue"].nunique()),
        "num_players": int(player_form[["team", "player"]].drop_duplicates().shape[0]),
        "scenario_labels": ["Low", "Baseline", "High", "Aggressive"],
    }


def run_feature_pipeline(config: MLConfig, bootstrap_if_missing: bool = False) -> None:
    _require_core_dependencies()
    ensure_directories(config)
    matches_path = config.processed_data_dir / "matches.csv"
    players_path = config.processed_data_dir / "player_match_stats.csv"

    if bootstrap_if_missing and (not matches_path.exists() or not players_path.exists()):
        print("No ingested dataset found. Generating deterministic sample dataset.")
        bootstrap_sample_datasets(config.processed_data_dir, seed=config.random_seed)

    if not matches_path.exists() or not players_path.exists():
        raise FileNotFoundError(
            "Required files missing: matches.csv and player_match_stats.csv under data/processed"
        )

    matches_df = pd.read_csv(matches_path)
    players_df = pd.read_csv(players_path)

    feature_table = build_feature_table(
        matches_df=matches_df,
        players_df=players_df,
        history_window=config.history_window,
    )
    team_profiles = build_team_profiles(
        matches_df=matches_df,
        players_df=players_df,
        history_window=config.history_window,
    )
    venue_profiles = build_venue_profiles(matches_df=matches_df)
    player_form = build_player_form(players_df=players_df)
    manifest = build_manifest(feature_table, team_profiles, venue_profiles, player_form)

    feature_table.to_csv(config.processed_data_dir / "model_features.csv", index=False)
    feature_table.to_csv(config.processed_data_dir / "match_feature_lookup.csv", index=False)
    team_profiles.to_csv(config.processed_data_dir / "team_profiles.csv", index=False)
    venue_profiles.to_csv(config.processed_data_dir / "venue_profiles.csv", index=False)
    player_form.to_csv(config.processed_data_dir / "player_form_latest.csv", index=False)
    (config.processed_data_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(
        "Feature pipeline complete: "
        f"{len(feature_table)} rows, {len(team_profiles)} team profiles, "
        f"{len(player_form)} player profiles"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IPL feature store for TimeMCL-inspired model")
    parser.add_argument(
        "--force-bootstrap",
        action="store_true",
        help="Ignore existing processed files and generate synthetic sample inputs",
    )
    args = parser.parse_args()

    config = MLConfig()
    ensure_directories(config)

    if not _core_dependencies_ready():
        print("Missing numpy/pandas in current environment.")
        print("Running in fallback mode (no trained model available).")
        if args.force_bootstrap:
            write_minimal_fallback_artifacts(config.processed_data_dir, seed=config.random_seed)
            print(
                "Fallback bootstrap artifacts generated without numpy/pandas. "
                "Install full deps via scripts/setup_ml.sh for complete feature engineering."
            )
            return
        raise SystemExit(
            "Install ML dependencies first: ./scripts/setup_ml.sh "
            "or activate ml/.venv and install ml/requirements.txt"
        )

    if args.force_bootstrap:
        bootstrap_sample_datasets(config.processed_data_dir, seed=config.random_seed)

    run_feature_pipeline(config=config, bootstrap_if_missing=False)


if __name__ == "__main__":
    main()
