from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ml.config import MLConfig, ensure_directories

RUN_OUT_LIKE_DISMISSALS = {"run out", "retired hurt", "retired out", "obstructing the field"}


def _clean_text(value: Any, default: str = "") -> str:
    text = " ".join(str(value or "").strip().split())
    return text or default


def download_cricsheet_zip(url: str, destination: Path, timeout_seconds: int = 90) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    destination.write_bytes(response.content)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_match_payload(payload: dict[str, Any], match_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    info = payload.get("info", {})
    teams = info.get("teams", [])
    if len(teams) < 2:
        return None, []

    team_a, team_b = _clean_text(teams[0]), _clean_text(teams[1])
    if not team_a or not team_b or team_a.lower() == team_b.lower():
        return None, []
    tournament = _clean_text(
        (
        info.get("event", {}).get("name")
        or info.get("competition")
        or "Indian Premier League"
        )
    )
    tournament_code = "IPL" if "premier league" in tournament.lower() or "ipl" in tournament.lower() else tournament
    if tournament_code != "IPL":
        return None, []

    match_dates = info.get("dates", [])
    match_date = match_dates[0] if match_dates else None

    innings = payload.get("innings", [])
    if not innings:
        return None, []

    toss = info.get("toss", {})
    toss_winner = _clean_text(toss.get("winner"))
    toss_decision = _clean_text(toss.get("decision"))

    outcome = info.get("outcome", {})
    winner = _clean_text(outcome.get("winner"))

    team_player_stats: dict[str, dict[str, dict[str, float]]] = {
        team_a: defaultdict(lambda: {"runs": 0.0, "balls": 0.0, "wickets": 0.0, "balls_bowled": 0.0, "runs_conceded": 0.0}),
        team_b: defaultdict(lambda: {"runs": 0.0, "balls": 0.0, "wickets": 0.0, "balls_bowled": 0.0, "runs_conceded": 0.0}),
    }

    innings_rows: list[dict[str, Any]] = []

    for innings_index, innings_blob in enumerate(innings[:2]):
        batting_team = _clean_text(innings_blob.get("team"))
        if batting_team not in {team_a, team_b}:
            continue
        bowling_team = team_b if batting_team == team_a else team_a

        innings_total = 0
        innings_wickets = 0

        for over in innings_blob.get("overs", []):
            for delivery in over.get("deliveries", []):
                runs_info = delivery.get("runs", {})
                total_run = _safe_int(runs_info.get("total"))
                batter_run = _safe_int(runs_info.get("batter"))
                innings_total += total_run

                batter_name = _clean_text(delivery.get("batter"))
                if batter_name:
                    batter_stats = team_player_stats[batting_team][batter_name]
                    batter_stats["runs"] += batter_run
                    batter_stats["balls"] += 1

                bowler_name = _clean_text(delivery.get("bowler"))
                if bowler_name:
                    bowler_stats = team_player_stats[bowling_team][bowler_name]
                    bowler_stats["balls_bowled"] += 1
                    bowler_stats["runs_conceded"] += total_run

                wicket_events = delivery.get("wickets", [])
                if wicket_events:
                    for wicket_event in wicket_events:
                        dismissal_kind = str(wicket_event.get("kind", "")).lower()
                        if dismissal_kind != "retired hurt":
                            innings_wickets += 1
                        if bowler_name and dismissal_kind not in RUN_OUT_LIKE_DISMISSALS:
                            team_player_stats[bowling_team][bowler_name]["wickets"] += 1

        innings_rows.append(
            {
                "innings_index": innings_index + 1,
                "team": batting_team,
                "total": innings_total,
                "wickets": innings_wickets,
            }
        )

    if not innings_rows:
        return None, []

    first = innings_rows[0]
    second = innings_rows[1] if len(innings_rows) > 1 else {"team": None, "total": None, "wickets": None}

    match_row: dict[str, Any] = {
        "match_id": _clean_text(match_id),
        "match_date": match_date,
        "tournament": "IPL",
        "team_a": team_a,
        "team_b": team_b,
        "venue": _clean_text(info.get("venue"), "Unknown Venue"),
        "winner": winner,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "first_innings_team": first["team"],
        "first_innings_total": first["total"],
        "first_innings_wickets": first["wickets"],
        "second_innings_team": second["team"],
        "second_innings_total": second["total"],
        "second_innings_wickets": second["wickets"],
    }

    player_rows: list[dict[str, Any]] = []
    for team_name in [team_a, team_b]:
        opponent = team_b if team_name == team_a else team_a
        is_winner = 1 if winner and winner == team_name else 0
        for player_name, stats in team_player_stats[team_name].items():
            player_rows.append(
                {
                    "match_id": match_id,
                    "match_date": match_date,
                    "team": team_name,
                    "opponent": opponent,
                    "venue": _clean_text(info.get("venue"), "Unknown Venue"),
                    "player": player_name,
                    "runs": round(stats["runs"], 2),
                    "balls": round(stats["balls"], 2),
                    "wickets": round(stats["wickets"], 2),
                    "balls_bowled": round(stats["balls_bowled"], 2),
                    "runs_conceded": round(stats["runs_conceded"], 2),
                    "is_winner": is_winner,
                }
            )

    return match_row, player_rows


def ingest_cricsheet_json_folder(json_dir: Path, max_files: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(json_dir.glob("*.json"))
    if max_files is not None:
        files = files[:max_files]

    match_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []

    for file_path in files:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            match_row, match_player_rows = parse_match_payload(payload, match_id=file_path.stem)
            if match_row is None:
                continue
            match_rows.append(match_row)
            player_rows.extend(match_player_rows)
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {file_path.name}: {exc}")

    matches_df = pd.DataFrame(match_rows)
    players_df = pd.DataFrame(player_rows)

    if not matches_df.empty:
        matches_df["match_date"] = pd.to_datetime(matches_df["match_date"], errors="coerce")
        matches_df = matches_df.sort_values("match_date").reset_index(drop=True)

    if not players_df.empty:
        players_df["match_date"] = pd.to_datetime(players_df["match_date"], errors="coerce")
        players_df = players_df.sort_values("match_date").reset_index(drop=True)

    return matches_df, players_df


def save_datasets(matches_df: pd.DataFrame, players_df: pd.DataFrame, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(processed_dir / "matches.csv", index=False)
    players_df.to_csv(processed_dir / "player_match_stats.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and parse IPL datasets from Cricsheet")
    parser.add_argument("--url", default=MLConfig().cricsheet_ipl_json_url)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--json-dir", type=Path, default=None)
    args = parser.parse_args()

    config = MLConfig()
    ensure_directories(config)

    zip_path = config.raw_data_dir / "ipl_json.zip"
    extracted_dir = config.raw_data_dir / "cricsheet_ipl_json"

    if not args.skip_download:
        print(f"Downloading IPL JSON archive from {args.url}")
        download_cricsheet_zip(args.url, zip_path)
        print(f"Extracting {zip_path.name} -> {extracted_dir}")
        extract_zip(zip_path, extracted_dir)

    json_dir = args.json_dir or extracted_dir
    if not json_dir.exists():
        raise FileNotFoundError(
            "No JSON directory found. Run without --skip-download or provide --json-dir."
        )

    matches_df, players_df = ingest_cricsheet_json_folder(json_dir=json_dir, max_files=args.max_files)
    if matches_df.empty:
        raise RuntimeError("Ingestion produced zero IPL matches. Check source dataset format.")

    save_datasets(matches_df=matches_df, players_df=players_df, processed_dir=config.processed_data_dir)
    print(
        f"Wrote {len(matches_df)} matches and {len(players_df)} player records to {config.processed_data_dir}"
    )


if __name__ == "__main__":
    main()
