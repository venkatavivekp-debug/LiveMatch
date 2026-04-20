from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ml.config import MLConfig, ensure_directories

FOOTBALL_TEAM_PLAYERS: dict[str, list[str]] = {
    "Arsenal": [
        "Bukayo Saka",
        "Martin Odegaard",
        "Kai Havertz",
        "Declan Rice",
        "Gabriel Martinelli",
        "William Saliba",
        "Ben White",
    ],
    "Manchester City": [
        "Erling Haaland",
        "Phil Foden",
        "Kevin De Bruyne",
        "Rodri",
        "Bernardo Silva",
        "Ruben Dias",
        "Kyle Walker",
    ],
    "Liverpool": [
        "Mohamed Salah",
        "Darwin Nunez",
        "Luis Diaz",
        "Alexis Mac Allister",
        "Dominik Szoboszlai",
        "Virgil van Dijk",
        "Trent Alexander-Arnold",
    ],
    "Tottenham Hotspur": [
        "Son Heung-min",
        "James Maddison",
        "Dejan Kulusevski",
        "Richarlison",
        "Yves Bissouma",
        "Cristian Romero",
        "Pedro Porro",
    ],
    "Chelsea": [
        "Cole Palmer",
        "Nicolas Jackson",
        "Raheem Sterling",
        "Enzo Fernandez",
        "Moises Caicedo",
        "Reece James",
        "Levi Colwill",
    ],
    "Manchester United": [
        "Bruno Fernandes",
        "Marcus Rashford",
        "Rasmus Hojlund",
        "Alejandro Garnacho",
        "Casemiro",
        "Lisandro Martinez",
        "Luke Shaw",
    ],
    "Newcastle United": [
        "Alexander Isak",
        "Anthony Gordon",
        "Bruno Guimaraes",
        "Harvey Barnes",
        "Joelinton",
        "Kieran Trippier",
        "Sven Botman",
    ],
    "Aston Villa": [
        "Ollie Watkins",
        "Leon Bailey",
        "Moussa Diaby",
        "Douglas Luiz",
        "John McGinn",
        "Pau Torres",
        "Ezri Konsa",
    ],
    "Barcelona": [
        "Robert Lewandowski",
        "Lamine Yamal",
        "Raphinha",
        "Pedri",
        "Frenkie de Jong",
        "Jules Kounde",
        "Ronald Araujo",
    ],
    "Real Madrid": [
        "Vinicius Junior",
        "Rodrygo",
        "Jude Bellingham",
        "Federico Valverde",
        "Aurelien Tchouameni",
        "Antonio Rudiger",
        "Dani Carvajal",
    ],
    "Atletico Madrid": [
        "Antoine Griezmann",
        "Alvaro Morata",
        "Angel Correa",
        "Rodrigo De Paul",
        "Koke",
        "Jose Maria Gimenez",
        "Jan Oblak",
    ],
    "Bayern Munich": [
        "Harry Kane",
        "Jamal Musiala",
        "Leroy Sane",
        "Thomas Muller",
        "Joshua Kimmich",
        "Matthijs de Ligt",
        "Alphonso Davies",
    ],
}


def bootstrap_football_datasets(config: MLConfig, matches_count: int = 64, seed: int = 42) -> None:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    teams = [
        "Arsenal",
        "Manchester City",
        "Liverpool",
        "Tottenham Hotspur",
        "Chelsea",
        "Manchester United",
        "Newcastle United",
        "Aston Villa",
        "Barcelona",
        "Real Madrid",
        "Atletico Madrid",
        "Bayern Munich",
    ]
    tournaments = ["EPL", "UCL", "LALIGA"]
    venues = {
        "Arsenal": "Emirates Stadium",
        "Manchester City": "Etihad Stadium",
        "Liverpool": "Anfield",
        "Tottenham Hotspur": "Tottenham Hotspur Stadium",
        "Chelsea": "Stamford Bridge",
        "Manchester United": "Old Trafford",
        "Newcastle United": "St James' Park",
        "Aston Villa": "Villa Park",
        "Barcelona": "Estadi Olimpic Lluis Companys",
        "Real Madrid": "Santiago Bernabeu",
        "Atletico Madrid": "Metropolitano Stadium",
        "Bayern Munich": "Allianz Arena",
    }

    attack = {team: rng.uniform(1.2, 2.0) for team in teams}
    defense = {team: rng.uniform(0.75, 1.45) for team in teams}
    form = {team: rng.uniform(6.0, 12.0) for team in teams}

    base_date = datetime(2025, 8, 1)
    match_rows: list[dict] = []
    player_rows: list[dict] = []

    for idx in range(matches_count):
        team_a, team_b = rng.sample(teams, 2)
        tournament = tournaments[idx % len(tournaments)]
        venue = venues.get(team_a, f"{team_a} Arena")
        match_date = base_date + timedelta(days=idx)

        home_adv = 0.18
        xg_home = max(0.2, attack[team_a] * 0.72 + defense[team_b] * 0.36 + home_adv)
        xg_away = max(0.2, attack[team_b] * 0.66 + defense[team_a] * 0.32)

        home_goals = int(np.clip(np_rng.poisson(lam=max(0.15, xg_home)), 0, 5))
        away_goals = int(np.clip(np_rng.poisson(lam=max(0.15, xg_away)), 0, 5))

        today = datetime.utcnow().date()
        if match_date.date() == today:
            state = "live"
        elif match_date.date() > today:
            state = "upcoming"
        else:
            state = "historical"

        match_rows.append(
            {
                "match_id": f"{tournament.lower()}_{match_date:%Y%m%d}_{idx:04d}",
                "sport": "football",
                "tournament": tournament,
                "team_a": team_a,
                "team_b": team_b,
                "venue": venue,
                "match_date": match_date.date().isoformat(),
                "state": state,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_xg": round(float(xg_home), 3),
                "away_xg": round(float(xg_away), 3),
            }
        )

    # Team profiles
    team_rows = []
    for team in teams:
        team_rows.append(
            {
                "team": team,
                "attack_index": round(float(attack[team]), 3),
                "defense_index": round(float(defense[team]), 3),
                "xg_for": round(float(attack[team] * 0.95), 3),
                "xg_against": round(float(defense[team] * 0.96), 3),
                "form_points_last5": round(float(form[team]), 3),
                "home_advantage": 0.18,
                "tournament": "EPL" if team in teams[:8] else "UCL",
            }
        )

    # Player form rows
    for team in teams:
        roster = FOOTBALL_TEAM_PLAYERS.get(team, [])
        if len(roster) < 7:
            roster = roster + [f"{team} XI {idx}" for idx in range(len(roster) + 1, 8)]
        for player_idx, player_name in enumerate(roster[:7], start=1):
            role = "standout"
            if player_idx <= 2:
                role = "goal_scorer"
            elif player_idx >= 6:
                role = "defender"

            goals_last5 = max(0, int(np.clip(np_rng.normal(2.0 if role == "goal_scorer" else 1.0, 1.2), 0, 7)))
            xg_per90 = float(np.clip(np_rng.normal(0.55 if role == "goal_scorer" else 0.28, 0.14), 0.05, 1.1))
            shot_conversion = float(np.clip(np_rng.normal(0.19 if role == "goal_scorer" else 0.12, 0.05), 0.03, 0.45))
            defensive_actions = float(np.clip(np_rng.normal(6.5 if role == "defender" else 3.0, 1.8), 0.5, 15.0))
            key_passes = float(np.clip(np_rng.normal(2.5, 1.0), 0.2, 6.5))
            impact_score = float(
                np.clip(
                    goals_last5 * 6.5
                    + xg_per90 * 30.0
                    + shot_conversion * 42.0
                    + defensive_actions * 1.1
                    + key_passes * 2.0,
                    30,
                    95,
                )
            )

            player_rows.append(
                {
                    "player": player_name,
                    "team": team,
                    "sport": "football",
                    "tournament": "EPL" if team in teams[:8] else "UCL",
                    "role": role,
                    "goals_last5": goals_last5,
                    "xg_per90": round(xg_per90, 3),
                    "shot_conversion": round(shot_conversion, 3),
                    "def_actions": round(defensive_actions, 3),
                    "key_passes": round(key_passes, 3),
                    "impact_score": round(impact_score, 3),
                    "form_points_last5": round(float(form[team]), 3),
                }
            )

    processed_dir = config.processed_data_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    matches_df = pd.DataFrame(match_rows)
    teams_df = pd.DataFrame(team_rows)
    players_df = pd.DataFrame(player_rows)

    matches_df.to_csv(processed_dir / "football_matches.csv", index=False)
    teams_df.to_csv(processed_dir / "football_team_profiles.csv", index=False)
    players_df.to_csv(processed_dir / "football_player_form_latest.csv", index=False)

    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "sport": "football",
        "num_matches": int(len(matches_df)),
        "num_teams": int(teams_df["team"].nunique()),
        "num_players": int(players_df["player"].nunique()),
        "tournaments": sorted(matches_df["tournament"].unique().tolist()),
        "states": sorted(matches_df["state"].unique().tolist()),
    }

    manifests_dir = config.repo_root / "data" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / "football_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"Football pipeline complete: {len(matches_df)} matches, {len(teams_df)} teams, "
        f"{len(players_df)} player rows"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build football fixtures and feature-ready profile tables")
    parser.add_argument("--matches", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-bootstrap", action="store_true")
    args = parser.parse_args()

    config = MLConfig()
    ensure_directories(config)

    if args.force_bootstrap:
        bootstrap_football_datasets(config=config, matches_count=args.matches, seed=args.seed)
        return

    output_path = config.processed_data_dir / "football_matches.csv"
    if output_path.exists():
        print("Football datasets already exist. Use --force-bootstrap to regenerate.")
        return

    bootstrap_football_datasets(config=config, matches_count=args.matches, seed=args.seed)


if __name__ == "__main__":
    main()
