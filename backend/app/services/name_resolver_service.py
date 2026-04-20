from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

PLACEHOLDER_PATTERN = re.compile(r"\bplayer\s*\d+\b", re.IGNORECASE)

GENERIC_NAME_MARKERS = {
    "top batter",
    "top batsman",
    "top bowler",
    "lead bowler",
    "primary striker",
    "central playmaker",
    "standout player",
    "goal scorer",
    "unknown player",
}

REAL_CRICKET_ROSTERS: dict[str, list[str]] = {
    "mumbai indians": [
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
    "chennai super kings": [
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
    "royal challengers bengaluru": [
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
    "kolkata knight riders": [
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
    "rajasthan royals": [
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
    "sunrisers hyderabad": [
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
    "delhi capitals": [
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
    "punjab kings": [
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
    "lucknow super giants": [
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
    "gujarat titans": [
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

REAL_FOOTBALL_PLAYERS: dict[str, list[str]] = {
    "arsenal": ["Bukayo Saka", "Martin Odegaard", "Declan Rice", "Kai Havertz"],
    "manchester city": ["Erling Haaland", "Phil Foden", "Kevin De Bruyne", "Rodri"],
    "liverpool": ["Mohamed Salah", "Darwin Nunez", "Alexis Mac Allister", "Virgil van Dijk"],
    "tottenham hotspur": ["Son Heung-min", "James Maddison", "Dejan Kulusevski", "Cristian Romero"],
    "real madrid": ["Vinicius Junior", "Jude Bellingham", "Rodrygo", "Federico Valverde"],
    "bayern munich": ["Harry Kane", "Jamal Musiala", "Leroy Sane", "Joshua Kimmich"],
    "barcelona": ["Robert Lewandowski", "Lamine Yamal", "Pedri", "Frenkie de Jong"],
    "atletico madrid": ["Antoine Griezmann", "Alvaro Morata", "Rodrigo De Paul", "Jan Oblak"],
}


class PlayerNameResolver:
    def __init__(self, processed_dir: Path) -> None:
        self.processed_dir = processed_dir
        self._historical_cricket_rosters = self._load_historical_rosters("player_form_latest.csv")
        self._historical_football_rosters = self._load_historical_rosters("football_player_form_latest.csv")

    @staticmethod
    def _norm(value: Optional[str]) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _is_placeholder_name(name: Optional[str]) -> bool:
        normalized = str(name or "").strip()
        if not normalized:
            return True
        lowered = normalized.lower()
        if lowered in GENERIC_NAME_MARKERS:
            return True
        if PLACEHOLDER_PATTERN.search(lowered):
            return True
        if lowered.startswith(("top ", "lead ", "primary ", "central ")):
            return True
        return False

    def _load_historical_rosters(self, filename: str) -> dict[str, list[str]]:
        path = self.processed_dir / filename
        if not path.exists():
            return {}
        try:
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return {}
        if "team" not in frame.columns or "player" not in frame.columns:
            return {}
        roster: dict[str, list[str]] = {}
        for _, row in frame.iterrows():
            team = self._norm(str(row.get("team", "")))
            player = str(row.get("player", "")).strip()
            if not team or not player or self._is_placeholder_name(player):
                continue
            roster.setdefault(team, [])
            if player not in roster[team]:
                roster[team].append(player)
        return roster

    @staticmethod
    def _pick_deterministic(candidates: list[str], seed: str) -> str:
        if not candidates:
            return "Unavailable"
        digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % len(candidates)
        return candidates[idx]

    def _team_candidates(self, sport: str, team: Optional[str]) -> list[str]:
        team_key = self._norm(team)
        if sport == "cricket":
            if team_key in self._historical_cricket_rosters and self._historical_cricket_rosters[team_key]:
                return self._historical_cricket_rosters[team_key]
            return REAL_CRICKET_ROSTERS.get(team_key, [])
        if team_key in self._historical_football_rosters and self._historical_football_rosters[team_key]:
            return self._historical_football_rosters[team_key]
        return REAL_FOOTBALL_PLAYERS.get(team_key, [])

    def resolve(
        self,
        *,
        name: Optional[str],
        sport: str,
        team: Optional[str],
        role: Optional[str] = None,
        seed: Optional[str] = None,
    ) -> tuple[str, bool]:
        raw_name = str(name or "").strip()
        if raw_name and not self._is_placeholder_name(raw_name):
            return raw_name, False

        candidates = self._team_candidates(sport=self._norm(sport), team=team)
        if not candidates:
            sport_key = self._norm(sport)
            if sport_key == "football":
                football_pool = sorted({player for players in REAL_FOOTBALL_PLAYERS.values() for player in players})
                if football_pool:
                    fallback_seed = seed or f"{self._norm(team)}|{self._norm(role)}|{raw_name}"
                    return self._pick_deterministic(football_pool, fallback_seed), True
            if sport_key == "cricket":
                cricket_pool = sorted({player for players in REAL_CRICKET_ROSTERS.values() for player in players})
                if cricket_pool:
                    fallback_seed = seed or f"{self._norm(team)}|{self._norm(role)}|{raw_name}"
                    return self._pick_deterministic(cricket_pool, fallback_seed), True
            return "Unavailable", True

        resolved = self._pick_deterministic(
            candidates,
            seed or f"{self._norm(team)}|{self._norm(role)}|{raw_name}",
        )
        return resolved, True


@lru_cache(maxsize=1)
def get_player_name_resolver(processed_dir: str) -> PlayerNameResolver:
    return PlayerNameResolver(Path(processed_dir))
