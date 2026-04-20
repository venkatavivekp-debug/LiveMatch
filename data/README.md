# Data Layout

LiveMatch uses a layered data layout to keep ingestion, feature engineering, and runtime assets traceable.

Provider intent:

- `Cricsheet` => historical/offline training foundation.
- `CricAPI` (or compatible live endpoint) => live/upcoming context conditioning.
- mock provider => explicit local testing fallback (not primary production path).

## `raw/`

Downloaded source data before transformation.

Examples:

- Cricsheet IPL JSON zip and extracted files
- Future external feed dumps for football/cricket APIs

Recommended cricket ingestion flow:

1. `python3 -m ml.data_ingest`
2. `python3 -m ml.features`

## `processed/`

Model-ready and API-ready assets.

Cricket assets:

- `matches.csv`
- `player_match_stats.csv`
- `model_features.csv`
- `match_feature_lookup.csv`
- `team_profiles.csv`
- `venue_profiles.csv`
- `player_form_latest.csv`
- `feature_manifest.json`

Football assets:

- `football_matches.csv`
- `football_team_profiles.csv`
- `football_player_form_latest.csv`

Generate football assets:

- `python3 -m ml.football_features --force-bootstrap`

## `external/`

Integration staging area for provider-specific payload snapshots.

Use this folder for temporary API payload captures when building a real live-feed connector.
It also stores live-provider cache/sync files used by backend refresh/status flows.

## `manifests/`

Data lineage and generation manifests.

Current examples:

- `football_manifest.json`

## Bootstrap/Fallback Flow

If external ingestion is blocked, use deterministic bootstrap to keep the platform runnable:

- `python3 -m ml.features --force-bootstrap`
- `python3 -m ml.football_features --force-bootstrap`

Use bootstrap only for local/dev bring-up. Do not treat bootstrap outputs as production truth data.
