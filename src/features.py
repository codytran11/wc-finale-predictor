"""
features.py
------------
Builds the model-ready feature table from the raw `matches` table:

  - Elo ratings (Python, sequential update match-by-match)
  - Head-to-head history: this specific team's win rate vs this specific
    opponent, based on all their past meetings (Python, sequential)
  - Rolling form: goals for/against, points (SQL window functions)

Output: a `match_features` table in Postgres, one row per match, with
pre-match features for both teams. This is what model.py will train on.
"""

import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "worldcup"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

K = 20               # how much Elo ratings move per match
HOME_ADVANTAGE = 65   # elo points added to the home team (skipped if neutral venue)
FORM_WINDOW = 5       # number of past matches used for rolling form


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def h2h_key(a: str, b: str) -> tuple:
    """Order-independent key so Argentina-vs-Spain and Spain-vs-Argentina
    share the same head-to-head record."""
    return tuple(sorted([a, b])) # guarantees both produce the identical key ("Argentina", "Spain") 
                                # regardless of who's home or away, so we always look up the same dictionary entry. 


def compute_elo_and_h2h(matches: pd.DataFrame) -> pd.DataFrame:
    """Both Elo and head-to-head record depend on match order, so we
    compute them together in a single pass through the data."""
    elo: dict[str, float] = {}
    h2h: dict[tuple, dict] = {}

    def get_elo(team: str) -> float:
        return elo.get(team, 1500.0)

    home_elo, away_elo = [], []
    home_h2h_winrate, away_h2h_winrate, h2h_matches_played = [], [], []

    for _, row in matches.iterrows():
        h, a = row["home_team"], row["away_team"]

        he, ae = get_elo(h), get_elo(a)
        home_elo.append(he)
        away_elo.append(ae)

        key = h2h_key(h, a)
        record = h2h.get(key, {})
        h_wins = record.get(h, 0)
        a_wins = record.get(a, 0)
        draws = record.get("draws", 0)
        total = h_wins + a_wins + draws
        home_h2h_winrate.append(h_wins / total if total > 0 else 0.5)
        away_h2h_winrate.append(a_wins / total if total > 0 else 0.5)
        h2h_matches_played.append(total)

        if pd.notna(row["home_score"]) and pd.notna(row["away_score"]):
            hs, as_ = row["home_score"], row["away_score"]

            adj_he = he + (0 if row["neutral"] else HOME_ADVANTAGE)
            expected_home = 1 / (1 + 10 ** ((ae - adj_he) / 400))
            actual_home = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
            elo[h] = he + K * (actual_home - expected_home)
            elo[a] = ae + K * ((1 - actual_home) - (1 - expected_home))

            if hs > as_:
                record[h] = record.get(h, 0) + 1
            elif hs < as_:
                record[a] = record.get(a, 0) + 1
            else:
                record["draws"] = record.get("draws", 0) + 1
            h2h[key] = record

    matches = matches.copy()
    matches["home_elo"] = home_elo
    matches["away_elo"] = away_elo
    matches["elo_difference"] = matches["home_elo"] - matches["away_elo"]
    matches["home_h2h_winrate"] = home_h2h_winrate
    matches["away_h2h_winrate"] = away_h2h_winrate
    matches["h2h_winrate_difference"] = (
        matches["home_h2h_winrate"] - matches["away_h2h_winrate"]
    )
    matches["h2h_matches_played"] = h2h_matches_played
    return matches


FORM_SQL = """
WITH team_matches AS (
    SELECT date, home_team AS team, away_team AS opponent,
           home_score AS goals_for, away_score AS goals_against,
           CASE WHEN home_score > away_score THEN 3
                WHEN home_score = away_score THEN 1
                ELSE 0 END AS points
    FROM matches
    UNION ALL
    SELECT date, away_team AS team, home_team AS opponent,
           away_score AS goals_for, home_score AS goals_against,
           CASE WHEN away_score > home_score THEN 3
                WHEN away_score = home_score THEN 1
                ELSE 0 END AS points
    FROM matches
)
SELECT
    date, team,
    AVG(goals_for) OVER w AS goals_for_form,
    AVG(goals_against) OVER w AS goals_against_form,
    AVG(points) OVER w AS points_form
FROM team_matches
WINDOW w AS (
    PARTITION BY team ORDER BY date
    ROWS BETWEEN %(window)s PRECEDING AND 1 PRECEDING
)
"""


def compute_form(conn) -> pd.DataFrame:
    return pd.read_sql(FORM_SQL, conn, params={"window": FORM_WINDOW})


def build_features() -> pd.DataFrame:
    conn = get_connection()

    matches = pd.read_sql("SELECT * FROM matches ORDER BY date", conn)
    matches = compute_elo_and_h2h(matches)

    form = compute_form(conn)
    home_form = form.rename(
        columns={c: f"home_{c}" for c in form.columns if c not in ("date", "team")}
    )
    away_form = form.rename(
        columns={c: f"away_{c}" for c in form.columns if c not in ("date", "team")}
    )

    matches = matches.merge(
        home_form, left_on=["date", "home_team"], right_on=["date", "team"], how="left"
    ).drop(columns="team")
    matches = matches.merge(
        away_form, left_on=["date", "away_team"], right_on=["date", "team"], how="left"
    ).drop(columns="team")

    matches["attack_form_difference"] = (
        matches["home_goals_for_form"] - matches["away_goals_for_form"]
    )
    matches["defense_form_difference"] = (
        matches["home_goals_against_form"] - matches["away_goals_against_form"]
    )
    matches["points_form_difference"] = (
        matches["home_points_form"] - matches["away_points_form"]
    )

    conn.close()
    return matches


def save_features(df: pd.DataFrame) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS match_features")

    cols = df.columns.tolist()
    col_defs = ", ".join(f'"{c}" TEXT' for c in cols)
    cur.execute(f"CREATE TABLE match_features ({col_defs})")

    placeholders = ", ".join(["%s"] * len(cols))
    rows = [tuple(None if pd.isna(v) else str(v) for v in row) for row in df.itertuples(index=False)]
    cur.executemany(f"INSERT INTO match_features VALUES ({placeholders})", rows)

    conn.commit()
    print(f"Saved {len(df)} rows to match_features")
    cur.close()
    conn.close()


if __name__ == "__main__":
    features = build_features()
    save_features(features)

    final = features[
        (features["home_team"] == "Spain") & (features["away_team"] == "Argentina")
    ].tail(1)
    print("\nArgentina vs Spain final - pre-match features:")
    print(
        final[
            [
                "date", "home_team", "away_team",
                "home_elo", "away_elo", "elo_difference",
                "home_goals_for_form", "away_goals_for_form",
                "home_h2h_winrate", "away_h2h_winrate", "h2h_matches_played",
            ]
        ].to_string(index=False)
    )