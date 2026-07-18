"""
model.py
--------
Trains two Poisson gradient-boosted regressors on match_features:
  - one predicts the home team's expected goals
  - one predicts the away team's expected goals

Validates on a held-out, more recent time period (never trained on) to get
an honest read on accuracy, then predicts the actual Argentina vs Spain
final. Saves both trained models to disk for simulate.py to use next.
"""

import os
import pickle

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "worldcup"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

FEATURES = [
    "home_elo", "away_elo", "elo_difference",
    "home_goals_for_form", "away_goals_for_form",
    "home_h2h_winrate", "away_h2h_winrate",
    "neutral_flag",
]

VALIDATION_START_DATE = "2025-01-01"  # train before this, validate on/after this
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_features() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("SELECT * FROM match_features", conn)
    conn.close()

    # match_features was stored as TEXT (see features.py), so cast back to numbers
    numeric_cols = [
        "home_score", "away_score", "home_elo", "away_elo", "elo_difference",
        "home_goals_for_form", "away_goals_for_form",
        "home_goals_against_form", "away_goals_against_form",
        "home_h2h_winrate", "away_h2h_winrate", "h2h_matches_played",
        "h2h_winrate_difference", "home_points_form", "away_points_form",
        "attack_form_difference", "defense_form_difference", "points_form_difference",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["neutral_flag"] = (df["neutral"].astype(str).str.lower() == "true").astype(int)
    return df


def train_and_validate(df: pd.DataFrame):
    played = df.dropna(subset=["home_score", "away_score"]).dropna(subset=FEATURES)
    train = played[played["date"] < VALIDATION_START_DATE]
    val = played[played["date"] >= VALIDATION_START_DATE]

    print(f"Training on {len(train)} matches, validating on {len(val)} matches")

    home_model = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=200, random_state=42
    )
    away_model = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=200, random_state=42
    )
    home_model.fit(train[FEATURES], train["home_score"])
    away_model.fit(train[FEATURES], train["away_score"])

    pred_home = home_model.predict(val[FEATURES])
    pred_away = away_model.predict(val[FEATURES])

    mae_home = np.mean(np.abs(pred_home - val["home_score"]))
    mae_away = np.mean(np.abs(pred_away - val["away_score"]))

    actual_outcome = np.sign(val["home_score"] - val["away_score"])
    pred_outcome = np.sign(pred_home - pred_away)
    accuracy = (actual_outcome == pred_outcome).mean()

    print(f"Home goal MAE: {mae_home:.3f}")
    print(f"Away goal MAE: {mae_away:.3f}")
    print(f"Match outcome accuracy (validation): {accuracy:.3f}")

    return home_model, away_model


def save_models(home_model, away_model) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "home_model.pkl"), "wb") as f:
        pickle.dump(home_model, f)
    with open(os.path.join(MODEL_DIR, "away_model.pkl"), "wb") as f:
        pickle.dump(away_model, f)
    print(f"Saved trained models to {MODEL_DIR}")


def predict_final(df: pd.DataFrame, home_model, away_model) -> None:
    final_row = df[
        (df["home_team"] == "Spain")
        & (df["away_team"] == "Argentina")
        & (df["date"] == "2026-07-19")
    ]
    if final_row.empty:
        print("Could not find the Spain vs Argentina final row in match_features.")
        return

    fh = home_model.predict(final_row[FEATURES])[0]
    fa = away_model.predict(final_row[FEATURES])[0]
    print(f"\nFinal prediction (expected goals):")
    print(f"  Spain:     {fh:.2f}")
    print(f"  Argentina: {fa:.2f}")


if __name__ == "__main__":
    df = load_features()
    home_model, away_model = train_and_validate(df)
    save_models(home_model, away_model)
    predict_final(df, home_model, away_model)