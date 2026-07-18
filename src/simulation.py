"""
simulation.py
-----------
Loads the trained Poisson goal models, gets expected goals for the
Argentina vs Spain final, then runs a Monte Carlo simulation (drawing
random scorelines thousands of times) to produce:

  - 90-minute win/draw/win probabilities
  - eventual outcome probabilities (accounting for penalties on a draw)
  - the most likely scorelines
  - a seaborn bar chart of the outcome probabilities
"""

import os
import pickle
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # render to file, no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns
from dotenv import load_dotenv

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

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
N_SIMULATIONS = 100_000
PENALTY_COIN_FLIP = 0.5  # simplifying assumption: shootouts modeled as ~50/50

HOME_TEAM = "Spain"
AWAY_TEAM = "Argentina"
MATCH_DATE = "2026-07-19"


def load_models():
    with open(os.path.join(MODEL_DIR, "home_model.pkl"), "rb") as f:
        home_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "away_model.pkl"), "rb") as f:
        away_model = pickle.load(f)
    return home_model, away_model


def get_final_features() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("SELECT * FROM match_features", conn)
    conn.close()

    for col in FEATURES[:-1]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["neutral_flag"] = (df["neutral"].astype(str).str.lower() == "true").astype(int)

    row = df[
        (df["home_team"] == HOME_TEAM)
        & (df["away_team"] == AWAY_TEAM)
        & (df["date"] == MATCH_DATE)
    ]
    if row.empty:
        raise ValueError("Could not find the final match in match_features.")
    return row


def simulate(home_xg: float, away_xg: float, n: int = N_SIMULATIONS) -> dict:
    rng = np.random.default_rng(42)
    home_goals = rng.poisson(home_xg, n)
    away_goals = rng.poisson(away_xg, n)

    home_win_90 = (home_goals > away_goals).mean()
    draw_90 = (home_goals == away_goals).mean()
    away_win_90 = (home_goals < away_goals).mean()

    eventual_home = home_win_90 + draw_90 * (1 - PENALTY_COIN_FLIP)
    eventual_away = away_win_90 + draw_90 * PENALTY_COIN_FLIP

    scorelines = Counter(zip(home_goals, away_goals)).most_common(5)

    return {
        "home_win_90": home_win_90,
        "draw_90": draw_90,
        "away_win_90": away_win_90,
        "eventual_home": eventual_home,
        "eventual_away": eventual_away,
        "top_scorelines": scorelines,
        "n": n,
    }


def print_results(results: dict) -> None:
    print("90-minute probabilities:")
    print(f"  {HOME_TEAM} win: {results['home_win_90']:.1%}")
    print(f"  Draw:      {results['draw_90']:.1%}")
    print(f"  {AWAY_TEAM} win: {results['away_win_90']:.1%}")

    print("\nEventual outcome probabilities (penalties resolve a draw):")
    print(f"  {HOME_TEAM}:     {results['eventual_home']:.1%}")
    print(f"  {AWAY_TEAM}: {results['eventual_away']:.1%}")

    print("\nMost likely scorelines:")
    for (h, a), count in results["top_scorelines"]:
        print(f"  {HOME_TEAM} {h} - {a} {AWAY_TEAM}: {count / results['n']:.1%}")


def plot_results(results: dict, home_xg: float, away_xg: float) -> None:
    sns.set_style("white")

    labels = [f"{HOME_TEAM} win", "Draw", f"{AWAY_TEAM} win"]
    values = [results["home_win_90"], results["draw_90"], results["away_win_90"]]
    colors = ["#C60B1E", "#B0B0B0", "#75AADB"]  # Spain red, neutral gray, Argentina sky blue

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")

    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white", linewidth=1.5, zorder=3)

    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

    ax.set_yticks([])
    ax.set_ylim(0, max(values) * 1.35)

    for bar, val, color in zip(bars, values, colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{val:.1%}", ha="center", va="bottom",
            fontsize=17, fontweight="bold", color=color,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12, fontweight="medium", color="#333333")
    ax.tick_params(axis="x", length=0)

    fig.suptitle(f"{HOME_TEAM} vs {AWAY_TEAM} — World Cup Final", fontsize=16, fontweight="bold", y=0.98, color="#1a1a1a")
    ax.set_title(
        f"90-Minute Outcome Probabilities  |  Predicted: {AWAY_TEAM} {away_xg:.2f} – {home_xg:.2f} {HOME_TEAM}",
        fontsize=10.5, color="#666666", pad=15,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(os.path.dirname(__file__), "..", "final_prediction.png")
    plt.savefig(out_path, dpi=160, facecolor="white")
    print(f"\nSaved chart to {out_path}")


if __name__ == "__main__":
    home_model, away_model = load_models()
    final_features = get_final_features()

    home_xg = home_model.predict(final_features[FEATURES])[0]
    away_xg = away_model.predict(final_features[FEATURES])[0]
    print(f"Expected goals -> {HOME_TEAM}: {home_xg:.2f}, {AWAY_TEAM}: {away_xg:.2f}\n")

    results = simulate(home_xg, away_xg)
    print_results(results)
    plot_results(results, home_xg, away_xg)