"""
ingest.py
---------
Downloads the historical international football results dataset and
loads it into a local PostgreSQL database so we can query it with SQL.

Data source: martj42/international_results (GitHub, mirrors the Kaggle
"International football results from 1872 to <year>" dataset).
"""

import csv
import os
import urllib.request
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv()  # reads the .env file into environment variables

# ---- paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "results.csv"

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# ---- db connection settings (from .env) -------------------------------
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "worldcup"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}


def download_csv() -> None:
    """Download the raw results CSV if we don't already have it."""
    RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, RAW_CSV_PATH)
    print(f"Saved to {RAW_CSV_PATH}")


def load_into_postgres() -> None:
    """Load the CSV into a `matches` table in PostgreSQL."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS matches")
    cur.execute(
        """
        CREATE TABLE matches (
            date DATE,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            tournament TEXT,
            city TEXT,
            country TEXT,
            neutral BOOLEAN
        )
        """
    )

    with open(RAW_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            (
                row["date"],
                row["home_team"],
                row["away_team"],
                # scores can be blank ("NA") for matches not yet played
                int(row["home_score"]) if row["home_score"] not in ("", "NA") else None,
                int(row["away_score"]) if row["away_score"] not in ("", "NA") else None,
                row["tournament"],
                row["city"],
                row["country"],
                row["neutral"].strip().upper() == "TRUE",
            )
            for row in reader
        ]

    cur.executemany(
        """
        INSERT INTO matches
        (date, home_team, away_team, home_score, away_score, tournament, city, country, neutral)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        rows,
    )
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM matches")
    count = cur.fetchone()[0]
    print(f"Loaded {count} matches into PostgreSQL database '{DB_CONFIG['dbname']}'")

    cur.close()
    conn.close()


if __name__ == "__main__":
    if not RAW_CSV_PATH.exists():
        download_csv()
    else:
        print(f"Found existing CSV at {RAW_CSV_PATH}, skipping download.")
    load_into_postgres()