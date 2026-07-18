# wc-finale-predictor
Predicting the 2026 World Cup winner (Argentina vs Spain) with a Poisson goal model trained on 150 years of international match data.

Inspired by @lerabyte on tiktok and https://data.fivethirtyeight.com/

Database: https://raw.githubusercontent.com/martj42/international_results/master/results.csv
  - Every international football match from 1872 all the way until now

## Structure: 
  - Gather and import the historical match dataset (49,000+ international football results, 1872–present) and loads it into a local PostgreSQL database ~ **ingest.py**
  - Builds Elo rating, head-to-head records, and rolling form using SQL + pandas ~ **features.py**
  - Trains two separate models - one predicts the home team's expected goals, one predicts the away team's. Matches before 2025 are used for training; matches from 2025 onward are held out     entirely and used only to validate accuracy on genuinely unseen data. The trained models are saved to disk so they don't need to be retrained every time a prediction is made ~ **model.py**
  - Runs a Monte Carlo simulation (100,000 simulated matches) to convert expected goals into win/draw/win probabilities, then visualizes the result ~ **simulation.py**

## Methodology:
  - Elo System: Every team starts at **1500**. After each match: expected_home_win = 1 / (1 + 10^((away_elo - home_elo) / 400))
    new_elo = old_elo + K × (actual_result − expected_result)
  - Head-to-Head History: For every match, the model calculates: out of all previous meetings between *these exact two teams*, what fraction did each team win?
  - Rolling Form: A team's current strength using SQL window functions to calculate each team's average goals scored and conceded over just their **last 5 matches** going into every game

## SetUp:
**Prerequisites:** Python 3, PostgreSQL installed and running locally, with a database created 
- Open a terminal in this folder and run:
  - pip install -r requirements.txt
    python src/ingest.py
    python src/features.py
    python src/model.py
    python src/simulation.py
- Outputs:
- `final_prediction.png` — the outcome probability chart (saved to the project root)
- `data/processed/home_model.pkl` / `away_model.pkl` — trained models (gitignored, regenerate by rerunning `model.py`)


## Simulation: 
  - Draws on the Poisson model: provides a simple, fact-based mathematical framework to model rare events (like goals) that happen at a known average rate. Goals are a count, which is always     0 or a positive whole number, with small values (0, 1, 2) far more common than large ones, which is exactly the shape a Poisson distribution describes.
  - Argentina vs Spain is simulated **100,000 times**: each simulation draws a random, Poisson-distributed scoreline centered on those expected goals, and the outcomes are       tallied. Draws are then split into eventual winners assuming a 50/50 penalty shootout, since penalties are most of the time.
  - The models are validated on a held-out, more recent time period they never trained on (2025–2026 matches), producing **62% match outcome accuracy**

## Result:
**Predicted: Argentina to win the 2026 World Cup Final, 57% to 43% over Spain.**
- Expected goals: Argentina 1.53 – 1.22 Spain
- 90-minute probabilities: Argentina win 44.4% / Draw 25.2% / Spain win 30.4%
- Most likely 90-minute scoreline: Spain 1 – 1 Argentina (12.0%)
   - Other Possible Scores:
    - Spain 0 - 1 Argentina: 9.8%
    - Spain 1 - 2 Argentina: 9.1%
    - Spain 1 - 0 Argentina: 7.8%
    - Spain 0 - 2 Argentina: 7.6%
- Eventual winner probabilities (accounting for penalties on a draw): Argentina 57.0% / Spain 43.0%

## Data Visualization:
- Built with matplotlib and seaborn — a styled bar chart showing 90-minute win/draw/win probabilities
- <img width="1120" height="928" alt="image" src="https://github.com/user-attachments/assets/f102526f-8080-48a4-b77c-b7ed2bb46a88" />

    
