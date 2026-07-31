# IEAF Fraud Monitoring — Streamlit App v2 (AutoGluon Edition)

An improved version of the IEAF demonstration app. The biggest change: **predictions now run
through AutoGluon itself** — the framework's actual AutoML engine — not a single hand-picked model,
so this app demonstrates the dissertation's AutoML pipeline working end to end.

## What's new in v2

- **AutoGluon does the scoring**, not XGBoost. The app loads the real, trained `TabularPredictor`
  and uses whichever model it automatically selected as best (`WeightedEnsemble_L2` in this run).
- **New page: AutoML Selection & Scorecards** — shows AutoGluon's full leaderboard (validation score
  vs. held-out test score side by side), which model it picked and why, plus performance scorecards
  and confusion matrices for every candidate model studied in the dissertation.
- **New: Time-series analysis on your own uploaded data** — if your CSV has a `timestamp` column, the
  app shows daily volume, daily fraud rate, and a full seasonal decomposition (trend / weekly pattern
  / residual), the same method used in Chapter Four.
- **Explainability kept, using the correct method for an ensemble** — SHAP's permutation explainer
  (not TreeSHAP, since AutoGluon's best model is an ensemble, not a single tree model — this
  distinction is explained in-app).

## What's inside

- `app.py` — the Streamlit app (5 pages)
- `feature_engineering.py` — reproduces the dissertation's exact feature-engineering pipeline
- `model_artifacts/` — the trained AutoGluon predictor (~47 MB), the manually-tuned XGBoost model
  (used only for comparison, not for live predictions in this version), and the real metrics from
  Chapter Four
- `sample_transactions.csv` — real sample data to try the Upload page immediately
- `screenshots/` — a preview of every page, captured from a live run of this exact app
- `requirements.txt` — Python dependencies (heavier than v1, since AutoGluon itself is a large
  package — expect the first install to take a few minutes)

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

First load takes a few seconds while AutoGluon's predictor loads into memory; after that it's cached
for the rest of the session, and predictions typically return in under 100 ms.

## Deploy it for free

**Streamlit Community Cloud** (https://share.streamlit.io) works the same way as before — push this
folder to a GitHub repository, connect it, and point it at `app.py`. Because AutoGluon and its model
folder are larger than v1's dependencies, expect a longer first build time on the hosting side.

## An honest trade-off, shown on purpose

Chapter Four found that AutoGluon's ensemble and a manually-tuned XGBoost model are statistically
indistinguishable in accuracy (McNemar p = 0.180), but AutoGluon is roughly 40x slower per
transaction. This app uses AutoGluon anyway, because the point of this version is to demonstrate the
AutoML pipeline actually working — and because ~16 ms per prediction is still fast enough to feel
instant to someone using the app. The AutoML Selection & Scorecards page shows this trade-off
directly, rather than hiding it.

## A note on the data and the model

The model was trained on a disclosed, **synthetic proxy dataset** (Chapter Four, Section 4.0.1),
since real bank transaction data could not be accessed for this dissertation. Treat any prediction
from this app as a demonstration of how the framework behaves, not a real-world fraud-risk
assessment.
