# IEAF Fraud Monitoring — Streamlit App v2 (AutoGluon Edition)

An improved version of the IEAF demonstration app. The biggest change: **predictions now run
through AutoGluon itself** — the framework's actual AutoML engine — not a single hand-picked model,
so this app demonstrates the dissertation's AutoML pipeline working end to end.

## Deployment troubleshooting history

If you ever redeploy this from scratch and hit an install error, check here first — these are the
real problems already hit and fixed while getting this app running on Streamlit Community Cloud, in
the order they came up:

1. **`pyarrow` build failure on Python 3.14.** Streamlit Cloud defaulted to a very new Python version
   that had no ready-built `pyarrow` wheel, so pip tried compiling it from source and failed (missing
   `cmake`). **Fix:** pinned the Python version to 3.11 via the app's own Settings page (Manage app →
   Settings → Python version) — not a `runtime.txt` file, which Streamlit Cloud no longer reads.
2. **`ModuleNotFoundError: No module named 'pkg_resources'`.** AutoGluon checks package versions on
   load using `pkg_resources`, which comes from `setuptools` — not auto-installed by Streamlit Cloud's
   installer in a fresh environment. **Fix:** added `setuptools` to `requirements.txt`.
3. **AutoGluon version mismatch warning** (predictor saved with 1.5.0, environment installed 1.3.1,
   since `requirements.txt` had `autogluon.tabular>=1.0` with no upper bound). **Fix:** pinned
   `autogluon.tabular==1.5.0`, the exact version the model was saved with.
4. **`ResolutionImpossible` — conflicting requirements.** Fixing #1 with `pyarrow>=21` and fixing #3
   with `autogluon.tabular==1.5.0` directly contradicted each other, since AutoGluon 1.5.0 requires
   `pyarrow<21.0.0`. **Fix:** removed the now-unnecessary `pyarrow>=21` pin entirely, since Python is
   pinned to 3.11 via Settings now, so any `pyarrow` version AutoGluon wants has a ready-built wheel.

**The `requirements.txt` in this version is unchanged from the last working one** — nothing about the
Executive Summary & Insights page needed a new package, deliberately, to avoid reopening any of the
above.

## What's new in this version

- **New page: Executive Summary & Insights.** Positioned right before About in the navigation, this is
  the decision-support layer of the app: a plain-language executive summary, automatically-generated
  key insights (colour-coded green/amber/red by tone), chart-by-chart interpretation, evidence-based
  recommendations with stated risks, an honest confidence-and-limitations section, business impact,
  and next steps. It reads the app's own real, already-computed results and interprets them — it does
  not call any external AI service and adds no new dependencies, so it can't reintroduce the
  deployment problems worked through below.
- **Pages now talk to each other.** Upload a file on the Upload page, and its results follow you:
  - The **Dashboard** gains a "Your Uploaded Data (This Session)" section with live KPIs for your file, shown alongside (never replacing) the dissertation's own fixed Chapter Four results.
  - The **AutoML Selection & Scorecards** page gains a live scorecard (precision, recall, F1, PR-AUC, MCC, confusion matrix) computed on your file — but only if it includes an `is_fraud` column, since those metrics need to know which transactions were really fraud.
  - The **Manual Prediction** page pre-fills its behavioural fields (recent transaction counts, average amount) using your uploaded file's own averages, instead of generic starting numbers.
  - A **sidebar status panel** shows whether a file is currently loaded this session, and a **"Clear uploaded data"** button resets everything back to the default view.
- This uses Streamlit's `session_state`, which lives only in your browser session — nothing is saved to a shared database, and other visitors never see your uploaded data.

## What's new in v2 (still included)

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
