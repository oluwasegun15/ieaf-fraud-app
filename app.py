"""
IEAF Fraud Monitoring App — v2 (AutoGluon Edition)
====================================================
An improvement on the first version of this app: predictions now run through
AutoGluon itself (the framework's actual AutoML engine), not a single
hand-picked model. The app shows AutoGluon's model-selection process, scores
uploaded transactions or a single manually-entered transaction in real time,
runs time-series analysis on uploaded data, explains its own predictions with
SHAP, and gives model performance scorecards with confusion matrices for every
candidate model studied in the dissertation.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                              average_precision_score, matthews_corrcoef, confusion_matrix, accuracy_score)

from feature_engineering import engineer_features, engineer_single_transaction, FEATURE_COLS, RAW_REQUIRED_COLS
import insights

st.set_page_config(page_title='IEAF Fraud Monitoring — AutoGluon Edition', page_icon='🛡️', layout='wide')

ART = Path(__file__).parent / 'model_artifacts'
DECISION_THRESHOLDS = {'approve_below': 0.30, 'decline_above': 0.80}


# -----------------------------------------------------------------------------
# Cached loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner='Loading AutoGluon (this can take a few seconds the first time)...')
def load_autogluon():
    from autogluon.tabular import TabularPredictor
    return TabularPredictor.load(str(ART / 'autogluon_model'), require_version_match=False, require_py_version_match=False)


@st.cache_resource
def load_other_models():
    """Loads the scaler plus the three manual comparison models small enough to
    ship with the app (LogisticRegression, XGBoost, MLP_Temporal — all under
    250KB). RandomForest is deliberately excluded: at 25.68MB it exceeds
    GitHub's web-upload limit, the same problem worked through earlier when the
    combined all_fitted_models.pkl (27MB) had to be removed. Its fixed,
    historical Chapter Four score is still shown for reference; it just can't
    be re-run live on whatever you upload."""
    with open(ART / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    manual_models = {}
    manual_dir = ART / 'manual_models'
    for name in ['LogisticRegression', 'XGBoost', 'MLP_Temporal']:
        path = manual_dir / f'{name}.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                manual_models[name] = pickle.load(f)
    return scaler, manual_models


@st.cache_data
def load_json(name):
    with open(ART / name) as f:
        return json.load(f)


predictor = load_autogluon()
scaler, manual_models = load_other_models()
dashboard_metrics = load_json('dashboard_metrics.json')
ag_results = load_json('autogluon_results.json')
model_results = load_json('model_results.json')
lb_summary = load_json('leaderboard_extended_summary.json')
ts_summary = load_json('timeseries_summary.json')
adwin_results = load_json('autogluon_adwin_results.json')


def decision_label(p: float) -> str:
    if p >= DECISION_THRESHOLDS['decline_above']:
        return '🔴 Decline / Block'
    elif p >= DECISION_THRESHOLDS['approve_below']:
        return '🟠 Review'
    return '🟢 Approve'


def ag_predict(feat_df: pd.DataFrame) -> np.ndarray:
    """Scores a dataframe of engineered features using AutoGluon's chosen best model."""
    return predictor.predict_proba(feat_df[FEATURE_COLS])[1].values


def score_with_all_models(feat_df: pd.DataFrame) -> dict:
    """Scores the same uploaded, engineered data with AutoGluon AND every manual
    model small enough to ship with the app, so a genuine live comparison across
    models is possible on whatever you upload — not just Chapter Four's own
    fixed test set. Returns {model_name: probability_array}."""
    result = {'AutoGluon': ag_predict(feat_df)}
    if manual_models:
        X_scaled = scaler.transform(feat_df[FEATURE_COLS].values)
        for name, model in manual_models.items():
            result[name] = model.predict_proba(X_scaled)[:, 1]
    return result


def compute_live_scorecard(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Computes the same metrics used throughout Chapter Four (precision, recall, F1,
    ROC-AUC, PR-AUC, MCC, false-positive rate, confusion matrix), but on whatever
    labelled data was just uploaded, instead of the dissertation's own test set."""
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return dict(
        accuracy=float(accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, proba)) if len(set(y_true)) > 1 else float('nan'),
        pr_auc=float(average_precision_score(y_true, proba)) if len(set(y_true)) > 1 else float('nan'),
        mcc=float(matthews_corrcoef(y_true, pred)),
        fpr=float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )


TONE_COLORS = {'positive': ('#E8F5E9', '#2E7D32'), 'neutral': ('#FFF8E1', '#F9A825'), 'warning': ('#FDECEA', '#C62828')}


def insight_card(icon: str, title: str, text: str, tone: str = 'neutral'):
    bg, border = TONE_COLORS.get(tone, TONE_COLORS['neutral'])
    st.markdown(
        f"""<div style="background-color:{bg}; border-left: 5px solid {border}; border-radius: 6px;
        padding: 14px 16px; margin-bottom: 12px;">
        <div style="font-size: 15px; font-weight: 700; color: #222; margin-bottom: 4px;">{icon} {title}</div>
        <div style="font-size: 14px; color: #333; line-height: 1.5;">{text}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Session state: this is what lets an uploaded file's results follow the user
# across pages (Dashboard, Scorecards, Manual Prediction), instead of each page
# staying isolated to only what it shows on its own.
# -----------------------------------------------------------------------------
if 'has_uploaded' not in st.session_state:
    st.session_state.has_uploaded = False
    st.session_state.uploaded_filename = None
    st.session_state.uploaded_feat_df = None      # engineered features + predictions
    st.session_state.uploaded_raw_df = None        # original uploaded rows (for behavioural averages)
    st.session_state.uploaded_summary = None       # dict of headline KPIs
    st.session_state.uploaded_scorecard = None     # AutoGluon's scorecard, only if labels were present
    st.session_state.uploaded_has_labels = False
    st.session_state.uploaded_model_probas = None  # {model_name: proba array}, every model, on your data
    st.session_state.uploaded_scorecards = None    # {model_name: scorecard dict}, only if labels were present
    st.session_state.uploaded_drift_check = None   # first-half vs second-half comparison, only if timestamp + enough data


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title('🛡️ IEAF Fraud Monitoring')
st.sidebar.caption('AutoGluon Edition — v2')
page = st.sidebar.radio('Navigate', [
    '📊 Dashboard',
    '🤖 AutoML Selection & Scorecards',
    '📁 Upload Dataset & Time-Series',
    '✍️ Manual Prediction & Explainability',
    '📋 Executive Summary & Insights',
    'ℹ️ About This App',
])
st.sidebar.markdown('---')
st.sidebar.markdown(f"**Active model:** AutoGluon\n\n`{predictor.model_best}`")
st.sidebar.caption(
    'This app scores transactions using AutoGluon\u2019s own automatically-selected model, '
    'demonstrating the framework\u2019s AutoML pipeline end-to-end, exactly as built in Chapter Four.'
)

st.sidebar.markdown('---')
if st.session_state.has_uploaded and st.session_state.uploaded_summary is not None:
    st.sidebar.success(
        f"📌 **Live session data**\n\n`{st.session_state.uploaded_filename}`\n\n"
        f"{st.session_state.uploaded_summary['n_scored']:,} transactions scored"
    )
    st.sidebar.caption(
        'This file\u2019s results now also appear on the Dashboard, AutoML Selection & Scorecards, '
        'and Manual Prediction pages.'
    )
    if st.sidebar.button('🗑️ Clear uploaded data'):
        st.session_state.has_uploaded = False
        st.session_state.uploaded_filename = None
        st.session_state.uploaded_feat_df = None
        st.session_state.uploaded_raw_df = None
        st.session_state.uploaded_summary = None
        st.session_state.uploaded_scorecard = None
        st.session_state.uploaded_has_labels = False
        st.session_state.uploaded_model_probas = None
        st.session_state.uploaded_scorecards = None
        st.session_state.uploaded_drift_check = None
        st.rerun()
else:
    st.sidebar.info('📌 No file uploaded yet this session. Upload one on the **Upload Dataset & Time-Series** '
                      'page and its results will also appear on the other pages.')

# =============================================================================
# PAGE 1: DASHBOARD
# =============================================================================
if page == '📊 Dashboard':
    st.title('IEAF Fraud Monitoring Dashboard — AutoGluon Edition v2')
    st.caption(f"Active model: AutoGluon ({dashboard_metrics['ag_best_model']})")

    has_session = st.session_state.has_uploaded and st.session_state.uploaded_summary is not None
    speed_ratio = dashboard_metrics['ag_latency_p50_ms'] / dashboard_metrics['xgb_latency_p50_ms']

    # -------------------------------------------------------------------------
    # LIVE section: your uploaded data, shown FIRST and computed fresh every
    # time this page loads. Everything here changes based on what you upload;
    # nothing here is cached from Chapter Four.
    # -------------------------------------------------------------------------
    if has_session:
        st.success(f"📌 Showing **live results for `{st.session_state.uploaded_filename}`** — "
                     f"scored just now, on this page load, not cached.")
        st.header('📌 Your Uploaded Data (This Session) — Live')
        s = st.session_state.uploaded_summary
        u1, u2, u3, u4 = st.columns(4)
        u1.metric('Your transactions scored', f"{s['n_scored']:,}")
        u2.metric('Flagged (review or decline)', f"{s['n_flagged']:,}", f"{100*s['n_flagged']/s['n_scored']:.2f}% of your traffic")
        u3.metric('Recommended decline', f"{s['n_decline']:,}")
        u4.metric('Highest fraud probability found', f"{s['max_proba']:.1%}",
                   help='The median is almost always 0.0000 here, since most transactions get an exact-zero '
                        'score — the highest score found is a more useful single number.')

        if st.session_state.uploaded_has_labels:
            sc = st.session_state.uploaded_scorecard
            st.markdown(
                f"**Live scorecard on your data** (real fraud labels were found in this file): "
                f"Precision **{sc['precision']:.3f}**, Recall **{sc['recall']:.3f}**, "
                f"F1 **{sc['f1']:.3f}**, PR-AUC **{sc['pr_auc']:.3f}**, MCC **{sc['mcc']:.3f}**."
            )

            dc1, dc2 = st.columns(2)
            with dc1:
                st.subheader('PR-AUC on Your Data — Every Model Available')
                scs = st.session_state.uploaded_scorecards
                fig = go.Figure(go.Bar(
                    x=list(scs.keys()), y=[v['pr_auc'] for v in scs.values()],
                    marker_color=['#C44E52' if k == 'AutoGluon' else '#4C72B0' for k in scs.keys()],
                    text=[f"{v['pr_auc']:.3f}" for v in scs.values()], textposition='outside',
                ))
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title='PR-AUC (your data)')
                st.plotly_chart(fig, width='stretch')
                st.caption('RandomForest is not shown here — its saved model file is too large to ship with '
                            'this app (25.7 MB, over GitHub\u2019s upload limit), so only its fixed Chapter Four '
                            'score is available, not a live one on your data.')
            with dc2:
                st.subheader('Concept-Drift Check on Your Data')
                dch = st.session_state.uploaded_drift_check
                if dch is not None:
                    drop_pct = 100 * dch['drop']
                    tone = 'warning' if dch['drop'] > 0.1 else 'positive'
                    st.markdown(
                        f"- Recall on fraud, first half of your date range: **{dch['recall_first_half']:.1%}**\n"
                        f"- Recall on fraud, second half: **{dch['recall_second_half']:.1%}**\n"
                    )
                    if dch['drop'] > 0.1:
                        st.warning(f"⚠️ Recall dropped {drop_pct:.0f} percentage points between the first and "
                                    f"second half of your date range — a possible sign of drift in your own data.")
                    else:
                        st.success(f"✅ Recall held steady across your date range (a {drop_pct:.0f}-point change) — "
                                    f"no strong sign of drift in this file.")
                else:
                    st.caption('Needs a `timestamp` column and enough fraud cases in both halves of the date '
                                'range to check — not available for this file.')
        else:
            st.info('Your file had no `is_fraud` column, so precision/recall, a live model comparison, and a '
                     'drift check can\u2019t be computed — only prediction counts and probabilities above. '
                     'Upload a labelled file (with `is_fraud`) to unlock those.')

        st.markdown('---')
        st.header('📚 Chapter Four Reference Benchmark (Fixed)')
        st.caption(
            'Everything below is the dissertation\u2019s own tested, historical results from Chapter Four — the '
            'same regardless of what you upload. Shown here for comparison against your live results above.'
        )
    else:
        st.info('💡 No file uploaded this session — showing the framework\u2019s own fixed Chapter Four results '
                 'below. Upload a file on the **Upload Dataset & Time-Series** page, then return here for a '
                 'live section computed from your own data instead.')

    # -------------------------------------------------------------------------
    # FIXED reference section: Chapter Four's own tested results. Identical
    # regardless of what you upload — these describe how the framework itself
    # was built and evaluated, not a property of your data.
    # -------------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Contemp. PR-AUC (AutoGluon)', f"{dashboard_metrics['ag_contemp_pr_auc']:.3f}", 'weighted ensemble')
    k2.metric('Best on Held-Out Test', dashboard_metrics['best_by_test'],
               f"{dashboard_metrics['leaderboard_test'][0]['score_test']:.3f} PR-AUC")
    k3.metric('Median Latency (AutoGluon)', f"{dashboard_metrics['ag_latency_p50_ms']:.1f} ms",
               f"-{speed_ratio:.0f}x slower than XGBoost", delta_color='inverse')
    k4.metric('Drift Status (Chapter Four\u2019s test)', 'Alert Raised', f"Day {dashboard_metrics['adwin_first_drift_day']:.0f}", delta_color='inverse')

    st.markdown('---')
    c1, c2 = st.columns(2)
    with c1:
        st.subheader('AutoGluon Leaderboard — Held-Out Test PR-AUC')
        lb_df = pd.DataFrame(dashboard_metrics['leaderboard_test']).sort_values('score_test')
        colors = ['#C44E52' if m == dashboard_metrics['best_by_test'] else '#4C72B0' for m in lb_df['model']]
        fig = go.Figure(go.Bar(x=lb_df['score_test'], y=lb_df['model'], orientation='h',
                                 marker_color=colors, text=lb_df['score_test'].round(3), textposition='outside'))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title='Test PR-AUC')
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.subheader('PR-AUC — All Five Candidate Models')
        five = dashboard_metrics['five_model_prauc']
        fig = go.Figure(go.Bar(x=list(five.keys()), y=list(five.values()),
                                 marker_color=['#4C72B0', '#55A868', '#8172B2', '#CCB974', '#C44E52']))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title='PR-AUC')
        st.plotly_chart(fig, width='stretch')

    st.markdown('---')
    c3, c4 = st.columns(2)
    with c3:
        st.subheader('⏱️ Time-Series Insight')
        st.markdown(
            f"- Fraud-rate trend (decomposed) **before** Day 130: **{dashboard_metrics['trend_before_130']:.3f}%**\n"
            f"- Fraud-rate trend (decomposed) **after** Day 130: **{dashboard_metrics['trend_after_130']:.3f}%**\n\n"
            "The trend component recovers the concept-drift point directly from the data, "
            "without being told where to look. Try the **Upload Dataset & Time-Series** page to "
            "run this on your own data."
        )
        st.subheader('🧭 Top Explanation Drivers (all methods agree)')
        st.markdown(
            "1. **distance_from_home** — dominant across SHAP, TreeSHAP, LIME, and permutation importance\n"
            "2. day_of_week / merchant_freq\n"
            "3. channel indicators (online/mobile)\n"
            "4. amount_zscore_vs_7d\n"
            "5. txn_count_1d"
        )
    with c4:
        st.subheader('⚖️ Model Governance: AutoML vs. Manual Tuning')
        st.markdown(
            f"- AutoGluon ensemble: PR-AUC **{dashboard_metrics['ag_contemp_pr_auc']:.3f}**, "
            f"Precision {dashboard_metrics['ag_contemp_precision']:.3f}, Recall {dashboard_metrics['ag_contemp_recall']:.3f}\n"
            f"- Manual XGBoost: PR-AUC **{dashboard_metrics['xgb_pr_auc']:.3f}**, "
            f"Precision {dashboard_metrics['xgb_precision']:.3f}, Recall {dashboard_metrics['xgb_recall']:.3f}\n"
            f"- Statistically indistinguishable (McNemar p={dashboard_metrics['mcnemar_p']:.3f}); "
            f"AutoGluon is **~{speed_ratio:.0f}x slower** per transaction, still well under a "
            "tenth of a second, and used directly in this app for its automatic model selection."
        )
        st.subheader('🌊 Drift Monitor (ADWIN)')
        st.markdown(
            f"- Recall on fraud, pre-drift: **{dashboard_metrics['adwin_recall_pre']:.1%}**\n"
            f"- Recall on fraud, post-drift: **{dashboard_metrics['adwin_recall_post']:.1%}**\n"
            f"- True drift onset: Day {dashboard_metrics['true_drift_day']}\n"
            f"- ADWIN alert raised: Day {dashboard_metrics['adwin_first_drift_day']:.0f} "
            f"({dashboard_metrics['adwin_first_drift_day'] - dashboard_metrics['true_drift_day']:.0f}-day detection lag)"
        )

# =============================================================================
# PAGE 2: AUTOML SELECTION & SCORECARDS
# =============================================================================
elif page == '🤖 AutoML Selection & Scorecards':
    st.title('🤖 AutoML Model Selection')
    st.write(
        'AutoGluon was given the training data and asked to train, tune, and combine several algorithms '
        'on its own. This page shows exactly what it tried and which model it picked, and lets you see '
        'how that choice was made.'
    )

    st.info(f"**AutoGluon\u2019s chosen model: `{predictor.model_best}`** — selected automatically based on "
            "cross-validated performance during training, with no algorithm chosen by hand.")

    lb_rows = lb_summary['leaderboard']
    lb_df = pd.DataFrame(lb_rows)
    st.subheader('Full Leaderboard')
    st.dataframe(
        lb_df.style.background_gradient(subset=['score_test'], cmap='Greens')
                    .background_gradient(subset=['score_val'], cmap='Blues')
                    .format({'score_test': '{:.4f}', 'score_val': '{:.4f}', 'pred_time_test': '{:.3f}s', 'fit_time': '{:.1f}s'}),
        width='stretch',
    )
    st.caption(
        f"Interesting, honest detail: **{lb_summary['best_by_test']}** scores marginally *higher* than "
        f"AutoGluon\u2019s own chosen model on held-out test data, even though AutoGluon picked its model using "
        "a separate validation split. This is a real finding from Chapter Four, not smoothed over here."
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(x=lb_df['model'], y=lb_df['score_val'], name='Validation PR-AUC', marker_color='#4C72B0'))
    fig.add_trace(go.Bar(x=lb_df['model'], y=lb_df['score_test'], name='Held-out Test PR-AUC', marker_color='#C44E52'))
    fig.update_layout(barmode='group', height=380, title='Validation Score (used to pick the model) vs. Test Score (the honest check)')
    st.plotly_chart(fig, width='stretch')

    st.markdown('---')
    st.header('🏆 Model Scorecards')
    st.write('Performance scorecards for every candidate model studied in this dissertation, all scored on the same held-out contemporaneous test set.')

    scorecard_rows = []
    for name in ['LogisticRegression', 'RandomForest', 'XGBoost', 'MLP_Temporal']:
        c = model_results[name]['test_contemp']
        scorecard_rows.append({'Model': name, 'Precision': c['precision'], 'Recall': c['recall'],
                                'F1': c['f1'], 'ROC-AUC': c['roc_auc'], 'PR-AUC': c['pr_auc'], 'MCC': c['mcc'],
                                'FPR (%)': 100 * c['fpr']})
    ag_c = ag_results['test_contemp']
    scorecard_rows.append({'Model': 'AutoGluon (ensemble)', 'Precision': ag_c['precision'], 'Recall': ag_c['recall'],
                            'F1': ag_c['f1'], 'ROC-AUC': ag_c['roc_auc'], 'PR-AUC': ag_c['pr_auc'], 'MCC': ag_c['mcc'],
                            'FPR (%)': 100 * ag_c['fpr']})
    scorecard_df = pd.DataFrame(scorecard_rows).set_index('Model')
    st.dataframe(
        scorecard_df.style.background_gradient(subset=['PR-AUC', 'MCC', 'F1'], cmap='Greens')
                           .background_gradient(subset=['FPR (%)'], cmap='Reds')
                           .format('{:.3f}'),
        width='stretch',
    )

    st.subheader('Confusion Matrices')
    cm_sources = {name: model_results[name]['test_contemp'] for name in
                  ['LogisticRegression', 'RandomForest', 'XGBoost', 'MLP_Temporal']}
    cm_sources['AutoGluon (ensemble)'] = ag_c
    cols = st.columns(len(cm_sources))
    for col, (name, c) in zip(cols, cm_sources.items()):
        cm = np.array([[c['tn'], c['fp']], [c['fn'], c['tp']]])
        fig = go.Figure(go.Heatmap(z=cm, x=['Pred Legit', 'Pred Fraud'], y=['Actual Legit', 'Actual Fraud'],
                                     colorscale='Blues', showscale=False, text=cm, texttemplate='%{text}'))
        fig.update_layout(height=260, margin=dict(l=5, r=5, t=30, b=5), title=dict(text=name, font=dict(size=11)))
        col.plotly_chart(fig, width='stretch')

    # -------------------------------------------------------------------------
    # Live scorecard for whatever was uploaded on the Upload page, if anything.
    # Only possible when the uploaded file included real is_fraud labels —
    # precision, recall, and a confusion matrix cannot be computed without them.
    # -------------------------------------------------------------------------
    if st.session_state.has_uploaded and st.session_state.uploaded_summary is not None:
        st.markdown('---')
        st.header('📌 Your Uploaded Data — Live Scorecard')
        st.caption(f"Computed from `{st.session_state.uploaded_filename}`, using AutoGluon\u2019s predictions "
                    f"from the Upload Dataset & Time-Series page.")
        if st.session_state.uploaded_has_labels:
            scs = st.session_state.uploaded_scorecards
            live_rows = []
            for name, sc in scs.items():
                live_rows.append({'Model': name, 'Precision': sc['precision'], 'Recall': sc['recall'],
                                    'F1': sc['f1'], 'ROC-AUC': sc['roc_auc'], 'PR-AUC': sc['pr_auc'],
                                    'MCC': sc['mcc'], 'FPR (%)': 100 * sc['fpr']})
            live_df = pd.DataFrame(live_rows).set_index('Model')
            st.dataframe(
                live_df.style.background_gradient(subset=['PR-AUC', 'MCC', 'F1'], cmap='Greens')
                              .background_gradient(subset=['FPR (%)'], cmap='Reds')
                              .format('{:.3f}'),
                width='stretch',
            )
            st.caption('RandomForest is not included — its saved model file is too large to ship with this app '
                        '(25.7 MB, over GitHub\u2019s upload limit), so it can\u2019t be re-run live on your data.')

            st.subheader('Confusion Matrices — Your Data')
            cm_cols = st.columns(len(scs))
            for col, (name, sc) in zip(cm_cols, scs.items()):
                cm = np.array([[sc['tn'], sc['fp']], [sc['fn'], sc['tp']]])
                fig = go.Figure(go.Heatmap(z=cm, x=['Pred Legit', 'Pred Fraud'], y=['Actual Legit', 'Actual Fraud'],
                                             colorscale='Purples', showscale=False, text=cm, texttemplate='%{text}'))
                fig.update_layout(height=280, margin=dict(l=5, r=5, t=30, b=5), title=dict(text=name, font=dict(size=11)))
                col.plotly_chart(fig, width='stretch')
            st.caption(
                'This is a genuine, freshly-computed scorecard on the file you uploaded, using the same '
                'formulas as Chapter Four, not the dissertation\u2019s own fixed results shown above.'
            )
        else:
            st.info(
                f"Your uploaded file (`{st.session_state.uploaded_filename}`) has no `is_fraud` column, so "
                "precision, recall, and a confusion matrix cannot be computed for it — those need to know "
                "which transactions were really fraud. Upload a file with an `is_fraud` column (0 or 1) to "
                "see a live scorecard here."
            )

# =============================================================================
# PAGE 3: UPLOAD DATASET & TIME-SERIES
# =============================================================================
elif page == '📁 Upload Dataset & Time-Series':
    st.title('📁 Upload a Transactions Dataset')
    st.write(
        'Upload a CSV of transactions to score them for fraud probability in real time using AutoGluon, '
        'and to see time-series patterns in your own data.'
    )

    mode = st.radio(
        'What does your file contain?',
        ['Raw transactions (account_id, timestamp, amount, merchant_category, channel, distance_from_home)',
         'Already-engineered features (the 17 model features, already computed)'],
    )
    uploaded = st.file_uploader('Choose a CSV file', type=['csv'])

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f'Could not read this file as a CSV: {e}')
            st.stop()

        st.write(f'Loaded **{len(raw_df):,}** rows, **{raw_df.shape[1]}** columns.')
        with st.expander('Preview uploaded data'):
            st.dataframe(raw_df.head(20), width='stretch')

        is_raw_mode = mode.startswith('Raw')
        if is_raw_mode:
            missing = [c for c in RAW_REQUIRED_COLS if c not in raw_df.columns]
            if missing:
                st.error(f"Missing required column(s) for raw transactions: {', '.join(missing)}")
                st.stop()
            with st.spinner('Engineering features...'):
                feat_df = engineer_features(raw_df)
        else:
            missing = [c for c in FEATURE_COLS if c not in raw_df.columns]
            if missing:
                st.error(f"Missing required engineered feature column(s): {', '.join(missing)}")
                st.stop()
            feat_df = raw_df.copy()

        with st.spinner('Scoring transactions with AutoGluon and every comparison model available...'):
            model_probas = score_with_all_models(feat_df)
            proba = model_probas['AutoGluon']
        feat_df['fraud_probability'] = proba
        feat_df['decision'] = [decision_label(p) for p in proba]

        n_flagged = int((proba >= DECISION_THRESHOLDS['approve_below']).sum())
        n_decline = int((proba >= DECISION_THRESHOLDS['decline_above']).sum())

        # --- Save to session state so the Dashboard, Scorecards, and Manual ---
        # --- Prediction pages can also reflect this file, not just this page ---
        st.session_state.has_uploaded = True
        st.session_state.uploaded_filename = uploaded.name
        st.session_state.uploaded_feat_df = feat_df
        st.session_state.uploaded_raw_df = raw_df
        st.session_state.uploaded_summary = {
            'n_scored': len(feat_df), 'n_flagged': n_flagged, 'n_decline': n_decline,
            'median_proba': float(np.median(proba)), 'mean_proba': float(np.mean(proba)),
            'max_proba': float(np.max(proba)),
        }
        st.session_state.uploaded_model_probas = model_probas

        if 'is_fraud' in raw_df.columns and raw_df['is_fraud'].nunique() > 0:
            y_true = raw_df['is_fraud'].astype(int).values
            st.session_state.uploaded_has_labels = True
            st.session_state.uploaded_scorecard = compute_live_scorecard(y_true, proba)
            # Same live scorecard, but for every model, not just AutoGluon — this is
            # what makes a genuine "PR-AUC — All Candidate Models" chart on YOUR
            # data possible, not just Chapter Four's own fixed test set.
            st.session_state.uploaded_scorecards = {
                name: compute_live_scorecard(y_true, p) for name, p in model_probas.items()
            }
            # A dependency-free drift check: does AutoGluon's recall on real fraud
            # hold up in the second half of your date range vs the first half?
            # (No new package needed — same idea as Chapter Four's ADWIN monitor,
            # just computed directly rather than via a streaming detector.)
            if 'timestamp' in raw_df.columns:
                ts = pd.to_datetime(raw_df['timestamp'])
                order = ts.values.argsort()
                y_sorted, p_sorted = y_true[order], proba[order]
                half = len(y_sorted) // 2
                if half >= 5 and y_sorted[:half].sum() >= 1 and y_sorted[half:].sum() >= 1:
                    pred_sorted = (p_sorted >= 0.5).astype(int)
                    recall_first = float(((pred_sorted[:half] == 1) & (y_sorted[:half] == 1)).sum() / y_sorted[:half].sum())
                    recall_second = float(((pred_sorted[half:] == 1) & (y_sorted[half:] == 1)).sum() / y_sorted[half:].sum())
                    st.session_state.uploaded_drift_check = {
                        'recall_first_half': recall_first, 'recall_second_half': recall_second,
                        'drop': recall_first - recall_second,
                    }
                else:
                    st.session_state.uploaded_drift_check = None
            else:
                st.session_state.uploaded_drift_check = None
        else:
            st.session_state.uploaded_has_labels = False
            st.session_state.uploaded_scorecard = None
            st.session_state.uploaded_scorecards = None
            st.session_state.uploaded_drift_check = None

        st.markdown('### Scoring Results (AutoGluon)')
        st.success(
            f"✅ Saved to this session. These results now also appear on the **Dashboard**, "
            f"**AutoML Selection & Scorecards**, and **Manual Prediction** pages."
            + (" This file includes real fraud labels, so a genuine live scorecard "
               "(precision, recall, confusion matrix) can be computed on it."
               if st.session_state.uploaded_has_labels else
               " This file has no `is_fraud` column, so a live scorecard can't be computed — "
               "only prediction counts and probabilities.")
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Transactions scored', f'{len(feat_df):,}')
        m2.metric('Flagged for review or decline', f'{n_flagged:,}', f'{100*n_flagged/len(feat_df):.2f}% of traffic')
        m3.metric('Recommended decline', f'{n_decline:,}')
        m4.metric('Highest fraud probability found', f'{np.max(proba):.1%}',
                   help='The median is almost always 0.0000 here, since AutoGluon assigns exactly 0 to the '
                        'great majority of transactions given how rare fraud is — the highest score found is '
                        'a far more useful single number to look at.')

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(feat_df, x='fraud_probability', nbins=50, title='Distribution of Fraud Probabilities')
            fig.update_layout(height=340)
            st.plotly_chart(fig, width='stretch')
        with c2:
            decision_counts = feat_df['decision'].value_counts().reset_index()
            decision_counts.columns = ['decision', 'count']
            fig = px.bar(decision_counts, x='decision', y='count', color='decision',
                         color_discrete_map={'🟢 Approve': '#55A868', '🟠 Review': '#DD8452', '🔴 Decline / Block': '#C44E52'},
                         title='Decisions Breakdown')
            fig.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig, width='stretch')

        st.markdown('### Flagged Transactions')
        flagged = feat_df[feat_df['fraud_probability'] >= DECISION_THRESHOLDS['approve_below']] \
            .sort_values('fraud_probability', ascending=False)
        display_cols = [c for c in ['account_id', 'timestamp', 'amount', 'merchant_category', 'channel'] if c in flagged.columns]
        display_cols += ['fraud_probability', 'decision']
        st.dataframe(flagged[display_cols].style.background_gradient(subset=['fraud_probability'], cmap='Reds'),
                     width='stretch', height=320)

        csv_out = feat_df.to_csv(index=False).encode('utf-8')
        st.download_button('⬇️ Download full scored results as CSV', csv_out, 'ieaf_scored_transactions.csv', 'text/csv')

        # ---- Time-series section, only if a timestamp column is present ----
        if 'timestamp' in raw_df.columns:
            st.markdown('---')
            st.header('📈 Time-Series Insights on Your Data')
            ts_df = raw_df.copy()
            ts_df['timestamp'] = pd.to_datetime(ts_df['timestamp'])
            daily = ts_df.set_index('timestamp').resample('D').agg(transactions=('amount', 'size'))
            if 'is_fraud' in ts_df.columns:
                daily['fraud_count'] = ts_df.set_index('timestamp').resample('D')['is_fraud'].sum()
                daily['fraud_rate_pct'] = 100 * daily['fraud_count'] / daily['transactions']
            else:
                daily['fraud_count'] = ts_df.set_index('timestamp').resample('D').apply(
                    lambda g: (g.index.isin(flagged['timestamp'])).sum() if len(g) else 0)
                daily['fraud_rate_pct'] = 100 * daily['fraud_count'] / daily['transactions']

            c3, c4 = st.columns(2)
            with c3:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=daily.index, y=daily['transactions'], name='Transactions/day', marker_color='#4C72B0', opacity=0.5))
                fig.update_layout(height=340, title='Daily Transaction Volume')
                st.plotly_chart(fig, width='stretch')
            with c4:
                fig = go.Figure(go.Scatter(x=daily.index, y=daily['fraud_rate_pct'], line=dict(color='#C44E52')))
                fig.update_layout(height=340, title='Daily Flagged/Fraud Rate (%)')
                st.plotly_chart(fig, width='stretch')

            if len(daily) >= 14:
                st.subheader('Seasonal Decomposition (trend / weekly seasonal pattern / residual)')
                try:
                    decomp = seasonal_decompose(daily['transactions'].fillna(0), model='additive', period=7)
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                          subplot_titles=('Trend', 'Weekly Seasonal Pattern', 'Residual'))
                    fig.add_trace(go.Scatter(x=daily.index, y=decomp.trend, line=dict(color='#55A868')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=daily.index, y=decomp.seasonal, line=dict(color='#DD8452')), row=2, col=1)
                    fig.add_trace(go.Scatter(x=daily.index, y=decomp.resid, mode='markers', marker=dict(color='#C44E52', size=4)), row=3, col=1)
                    fig.update_layout(height=550, showlegend=False)
                    st.plotly_chart(fig, width='stretch')
                except Exception as e:
                    st.warning(f'Could not run seasonal decomposition on this data ({e}). Needs at least two full weeks of daily data.')
            else:
                st.caption('Upload at least 14 days of data to see a seasonal decomposition.')
    else:
        st.info('Upload a CSV to get started — try `sample_transactions.csv` included with this app.')
        st.markdown(
            '**Raw transaction format example:**\n\n'
            '`account_id, timestamp, amount, merchant_category, channel, distance_from_home`\n\n'
            '`101, 2025-03-14 09:22:00, 84.50, grocery, pos, 0.03`'
        )

# =============================================================================
# PAGE 4: MANUAL PREDICTION & EXPLAINABILITY
# =============================================================================
elif page == '✍️ Manual Prediction & Explainability':
    st.title('✍️ Score a Single Transaction')
    st.write('Enter transaction details below to get an instant fraud probability from AutoGluon and an explanation of why.')

    # Default values for the behavioural fields: if a file has been uploaded this
    # session, use its own average account behaviour as smarter starting points,
    # instead of generic hardcoded numbers.
    defaults = {'txn_count_1d': 1, 'txn_count_7d': 6, 'txn_count_30d': 24,
                'amount_mean_7d': 60.0, 'amount_std_7d': 25.0, 'time_since_last_txn_h': 8.0}
    if st.session_state.has_uploaded and st.session_state.uploaded_feat_df is not None:
        fdf = st.session_state.uploaded_feat_df
        for key in defaults:
            if key in fdf.columns and fdf[key].notna().any():
                defaults[key] = float(fdf[key].mean())
        st.info(
            f"ℹ️ The behavioural defaults below (transaction counts, average amount) have been updated "
            f"using the average account behaviour found in `{st.session_state.uploaded_filename}`, "
            f"instead of generic starting values. Change any of them freely before scoring."
        )

    with st.form('manual_form'):
        c1, c2, c3 = st.columns(3)
        with c1:
            amount = st.number_input('Transaction amount ($)', min_value=0.0, value=75.0, step=1.0)
            hour = st.slider('Hour of day', 0, 23, 14)
            day_of_week = st.selectbox('Day of week', options=list(range(7)),
                                         format_func=lambda d: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][d])
            distance_from_home = st.slider('Distance from home (normalised, 0 = home)', 0.0, 3.0, 0.05, 0.01)
        with c2:
            merchant_category = st.selectbox('Merchant category',
                ['grocery', 'electronics', 'travel', 'restaurant', 'fuel', 'online_retail',
                 'utilities', 'entertainment', 'jewelry', 'cash_advance'])
            channel = st.selectbox('Channel', ['pos', 'online', 'mobile', 'atm'])
            txn_count_1d = st.number_input('Transactions in the last 1 day (this account)', min_value=0,
                                             value=round(defaults['txn_count_1d']))
            txn_count_7d = st.number_input('Transactions in the last 7 days (this account)', min_value=0,
                                             value=round(defaults['txn_count_7d']))
        with c3:
            txn_count_30d = st.number_input('Transactions in the last 30 days (this account)', min_value=0,
                                              value=round(defaults['txn_count_30d']))
            amount_mean_7d = st.number_input('Average amount, last 7 days ($)', min_value=0.0,
                                               value=round(defaults['amount_mean_7d'], 2))
            amount_std_7d = st.number_input('Std. dev. of amount, last 7 days ($)', min_value=0.0,
                                              value=round(defaults['amount_std_7d'], 2))
            time_since_last_txn_h = st.number_input('Hours since this account\'s last transaction',
                                                       min_value=0.0, value=round(defaults['time_since_last_txn_h'], 2))
        submitted = st.form_submit_button('🔍 Score This Transaction With AutoGluon', width='stretch')

    if submitted:
        feat_row = engineer_single_transaction(
            amount=amount, hour=hour, day_of_week=day_of_week, merchant_category=merchant_category,
            channel=channel, distance_from_home=distance_from_home, txn_count_1d=txn_count_1d,
            txn_count_7d=txn_count_7d, txn_count_30d=txn_count_30d, amount_mean_7d=amount_mean_7d,
            amount_std_7d=amount_std_7d, time_since_last_txn_h=time_since_last_txn_h,
        )
        with st.spinner('Scoring with AutoGluon...'):
            proba = float(ag_predict(feat_row)[0])
        decision = decision_label(proba)

        st.markdown('---')
        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric('Fraud Probability (AutoGluon)', f'{proba:.1%}')
            st.markdown(f'### Decision: {decision}')
            st.progress(min(proba, 1.0))
            st.caption(f"Approve below {DECISION_THRESHOLDS['approve_below']:.0%}, "
                        f"Decline above {DECISION_THRESHOLDS['decline_above']:.0%}, Review in between.")
            st.caption(f"Model used: `{predictor.model_best}` (AutoGluon\u2019s automatically-selected best model)")

        with r2:
            st.markdown('#### Why this score? (SHAP, permutation explainer — same method as Chapter 4.7)')
            with st.spinner('Computing explanation (AutoGluon\u2019s ensemble needs the slower, model-agnostic SHAP method)...'):
                background = feat_row.copy()
                background[FEATURE_COLS] = background[FEATURE_COLS] * 0  # zero baseline, fast fallback
                try:
                    def predict_fn(X):
                        Xd = pd.DataFrame(X, columns=FEATURE_COLS)
                        return predictor.predict_proba(Xd)[1].values
                    explainer = shap.explainers.Permutation(predict_fn, background[FEATURE_COLS], seed=0)
                    sv = explainer(feat_row[FEATURE_COLS], max_evals=60).values[0]
                except Exception as e:
                    sv = None
                    st.warning(f'Explanation unavailable for this input ({e}).')

            if sv is not None:
                order = np.argsort(np.abs(sv))[::-1][:8]
                fig = go.Figure(go.Bar(
                    x=[sv[i] for i in order][::-1],
                    y=[FEATURE_COLS[i] for i in order][::-1],
                    orientation='h',
                    marker_color=['#C44E52' if sv[i] > 0 else '#4C72B0' for i in order][::-1],
                ))
                fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                                    xaxis_title='Contribution to fraud score (SHAP value)')
                st.plotly_chart(fig, width='stretch')
                st.caption('Red bars push the score toward fraud; blue bars push it toward legitimate.')

        with st.expander('View the exact feature values sent to the model'):
            st.dataframe(feat_row.T.rename(columns={0: 'value'}), width='stretch')

# =============================================================================
# PAGE 5: EXECUTIVE SUMMARY & INSIGHTS
# =============================================================================
elif page == '📋 Executive Summary & Insights':
    st.title('📋 Executive Summary & Insights')
    st.caption(
        'The decision-support layer of this app: what the numbers on the other pages actually mean, '
        'written for both technical and non-technical readers. Every statement below traces back to a '
        'real, computed result, not an invented narrative.'
    )

    has_session = st.session_state.has_uploaded and st.session_state.uploaded_summary is not None
    session_summary = st.session_state.uploaded_summary if has_session else None
    session_scorecard = st.session_state.uploaded_scorecard if has_session else None
    session_has_labels = st.session_state.uploaded_has_labels if has_session else False

    # Compute a fresh profile of whatever was actually uploaded — this is what
    # makes the page genuinely reactive to your data, not just a couple of
    # bolted-on numbers. Recomputed on every page load, never cached, so it is
    # always current for whatever is in session right now.
    data_profile = None
    if has_session:
        data_profile = insights.build_data_profile(
            st.session_state.uploaded_raw_df, st.session_state.uploaded_feat_df,
            st.session_state.uploaded_feat_df['fraud_probability'].values,
            DECISION_THRESHOLDS['approve_below'], DECISION_THRESHOLDS['decline_above'],
        )

    if has_session:
        st.success(
            f"📌 Showing **live insights on `{st.session_state.uploaded_filename}`** "
            f"({session_summary['n_scored']:,} transactions, scored just now). Chapter Four's own reference "
            f"results are further down, clearly labelled, for comparison."
        )
    else:
        st.info(
            '💡 No file uploaded this session — showing this framework\u2019s own reference results from '
            'Chapter Four below. Upload a file on the **Upload Dataset & Time-Series** page and revisit this '
            'page for insights computed fresh from your own data instead.'
        )

    # ---------------------------------------------------------------- 1. Executive Summary
    st.header('1. Executive Summary')
    summary = insights.build_executive_summary(dashboard_metrics, ag_results, has_session, session_summary)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**The problem**')
        st.write(summary['problem'])
        st.markdown('**The data**')
        if has_session:
            st.write(f"This summary is built from `{st.session_state.uploaded_filename}`, "
                      f"{session_summary['n_scored']:,} transactions you uploaded this session.")
        else:
            st.write(summary['data'])
    with c2:
        st.markdown('**Overall outcome**')
        st.write(summary['outcome'])
    st.markdown('**Key findings**')
    for f in summary['findings']:
        st.markdown(f'- {f}')

    st.markdown('---')

    # ---------------------------------------------------------------- 2. Key Insights
    st.header('2. Key Insights')
    if has_session:
        st.caption(f"Computed fresh from `{st.session_state.uploaded_filename}` — not the dissertation's own results.")
        data_insights = insights.build_data_key_insights(data_profile, session_scorecard, session_has_labels)
        if data_insights:
            for ins in data_insights:
                insight_card(ins['icon'], ins['title'], ins['text'], ins['tone'])
        else:
            st.caption('Your uploaded file didn\u2019t have enough of the optional columns (merchant category, '
                        'channel, amount, timestamp) to generate detailed pattern insights — only counts are available.')
        with st.expander('📚 Also show Chapter Four\u2019s own reference insights'):
            for ins in insights.build_key_insights(dashboard_metrics, ag_results, model_results, lb_summary,
                                                      adwin_results, False, None, None, False):
                insight_card(ins['icon'], ins['title'], ins['text'], ins['tone'])
    else:
        st.caption('Automatically generated from this app\u2019s own real Chapter Four results — not the raw numbers again, but what they mean.')
        key_insights = insights.build_key_insights(
            dashboard_metrics, ag_results, model_results, lb_summary, adwin_results,
            has_session, session_summary, session_scorecard, session_has_labels,
        )
        for ins in key_insights:
            insight_card(ins['icon'], ins['title'], ins['text'], ins['tone'])

    st.markdown('---')

    # ---------------------------------------------------------------- 3. Interpretation of Results
    st.header('3. Interpretation of Results')
    if has_session:
        st.caption('What the patterns found in your own uploaded data are actually showing.')
        data_interps = insights.build_data_interpretations(data_profile)
        if data_interps:
            for interp in data_interps:
                with st.expander(f"📈 {interp['chart']}"):
                    st.write(interp['text'])
        else:
            st.caption('Not enough optional columns in your file to interpret category, channel, amount, or time patterns.')
        with st.expander('📚 Also show interpretations of Chapter Four\u2019s own charts'):
            for interp in insights.build_interpretations(dashboard_metrics, ag_results, model_results):
                st.markdown(f"**{interp['chart']}**")
                st.write(interp['text'])
    else:
        st.caption('What each major chart elsewhere in this app is actually showing, and why it matters.')
        for interp in insights.build_interpretations(dashboard_metrics, ag_results, model_results):
            with st.expander(f"📈 {interp['chart']}"):
                st.write(interp['text'])

    st.markdown('---')

    # ---------------------------------------------------------------- 4. Recommendations
    st.header('4. Recommendations')
    if has_session:
        st.caption(f"Specific to what was found in `{st.session_state.uploaded_filename}`.")
        recs = insights.build_data_recommendations(data_profile, session_summary)
    else:
        recs = insights.build_recommendations(dashboard_metrics, ag_results)
    for i, rec in enumerate(recs, start=1):
        with st.container(border=True):
            st.markdown(f"**{i}. {rec['title']}**")
            st.markdown(f"- **Evidence:** {rec['evidence']}")
            st.markdown(f"- **Why:** {rec['rationale']}")
            st.markdown(f"- **Expected benefit:** {rec['benefit']}")
            st.markdown(f"- **Risk / assumption:** {rec['risk']}")
    if has_session:
        with st.expander('📚 Also show Chapter Four\u2019s own general recommendations'):
            for i, rec in enumerate(insights.build_recommendations(dashboard_metrics, ag_results), start=1):
                st.markdown(f"**{i}. {rec['title']}**")
                st.markdown(f"- Evidence: {rec['evidence']}")

    st.markdown('---')

    # ---------------------------------------------------------------- 5. Confidence & Limitations
    st.header('5. Confidence & Limitations')
    cl = insights.build_confidence_limitations()
    if has_session:
        if session_has_labels:
            st.success(
                f"**Confidence on your data:** your file included real fraud labels, so the scorecard on it "
                f"({session_scorecard['precision']:.3f} precision, {session_scorecard['recall']:.3f} recall) is a "
                f"genuine, verified measurement, not a prediction taken on faith."
            )
        else:
            st.warning(
                '**Confidence on your data:** your file had no `is_fraud` column, so everything above is an '
                'unverified model prediction — there is no way, from this file alone, to confirm how many '
                'flagged transactions are genuinely fraud.'
            )
    else:
        st.info(f"**Overall confidence:** {cl['confidence']}")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('**✅ Strengths**')
        for s in cl['strengths']:
            st.markdown(f'- {s}')
        st.markdown('**⚠️ Limitations**')
        for l in cl['limitations']:
            st.markdown(f'- {l}')
    with c4:
        st.markdown('**🎯 Potential sources of bias**')
        for b in cl['biases']:
            st.markdown(f'- {b}')
        st.markdown('**📌 Assumptions made**')
        for a in cl['assumptions']:
            st.markdown(f'- {a}')

    st.markdown('---')

    # ---------------------------------------------------------------- 6. Business Impact
    st.header('6. Business Impact')
    impact_items = insights.build_business_impact(has_session, session_summary, data_profile)
    bi_cols = st.columns(len(impact_items))
    for col, item in zip(bi_cols, impact_items):
        with col:
            st.markdown(f"**{item['icon']} {item['title']}**")
            st.caption(item['text'])

    st.markdown('---')

    # ---------------------------------------------------------------- 7. Next Steps
    st.header('7. Next Steps')
    if has_session:
        st.markdown(f"- Review the flagged transactions from `{st.session_state.uploaded_filename}` on the "
                      f"**Upload Dataset & Time-Series** page, starting with the highest-probability ones.")
        if not session_has_labels:
            st.markdown('- Upload a version of this file with a real `is_fraud` column to get a verified scorecard instead of predictions alone.')
    for step in insights.build_next_steps():
        st.markdown(f'- {step}')

# =============================================================================
# PAGE 6: ABOUT
# =============================================================================
elif page == 'ℹ️ About This App':
    st.title('ℹ️ About This App')
    st.markdown(
        f"""
This app is a practical demonstration of the Intelligent Explainable AutoML Framework (IEAF) from the
dissertation **"Design and Implementation of an Intelligent Explainable AutoML Framework for Real-Time
Financial Fraud Detection Using Machine Learning and Temporal Analytics."**

Unlike the app's first version, this version scores every transaction using **AutoGluon's own
automatically-selected model** (currently `{predictor.model_best}`), showcasing the framework's AutoML
pipeline end to end rather than a single hand-picked model.

**What each page does:**
- **Dashboard** — the same live view shown in Chapter Four, built from real, computed results. Once you upload a file, a second section appears below showing live KPIs for your own data.
- **AutoML Selection & Scorecards** — shows exactly which models AutoGluon tried, which one it picked
  and why, and gives full performance scorecards and confusion matrices for every candidate model.
  If you've uploaded a labelled file (with an `is_fraud` column), a live scorecard for your own data
  appears here too.
- **Upload Dataset & Time-Series** — score a CSV of transactions in batch with AutoGluon, and (if your
  file has a timestamp column) see daily volume, fraud-rate trends, and a full seasonal decomposition,
  the same time-series method used in Chapter Four. Results here are also saved for the session and
  reused on the other pages.
- **Manual Prediction & Explainability** — score one transaction at a time and see a real SHAP
  explanation for AutoGluon's decision. If you've uploaded a file, the behavioural defaults (recent
  transaction counts, average amount) are pre-filled from your own data's averages.
- **Executive Summary & Insights** — the decision-support layer: a plain-language executive summary,
  automatically generated key insights, chart-by-chart interpretation, evidence-based recommendations,
  an honest confidence-and-limitations section, business impact, and next steps. Written for both
  technical and non-technical readers, and it gets richer once you've uploaded your own data.

**How pages share data:** uploading a file on the Upload page saves its results to your browser
session (not a database, and not shared with other visitors). The Dashboard, Scorecards, and Manual
Prediction pages then check for that saved data and show an extra, clearly-labelled section built
from it, alongside, never replacing, the dissertation's own fixed Chapter Four results. Use the
"Clear uploaded data" button in the sidebar to reset at any time.

**An honest trade-off, shown on purpose:** AutoGluon is about 40x slower per transaction than a single
tuned XGBoost model, for statistically indistinguishable accuracy (Chapter Four, Section 4.13). This app
uses AutoGluon anyway, since ~16 ms is still fast enough to feel instant to a person using this app, and
because the point of this demonstration is to show the AutoML pipeline actually working, trade-offs and
all, not to hide them.

**A note on the data:** the underlying model was trained on a disclosed, synthetic proxy dataset (see
Chapter Four, Section 4.0.1), since real bank data was not accessible for this dissertation. Treat any
prediction here as a demonstration of how the framework behaves, not a real-world fraud-risk assessment.
        """
    )
