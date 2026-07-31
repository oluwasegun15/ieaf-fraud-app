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

from feature_engineering import engineer_features, engineer_single_transaction, FEATURE_COLS, RAW_REQUIRED_COLS

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
    with open(ART / 'xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open(ART / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return xgb_model, scaler


@st.cache_data
def load_json(name):
    with open(ART / name) as f:
        return json.load(f)


predictor = load_autogluon()
xgb_model, scaler = load_other_models()
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
    'ℹ️ About This App',
])
st.sidebar.markdown('---')
st.sidebar.markdown(f"**Active model:** AutoGluon\n\n`{predictor.model_best}`")
st.sidebar.caption(
    'This app scores transactions using AutoGluon\u2019s own automatically-selected model, '
    'demonstrating the framework\u2019s AutoML pipeline end-to-end, exactly as built in Chapter Four.'
)

# =============================================================================
# PAGE 1: DASHBOARD
# =============================================================================
if page == '📊 Dashboard':
    st.title('IEAF Fraud Monitoring Dashboard — AutoGluon Edition v2')
    st.caption(f"Live view — Active model: AutoGluon ({dashboard_metrics['ag_best_model']})")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Contemp. PR-AUC (AutoGluon)', f"{dashboard_metrics['ag_contemp_pr_auc']:.3f}", 'weighted ensemble')
    k2.metric('Best on Held-Out Test', dashboard_metrics['best_by_test'],
               f"{dashboard_metrics['leaderboard_test'][0]['score_test']:.3f} PR-AUC")
    speed_ratio = dashboard_metrics['ag_latency_p50_ms'] / dashboard_metrics['xgb_latency_p50_ms']
    k3.metric('Median Latency (AutoGluon)', f"{dashboard_metrics['ag_latency_p50_ms']:.1f} ms",
               f"-{speed_ratio:.0f}x slower than XGBoost", delta_color='inverse')
    k4.metric('Drift Status', 'Alert Raised', f"Day {dashboard_metrics['adwin_first_drift_day']:.0f}", delta_color='inverse')

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

        with st.spinner('Scoring transactions with AutoGluon...'):
            proba = ag_predict(feat_df)
        feat_df['fraud_probability'] = proba
        feat_df['decision'] = [decision_label(p) for p in proba]

        n_flagged = int((proba >= DECISION_THRESHOLDS['approve_below']).sum())
        n_decline = int((proba >= DECISION_THRESHOLDS['decline_above']).sum())

        st.markdown('### Scoring Results (AutoGluon)')
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Transactions scored', f'{len(feat_df):,}')
        m2.metric('Flagged for review or decline', f'{n_flagged:,}', f'{100*n_flagged/len(feat_df):.2f}% of traffic')
        m3.metric('Recommended decline', f'{n_decline:,}')
        m4.metric('Median fraud probability', f'{np.median(proba):.4f}')

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
            txn_count_1d = st.number_input('Transactions in the last 1 day (this account)', min_value=0, value=1)
            txn_count_7d = st.number_input('Transactions in the last 7 days (this account)', min_value=0, value=6)
        with c3:
            txn_count_30d = st.number_input('Transactions in the last 30 days (this account)', min_value=0, value=24)
            amount_mean_7d = st.number_input('Average amount, last 7 days ($)', min_value=0.0, value=60.0)
            amount_std_7d = st.number_input('Std. dev. of amount, last 7 days ($)', min_value=0.0, value=25.0)
            time_since_last_txn_h = st.number_input('Hours since this account\'s last transaction',
                                                       min_value=0.0, value=8.0)
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
# PAGE 5: ABOUT
# =============================================================================
else:
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
- **Dashboard** — the same live view shown in Chapter Four, built from real, computed results.
- **AutoML Selection & Scorecards** — shows exactly which models AutoGluon tried, which one it picked
  and why, and gives full performance scorecards and confusion matrices for every candidate model.
- **Upload Dataset & Time-Series** — score a CSV of transactions in batch with AutoGluon, and (if your
  file has a timestamp column) see daily volume, fraud-rate trends, and a full seasonal decomposition,
  the same time-series method used in Chapter Four.
- **Manual Prediction & Explainability** — score one transaction at a time and see a real SHAP
  explanation for AutoGluon's decision.

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
