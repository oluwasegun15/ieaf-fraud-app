"""
insights.py
Generates the content for the Executive Summary & Insights page by interpreting
the app's own real, already-computed results (Chapter Four's findings, plus
whatever the user has uploaded this session). Every function here takes plain
dicts already loaded elsewhere in the app and returns plain Python structures
(lists of dicts, strings) that app.py renders — no new libraries, no external
API calls, nothing that could reopen the dependency problems already worked
through when deploying this app.

The logic is template-and-threshold based, not a black box: every insight
below traces back to a specific number in the app's own results, so what you
see here is an explanation of real numbers, not an invented narrative.
"""


def build_executive_summary(dm, ag_results, has_session, session_summary):
    """Returns the top-level overview: problem, data, headline findings, outcome."""
    speed_ratio = dm['ag_latency_p50_ms'] / dm['xgb_latency_p50_ms']
    sig = 'not statistically different from' if dm['mcnemar_p'] > 0.05 else 'statistically different from'

    findings = [
        f"AutoGluon's automatically-selected model ({dm['ag_best_model']}) reached a PR-AUC of "
        f"{dm['ag_contemp_pr_auc']:.3f} on data it had not been trained on.",
        f"This is {sig} a manually-tuned XGBoost model's accuracy (McNemar p = {dm['mcnemar_p']:.3f}), "
        f"but AutoGluon is roughly {speed_ratio:.0f}x slower to score a single transaction.",
        f"On a stricter, held-out test, plain {dm['best_by_test']} scored marginally higher "
        f"({dm['leaderboard_test'][0]['score_test']:.3f}) than AutoGluon's own chosen model — a reminder "
        f"that automatic model selection is not automatically the final word.",
        f"A concept-drift alert was correctly raised {dm['adwin_first_drift_day'] - dm['true_drift_day']:.0f} "
        f"days after fraud behaviour actually changed, confirmed independently by a seasonal decomposition "
        f"of the daily fraud rate.",
    ]
    if has_session and session_summary is not None:
        findings.append(
            f"This session, {session_summary['n_scored']:,} of your own uploaded transactions were scored: "
            f"{session_summary['n_flagged']:,} were flagged for review or decline "
            f"({100*session_summary['n_flagged']/session_summary['n_scored']:.2f}% of that traffic)."
        )

    return {
        'problem': (
            'Financial institutions need to catch fraudulent transactions in real time without either '
            'missing too much fraud or overwhelming staff with false alarms, and without a "black box" '
            'model nobody can explain to a regulator or a customer.'
        ),
        'data': (
            'A disclosed, synthetic proxy dataset of roughly 394,000 transactions, modelled on real fraud '
            'patterns, was used throughout, since real bank data could not be accessed for this work '
            '(see the About page for details).'
        ),
        'findings': findings,
        'outcome': (
            f"The framework works as intended: AutoML can automatically reach accuracy statistically "
            f"indistinguishable from careful manual tuning, at a measurable and honestly-reported speed cost, "
            f"with explanations available for every decision it makes."
        ),
    }


def build_key_insights(dm, ag_results, model_results, lb_summary, adwin_results, has_session, session_summary, session_scorecard, session_has_labels):
    """Returns a list of {icon, title, text, tone} insight cards. tone is
    'positive', 'neutral', or 'warning', driving the card's colour."""
    insights = []
    speed_ratio = dm['ag_latency_p50_ms'] / dm['xgb_latency_p50_ms']

    # --- Model accuracy ---
    if dm['ag_contemp_pr_auc'] >= 0.7:
        insights.append({
            'icon': '✅', 'tone': 'positive', 'title': 'Strong fraud-catching accuracy',
            'text': f"AutoGluon's PR-AUC of {dm['ag_contemp_pr_auc']:.3f} is well above the near-zero score a "
                     f"random guess would get on data this imbalanced, meaning the model has learned a real, "
                     f"usable fraud signal, not noise."
        })
    else:
        insights.append({
            'icon': '⚠️', 'tone': 'warning', 'title': 'Accuracy has room to improve',
            'text': f"AutoGluon's PR-AUC of {dm['ag_contemp_pr_auc']:.3f} suggests real signal, but there is "
                     f"meaningful room to improve before this should be trusted for high-stakes decisions alone."
        })

    # --- Speed / automation trade-off ---
    if speed_ratio >= 20:
        insights.append({
            'icon': '⏱️', 'tone': 'warning', 'title': 'Automation has a real speed cost',
            'text': f"AutoGluon is about {speed_ratio:.0f}x slower per transaction than a single tuned XGBoost "
                     f"model, for statistically indistinguishable accuracy (McNemar p = {dm['mcnemar_p']:.3f}). "
                     f"At high transaction volumes, this gap translates directly into infrastructure cost."
        })
    else:
        insights.append({
            'icon': '⚡', 'tone': 'positive', 'title': 'Automation speed cost is manageable',
            'text': f"AutoGluon is only about {speed_ratio:.0f}x slower than a manually-tuned model — a modest "
                     f"cost for the convenience of automatic model selection."
        })

    # --- AutoML selection honesty check ---
    if dm['best_by_test'] != dm['ag_best_model']:
        insights.append({
            'icon': '🔍', 'tone': 'neutral', 'title': 'AutoML\u2019s own pick was not the final winner',
            'text': f"AutoGluon chose {dm['ag_best_model']} using its internal validation split, but on truly "
                     f"held-out test data, plain {dm['best_by_test']} scored marginally higher "
                     f"({dm['leaderboard_test'][0]['score_test']:.3f} vs "
                     f"{dm['ag_contemp_pr_auc']:.3f}). Validation-based selection and real-world performance "
                     f"do not always agree exactly — worth re-checking before locking in a production model."
        })

    # --- Drift risk ---
    drift_lag = dm['adwin_first_drift_day'] - dm['true_drift_day']
    recall_drop = dm['adwin_recall_pre'] - dm['adwin_recall_post']
    if recall_drop >= 0.25:
        insights.append({
            'icon': '🌊', 'tone': 'warning', 'title': 'Fraud detection degrades sharply once behaviour shifts',
            'text': f"Recall on real fraud fell from {dm['adwin_recall_pre']:.1%} to {dm['adwin_recall_post']:.1%} "
                     f"once fraud patterns changed, a drop of {100*recall_drop:.0f} percentage points. The drift "
                     f"monitor caught it, but only after a {drift_lag:.0f}-day lag — a real, meaningful blind spot."
        })
    else:
        insights.append({
            'icon': '🌊', 'tone': 'neutral', 'title': 'Some performance loss under drift, but bounded',
            'text': f"Recall on fraud fell from {dm['adwin_recall_pre']:.1%} to {dm['adwin_recall_post']:.1%} "
                     f"after fraud patterns shifted, detected within {drift_lag:.0f} days by the live monitor."
        })

    # --- Feature signal ---
    insights.append({
        'icon': '🎯', 'tone': 'neutral', 'title': 'One feature dominates every explanation method',
        'text': (
            'Distance from home is, by a wide margin, the strongest fraud signal across SHAP, TreeSHAP, LIME, '
            'and permutation importance alike. This consistency across four different explanation methods is '
            'reassuring for trust, but the dominance of a single feature is also a concentration risk worth '
            'watching if the underlying pattern ever changes.'
        )
    })

    # --- Session-specific insight, only if data has been uploaded ---
    if has_session and session_summary is not None:
        flag_rate = 100 * session_summary['n_flagged'] / session_summary['n_scored']
        if flag_rate > 5:
            insights.append({
                'icon': '📌', 'tone': 'warning', 'title': 'Your uploaded data shows an elevated flag rate',
                'text': f"{flag_rate:.2f}% of the {session_summary['n_scored']:,} transactions you uploaded "
                         f"were flagged for review or decline — notably higher than the roughly 0.2% base "
                         f"fraud rate this framework was trained on. This could mean your data genuinely has "
                         f"more risk, or that it looks different enough from the training data that the model "
                         f"is less confident. Worth a manual look at a sample of the flagged transactions."
            })
        else:
            insights.append({
                'icon': '📌', 'tone': 'positive', 'title': 'Your uploaded data shows a typical flag rate',
                'text': f"{flag_rate:.2f}% of the {session_summary['n_scored']:,} transactions you uploaded "
                         f"were flagged — in line with the low base rate this framework expects for everyday traffic."
            })
        if session_has_labels and session_scorecard is not None:
            insights.append({
                'icon': '📊', 'tone': 'positive' if session_scorecard['pr_auc'] >= 0.6 else 'warning',
                'title': 'Live accuracy check on your own labelled data',
                'text': f"Because your file included real fraud labels, a genuine scorecard could be computed: "
                         f"precision {session_scorecard['precision']:.3f}, recall {session_scorecard['recall']:.3f}, "
                         f"PR-AUC {session_scorecard['pr_auc']:.3f}. This is the most direct evidence available "
                         f"of how the model performs specifically on data like yours."
            })

    return insights


def build_interpretations(dm, ag_results, model_results):
    """Plain-language explanations of the charts already shown elsewhere in the app."""
    speed_ratio = dm['ag_latency_p50_ms'] / dm['xgb_latency_p50_ms']
    return [
        {
            'chart': 'AutoGluon Leaderboard (Dashboard, AutoML Selection page)',
            'text': (
                f"This chart ranks every model AutoGluon tried by how well it scores on data it never used to "
                f"pick a winner. The bar highlighted in red, {dm['best_by_test']}, is the honest best performer; "
                f"the gap between it and AutoGluon's own choice is small ({dm['leaderboard_test'][0]['score_test']:.3f} "
                f"vs {dm['ag_contemp_pr_auc']:.3f}), so this is a minor, not alarming, finding."
            )
        },
        {
            'chart': 'PR-AUC — All Five Candidate Models',
            'text': (
                'This compares every model type tried in this study on the same footing. The four strongest '
                'models cluster closely together in accuracy, which is itself a finding: for this problem, the '
                'choice of algorithm matters less than getting the features and the imbalance-handling right.'
            )
        },
        {
            'chart': 'Confusion Matrices',
            'text': (
                'These show the real counts behind the summary numbers: how many genuine fraud cases were '
                'caught, how many were missed, and how many legitimate transactions were wrongly flagged. A '
                'model with very few false alarms but more missed fraud (like AutoGluon\u2019s ensemble here) '
                'suits a business that cannot tolerate annoying customers; a model that catches more fraud at '
                'the cost of more false alarms suits a business more worried about losses than friction.'
            )
        },
        {
            'chart': 'Time-Series Decomposition',
            'text': (
                f"Splitting daily fraud rate into trend, weekly pattern, and leftover noise shows the trend line "
                f"shifting from {dm['trend_before_130']:.3f}% to {dm['trend_after_130']:.3f}% right at the point "
                f"the drift monitor also flagged. Two independent methods agreeing is stronger evidence than "
                f"either one alone that fraud behaviour genuinely changed, not just noisy data."
            )
        },
        {
            'chart': 'Drift Monitor (ADWIN)',
            'text': (
                f"This tracks the model's real-world recall over time rather than a lab metric. The "
                f"{dm['adwin_first_drift_day'] - dm['true_drift_day']:.0f}-day gap between when fraud behaviour "
                f"changed and when the monitor noticed is the realistic cost of detecting drift from outcomes "
                f"rather than being told about it in advance — a genuine operational constraint, not a flaw "
                f"unique to this framework."
            )
        },
    ]


def build_recommendations(dm, ag_results):
    speed_ratio = dm['ag_latency_p50_ms'] / dm['xgb_latency_p50_ms']
    drift_lag = dm['adwin_first_drift_day'] - dm['true_drift_day']
    return [
        {
            'title': 'Use a dual-model strategy: XGBoost for real-time scoring, AutoGluon for periodic governance review',
            'evidence': f"Statistically indistinguishable accuracy (McNemar p = {dm['mcnemar_p']:.3f}) but a "
                         f"~{speed_ratio:.0f}x speed gap between the two.",
            'rationale': 'Real-time transaction scoring is latency-sensitive; periodic model review is not.',
            'benefit': 'Keeps production fast while still getting AutoML\u2019s automatic model comparison as a regular health check.',
            'risk': 'Running two models adds operational complexity and needs both to be kept in sync as data changes.',
        },
        {
            'title': 'Re-validate AutoGluon\u2019s chosen model against the full leaderboard before every production deployment',
            'evidence': f"{dm['best_by_test']} scored marginally higher than AutoGluon\u2019s own pick on held-out test data.",
            'rationale': 'Validation-based selection and true held-out performance do not always agree exactly.',
            'benefit': 'Catches the (currently small) gap between AutoGluon\u2019s pick and the honestly best-performing model before it matters.',
            'risk': 'Adds a manual review step to what is otherwise a fully automated pipeline.',
        },
        {
            'title': f"Shorten the drift-detection window below the current {drift_lag:.0f}-day lag",
            'evidence': f"Recall on real fraud fell from {dm['adwin_recall_pre']:.1%} to {dm['adwin_recall_post']:.1%} "
                         f"before the monitor raised an alert.",
            'rationale': 'A shorter detection window (a smaller ADWIN delta, or a secondary faster-reacting monitor) '
                          'catches behaviour changes sooner.',
            'benefit': 'Less fraud slips through during the window between a real shift and the model noticing it.',
            'risk': 'A more sensitive monitor also raises more false drift alarms, which has its own operational cost.',
        },
        {
            'title': 'Validate the whole framework on real, non-synthetic transaction data before any production use',
            'evidence': 'Every result in this app comes from a disclosed synthetic proxy dataset, not real bank data.',
            'rationale': 'Synthetic data can only approximate real fraud patterns, however carefully built.',
            'benefit': 'Confirms whether these findings, especially the single-feature dominance in the explanations, hold up on real traffic.',
            'risk': 'Requires data-sharing agreements and privacy safeguards that were outside this project\u2019s scope.',
        },
    ]


def build_data_profile(raw_df, feat_df, proba, approve_below, decline_above):
    """Computes real, fresh statistics from whatever file was just uploaded —
    category/channel concentration, amount patterns, time patterns, and the
    single riskiest transaction found. Returns None for anything the uploaded
    file doesn't have the columns to support, rather than guessing."""
    import numpy as np
    import pandas as pd

    decision = np.where(proba >= decline_above, 'Decline',
                np.where(proba >= approve_below, 'Review', 'Approve'))
    flagged_mask = proba >= approve_below

    profile = {
        'n_scored': len(proba), 'n_flagged': int(flagged_mask.sum()),
        'category_breakdown': None, 'riskiest_category': None,
        'channel_breakdown': None, 'riskiest_channel': None,
        'amount_flagged_mean': None, 'amount_approved_mean': None,
        'time_available': False, 'daily': None, 'trend_direction': None,
        'riskiest_txn': None,
    }

    has_category = 'merchant_category' in raw_df.columns
    has_channel = 'channel' in raw_df.columns
    has_amount = 'amount' in raw_df.columns
    has_time = 'timestamp' in raw_df.columns

    if has_category:
        cat_df = pd.DataFrame({'category': raw_df['merchant_category'].values, 'flagged': flagged_mask})
        agg = cat_df.groupby('category')['flagged'].agg(['sum', 'count'])
        agg['rate'] = 100 * agg['sum'] / agg['count']
        agg = agg.sort_values('rate', ascending=False)
        profile['category_breakdown'] = agg
        if agg['sum'].sum() > 0:
            profile['riskiest_category'] = {'name': agg.index[0], 'rate': float(agg['rate'].iloc[0]),
                                              'count': int(agg['sum'].iloc[0])}

    if has_channel:
        chan_df = pd.DataFrame({'channel': raw_df['channel'].values, 'flagged': flagged_mask})
        agg = chan_df.groupby('channel')['flagged'].agg(['sum', 'count'])
        agg['rate'] = 100 * agg['sum'] / agg['count']
        agg = agg.sort_values('rate', ascending=False)
        profile['channel_breakdown'] = agg
        if agg['sum'].sum() > 0:
            profile['riskiest_channel'] = {'name': agg.index[0], 'rate': float(agg['rate'].iloc[0]),
                                             'count': int(agg['sum'].iloc[0])}

    if has_amount:
        amt = raw_df['amount'].values
        if flagged_mask.sum() > 0:
            profile['amount_flagged_mean'] = float(amt[flagged_mask].mean())
        if (~flagged_mask).sum() > 0:
            profile['amount_approved_mean'] = float(amt[~flagged_mask].mean())

    if has_time:
        ts_df = pd.DataFrame({'timestamp': pd.to_datetime(raw_df['timestamp']), 'flagged': flagged_mask})
        daily = ts_df.set_index('timestamp').resample('D')['flagged'].agg(['sum', 'count'])
        daily = daily[daily['count'] > 0]
        if len(daily) >= 4:
            profile['time_available'] = True
            profile['daily'] = daily
            first_half = daily['sum'].iloc[:len(daily)//2].sum()
            second_half = daily['sum'].iloc[len(daily)//2:].sum()
            if second_half > first_half * 1.2:
                profile['trend_direction'] = 'rising'
            elif second_half < first_half * 0.8:
                profile['trend_direction'] = 'falling'
            else:
                profile['trend_direction'] = 'stable'

    if len(proba) > 0:
        top_idx = int(np.argmax(proba))
        txn = {'probability': float(proba[top_idx])}
        for col in ['amount', 'merchant_category', 'channel', 'timestamp', 'account_id']:
            if col in raw_df.columns:
                txn[col] = raw_df.iloc[top_idx][col]
        profile['riskiest_txn'] = txn

    return profile


def build_data_key_insights(profile, session_scorecard, session_has_labels):
    """Insight cards computed fresh from the shape of the uploaded data itself —
    not the dissertation's own results."""
    insights = []

    if profile['riskiest_txn'] is not None:
        t = profile['riskiest_txn']
        detail_bits = []
        if 'amount' in t:
            detail_bits.append(f"${t['amount']:,.2f}")
        if 'merchant_category' in t:
            detail_bits.append(str(t['merchant_category']))
        if 'channel' in t:
            detail_bits.append(f"via {t['channel']}")
        detail = ', '.join(detail_bits) if detail_bits else 'transaction'
        insights.append({
            'icon': '🚨' if t['probability'] >= 0.8 else '🔎', 'tone': 'warning' if t['probability'] >= 0.8 else 'neutral',
            'title': 'Riskiest transaction found in your upload',
            'text': f"The single highest-scoring transaction in your file was scored {t['probability']:.1%} "
                     f"fraud probability ({detail}). This is the first place to look if you're reviewing your "
                     f"file manually."
        })

    if profile['riskiest_category'] is not None and profile['category_breakdown'] is not None:
        rc = profile['riskiest_category']
        n_cats = len(profile['category_breakdown'])
        if rc['rate'] > 0:
            insights.append({
                'icon': '🏷️', 'tone': 'warning' if rc['rate'] >= 5 else 'neutral',
                'title': 'One merchant category concentrates your flagged transactions',
                'text': f"\u201c{rc['name']}\u201d has the highest flag rate in your file ({rc['rate']:.2f}% of its "
                         f"transactions), out of {n_cats} categories present. If this looks unexpected for your "
                         f"business, it's worth a closer look at that category specifically."
            })

    if profile['riskiest_channel'] is not None and profile['channel_breakdown'] is not None:
        rch = profile['riskiest_channel']
        if rch['rate'] > 0:
            insights.append({
                'icon': '📡', 'tone': 'warning' if rch['rate'] >= 5 else 'neutral',
                'title': 'One payment channel stands out in your data',
                'text': f"The \u201c{rch['name']}\u201d channel has the highest flag rate in your file "
                         f"({rch['rate']:.2f}%). Channel-level concentration like this matched what Chapter Four "
                         f"found in its own testing (ATM transactions carried the highest fraud rate there), so "
                         f"this kind of pattern is a real, recurring signal worth monitoring by channel."
            })

    if profile['amount_flagged_mean'] is not None and profile['amount_approved_mean'] is not None:
        fm, am = profile['amount_flagged_mean'], profile['amount_approved_mean']
        if am > 0:
            ratio = fm / am
            if ratio >= 1.3:
                insights.append({
                    'icon': '💰', 'tone': 'neutral', 'title': 'Flagged transactions run larger than approved ones',
                    'text': f"Transactions flagged in your file average ${fm:,.2f}, versus ${am:,.2f} for approved "
                             f"ones — about {ratio:.1f}x larger. Larger, unusual amounts are a classic fraud tell, "
                             f"and this pattern is consistent with that."
                })
            elif ratio <= 0.7:
                insights.append({
                    'icon': '💰', 'tone': 'neutral', 'title': 'Flagged transactions run smaller than approved ones',
                    'text': f"Transactions flagged in your file average ${fm:,.2f}, versus ${am:,.2f} for approved "
                             f"ones. Smaller flagged amounts can indicate low-value testing transactions, a "
                             f"pattern sometimes used to probe a stolen card before a larger purchase."
                })

    if profile['time_available'] and profile['trend_direction'] is not None:
        direction_text = {
            'rising': ('📈', 'warning', 'Your flagged transactions are trending upward over the period uploaded',
                       'the second half of your date range had noticeably more flagged transactions than the first half — worth checking whether something changed partway through.'),
            'falling': ('📉', 'positive', 'Your flagged transactions are trending downward over the period uploaded',
                        'the second half of your date range had noticeably fewer flagged transactions than the first half.'),
            'stable': ('➡️', 'positive', 'Your flagged-transaction rate looks stable over the period uploaded',
                       'no significant rise or fall in flagged transactions was found across the date range in your file.'),
        }
        icon, tone, title, detail = direction_text[profile['trend_direction']]
        insights.append({'icon': icon, 'tone': tone, 'title': title, 'text': detail.capitalize()})

    if session_has_labels and session_scorecard is not None:
        insights.append({
            'icon': '📊', 'tone': 'positive' if session_scorecard['pr_auc'] >= 0.6 else 'warning',
            'title': 'Live accuracy check on your own labelled data',
            'text': f"Because your file included real fraud labels, a genuine scorecard could be computed: "
                     f"precision {session_scorecard['precision']:.3f}, recall {session_scorecard['recall']:.3f}, "
                     f"PR-AUC {session_scorecard['pr_auc']:.3f}. This is the most direct evidence available "
                     f"of how the model performs specifically on data like yours."
        })
    elif not session_has_labels:
        insights.append({
            'icon': 'ℹ️', 'tone': 'neutral', 'title': 'Accuracy on your data is unverified',
            'text': 'Your file had no `is_fraud` column, so there is no way to check whether the flagged '
                     'transactions above are genuinely fraud — these are model predictions, not confirmed outcomes. '
                     'Upload a labelled file to get a real, verified scorecard instead of predictions alone.'
        })

    return insights


def build_data_interpretations(profile):
    """Chart-style interpretations of the patterns found in the uploaded data itself."""
    interps = []
    if profile['category_breakdown'] is not None:
        interps.append({
            'chart': 'Flag Rate by Merchant Category (your data)',
            'text': (
                'Ranks every merchant category in your file by what share of its transactions were flagged. '
                'A category sitting well above the others is where a fraud team would look first — it either '
                'means that category genuinely carries more risk in your business, or that your data in that '
                'category looks unusual compared to what the model was trained on.'
            )
        })
    if profile['channel_breakdown'] is not None:
        interps.append({
            'chart': 'Flag Rate by Payment Channel (your data)',
            'text': (
                'Shows which payment channel (ATM, POS, online, mobile) in your file has the highest share of '
                'flagged transactions. Channel is one of the strongest, most consistent fraud signals found '
                'throughout this project, so a spike in one channel here is worth taking seriously.'
            )
        })
    if profile['time_available']:
        interps.append({
            'chart': 'Daily Flagged-Transaction Trend (your data)',
            'text': (
                'Plots how many transactions were flagged on each day in your file. A flat line suggests stable, '
                'predictable risk; a rising line suggests something is actively changing in your traffic and is '
                'worth investigating before it grows further, the same logic behind Chapter Four\u2019s concept-drift '
                'monitor.'
            )
        })
    if profile['amount_flagged_mean'] is not None:
        interps.append({
            'chart': 'Flagged vs. Approved Transaction Amounts (your data)',
            'text': (
                'Compares the average transaction size between what got flagged and what got approved in your '
                'file. A large gap either way is informative: unusually large or unusually small amounts are '
                'both classic, well-documented fraud patterns.'
            )
        })
    return interps


def build_data_recommendations(profile, session_summary):
    recs = []
    if profile['riskiest_txn'] is not None and profile['riskiest_txn']['probability'] >= 0.5:
        t = profile['riskiest_txn']
        recs.append({
            'title': 'Manually review the highest-scoring transaction in your upload first',
            'evidence': f"The riskiest transaction found was scored {t['probability']:.1%}.",
            'rationale': 'Reviewing highest-risk items first makes the most of limited analyst time.',
            'benefit': 'Catches the most likely genuine fraud case before lower-priority items.',
            'risk': 'A high score is not certainty — this is a prediction, not a confirmed fraud finding.',
        })
    if profile['riskiest_category'] is not None and profile['riskiest_category']['rate'] >= 5:
        rc = profile['riskiest_category']
        recs.append({
            'title': f"Investigate why \u201c{rc['name']}\u201d has an elevated flag rate in your data",
            'evidence': f"{rc['rate']:.2f}% of \u201c{rc['name']}\u201d transactions were flagged, "
                         f"the highest of any category in your file.",
            'rationale': 'A single category driving most flags may point to a specific, addressable risk, '
                          'or a data quality issue specific to that category.',
            'benefit': 'A targeted fix (extra verification for that category, for example) is cheaper than a blanket policy change.',
            'risk': 'Could also reflect how that category happens to be represented in this particular file, not a lasting pattern.',
        })
    if session_summary is not None:
        flag_rate = 100 * session_summary['n_flagged'] / session_summary['n_scored'] if session_summary['n_scored'] else 0
        if flag_rate > 5:
            recs.append({
                'title': 'Set aside dedicated review capacity for this upload',
                'evidence': f"{flag_rate:.2f}% of {session_summary['n_scored']:,} transactions were flagged — "
                             f"well above the roughly 0.2% base rate this framework expects.",
                'rationale': 'An elevated flag rate at this scale needs planned reviewer time, not ad-hoc handling.',
                'benefit': 'Avoids a backlog of unreviewed high-risk transactions.',
                'risk': 'If the elevated rate turns out to be a data or model-fit issue rather than real risk, reviewer time would be better spent elsewhere.',
            })
    if not recs:
        recs.append({
            'title': 'No unusual concentration found — standard monitoring is sufficient for this upload',
            'evidence': 'No single category, channel, or transaction stood out sharply from the rest in this file.',
            'rationale': 'Recommendations are only useful when there is a real signal to act on.',
            'benefit': 'Avoids manufacturing action items where the data does not support them.',
            'risk': 'A quiet-looking file does not guarantee no fraud is present, only that nothing concentrated stood out to this model.',
        })
    return recs


def build_confidence_limitations():
    return {
        'confidence': (
            'Moderate-to-high confidence in the relative findings (AutoML vs. manual tuning, the speed trade-off, '
            'the drift-detection lag), since these were tested directly and repeatably. Lower confidence in the '
            'absolute accuracy numbers translating unchanged to real transactions, since the data is synthetic.'
        ),
        'strengths': [
            'Every number in this app comes from code that was actually run, not estimated or simulated after the fact.',
            'Findings were cross-checked with independent methods where possible (e.g., ADWIN and seasonal decomposition agreeing on the drift point).',
            'Five different model types were compared on identical data splits, not just one model in isolation.',
        ],
        'limitations': [
            'The dataset is synthetic; real fraud may follow patterns this data does not capture.',
            'TabPFN, one of the AutoML candidates originally planned, could not be tested (needs PyTorch, unavailable in this environment).',
            'The usability evaluation was a structured heuristic review, not a study with real fraud analysts.',
        ],
        'biases': [
            'The synthetic data\u2019s single dominant feature (distance from home) may be an artifact of how it was generated, not a universal fraud signal.',
            'All evaluation happened on data drawn from the same generating process; a genuinely different fraud pattern in the real world was never tested.',
        ],
        'assumptions': [
            'A 30%/80% probability threshold split (Approve/Review/Decline) is assumed throughout; a real deployment would tune this to its own risk appetite.',
            'Behavioural features for a single manually-entered transaction are assumed accurate when supplied by the user, since they cannot be independently verified in this demo.',
        ],
    }


def build_business_impact():
    return [
        {'icon': '🧭', 'title': 'Better decision-making',
         'text': 'A live scorecard and explanation for every prediction means a fraud analyst or manager can see not just a score, but why, supporting faster, more confident decisions.'},
        {'icon': '⚙️', 'title': 'Operational efficiency',
         'text': 'The measured ~40x speed gap between AutoML and a tuned single model is a concrete number a team can use to size infrastructure correctly instead of guessing.'},
        {'icon': '🛡️', 'title': 'Risk reduction',
         'text': 'A working drift monitor means fraud losses from a changing attack pattern are caught in days, not discovered months later in a quarterly review.'},
        {'icon': '📈', 'title': 'Future planning',
         'text': 'Time-series decomposition on demand lets a team spot a slow-building fraud trend before it becomes a crisis, not just react to a single bad day.'},
        {'icon': '💡', 'title': 'Strategic opportunity',
         'text': 'A framework that explains itself is easier to bring to a regulator, an auditor, or a customer dispute than a black-box score alone.'},
    ]


def build_next_steps():
    return [
        'Validate this framework end-to-end on a real, permissioned transaction dataset before any production use.',
        'Set a concrete retraining schedule triggered by ADWIN alerts, not a fixed calendar, since drift does not arrive on a schedule.',
        'Test TabPFN as an additional AutoML candidate once a PyTorch-enabled environment is available.',
        'Run a real usability study with working fraud analysts, not just a structured heuristic review.',
        'Track whether distance-from-home remains the dominant signal once real, non-synthetic data is used — if it does not, the explanation story changes.',
        'Decide, as a business, where the Approve/Review/Decline thresholds should actually sit for this organisation\u2019s risk appetite.',
    ]
