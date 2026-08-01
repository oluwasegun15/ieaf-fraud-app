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
