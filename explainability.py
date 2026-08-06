"""
explainability.py
Turns a prediction's SHAP values into plain-English "reason cards" a
non-technical person can read and trust — grouping the 17 raw model features
into a handful of human concepts (location, spending pattern, timing, payment
channel, account activity, merchant type), ranking them by how much each
actually contributed to that specific prediction, and writing a real sentence
about the real numbers involved.

This is deliberately NOT powered by an external AI service: every sentence is
built from a template filled in with the exact feature values and SHAP
contributions already computed elsewhere in the app, so what is shown here is
guaranteed to match the numbers behind it, not a rephrasing that could drift
from them. See Chapter Six, Section 6.4.4 for the reasoning behind this choice.
"""
import numpy as np

from feature_engineering import FEATURE_COLS

# Each group: (icon, human label, [feature names in this group], sentence-builder key)
FEATURE_GROUPS = [
    ('📍', 'Location', ['distance_from_home'], 'location'),
    ('💰', 'Spending amount', ['log_amount', 'amount_zscore_vs_7d', 'amount_mean_7d', 'amount_std_7d'], 'amount'),
    ('🕐', 'Time of transaction', ['is_night', 'hour'], 'time'),
    ('💳', 'Payment channel', ['ch_mobile', 'ch_online', 'ch_pos'], 'channel'),
    ('📊', 'Account activity level', ['txn_count_1d', 'txn_count_7d', 'txn_count_30d', 'velocity'], 'activity'),
    ('⏱️', 'Time since last transaction', ['time_since_last_txn_h'], 'recency'),
    ('🏷️', 'Merchant type', ['merchant_freq'], 'merchant'),
    ('📅', 'Day of the week', ['day_of_week'], 'day'),
]

_DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def _fmt_money(x):
    return f"${x:,.2f}"


def _sentence_for_group(key, values):
    """values: dict of {feature_name: raw_value} for this group.
    Each branch below decides its own closing clause from the actual pattern
    it just described (e.g. "far from home" -> suspicious), rather than from
    a separately-computed SHAP sign — a group can blend several raw features
    that pull in different directions, and tagging a sentence with a blind
    aggregate sign risks saying something that contradicts its own content
    (e.g. "close to normal spending, which points toward flagging")."""

    if key == 'location':
        d = values['distance_from_home']
        if d <= 0.1:
            return f"This happened very close to where this customer normally is (distance score {d:.2f}), consistent with normal behaviour."
        elif d <= 0.5:
            return f"This happened a moderate distance from where this customer normally is (distance score {d:.2f}), somewhat unusual for this account."
        else:
            return f"This happened far from where this customer normally is (distance score {d:.2f}), a pattern often seen in fraud."

    if key == 'amount':
        amt = np.expm1(values['log_amount']) if 'log_amount' in values else None
        mean7 = values.get('amount_mean_7d')
        if amt is not None and mean7 is not None and mean7 > 0:
            ratio = amt / mean7
            if ratio >= 1.5:
                return (f"The amount, {_fmt_money(amt)}, is about {ratio:.1f}x this account's typical recent "
                         f"spending of {_fmt_money(mean7)}, an unusually large jump.")
            elif ratio <= 0.6:
                return (f"The amount, {_fmt_money(amt)}, is noticeably smaller than this account's typical recent "
                         f"spending of {_fmt_money(mean7)}, sometimes used to quietly test a stolen card.")
            else:
                return (f"The amount, {_fmt_money(amt)}, is close to this account's typical recent spending of "
                         f"{_fmt_money(mean7)}, consistent with normal behaviour.")
        return "The transaction amount is in line with what this account usually spends."

    if key == 'time':
        hour = values.get('hour')
        is_night = values.get('is_night')
        if hour is not None:
            hour = int(hour)
            hh = hour % 12 or 12
            ampm = 'AM' if hour < 12 else 'PM'
            time_desc = f"{hh}:00 {ampm}"
            if is_night:
                return f"This happened late at night ({time_desc}), an unusual time for most legitimate activity."
            return f"This happened during normal daytime hours ({time_desc}), consistent with typical activity."
        return "The time of day is within this account's usual pattern."

    if key == 'channel':
        if values.get('ch_online'):
            chan = 'online'
        elif values.get('ch_mobile'):
            chan = 'mobile app'
        elif values.get('ch_pos'):
            chan = 'in-person card terminal (POS)'
        else:
            chan = 'ATM'
        return f"This was a {chan} transaction — some channels are more commonly linked to fraud than others."

    if key == 'activity':
        c1d = values.get('txn_count_1d')
        c7d = values.get('txn_count_7d')
        if c1d is not None and c7d is not None:
            avg_daily = c7d / 7.0 if c7d else 0
            if c1d >= 1 and avg_daily > 0 and c1d >= avg_daily * 2:
                return (f"This account made {int(c1d)} transaction(s) in the last day, well above its recent daily "
                         f"average of about {avg_daily:.1f} — a burst of activity sometimes seen when a card is compromised.")
            return (f"This account made {int(c1d)} transaction(s) in the last day, close to its recent daily "
                     f"average of about {avg_daily:.1f}, consistent with normal behaviour.")
        return "This account's recent activity level is within its usual range."

    if key == 'recency':
        h = values.get('time_since_last_txn_h')
        if h is not None:
            if h < 1:
                return f"This came only {h*60:.0f} minutes after the account's previous transaction, an unusually quick follow-up."
            elif h < 24:
                return f"This came about {h:.1f} hours after the account's previous transaction, a fairly typical gap."
            else:
                return f"This came about {h/24:.1f} days after the account's previous transaction, a fairly typical gap."
        return "The gap since the last transaction is within this account's usual range."

    if key == 'merchant':
        freq = values.get('merchant_freq')
        if freq is not None:
            if freq < 0.08:
                return "This merchant category is relatively uncommon in general transaction activity, which can raise attention."
            return "This merchant category is common in general transaction activity, consistent with normal behaviour."
        return "The merchant category is within a typical range."

    if key == 'day':
        dow = values.get('day_of_week')
        if dow is not None:
            name = _DAY_NAMES[int(dow) % 7]
            return f"This happened on a {name}."
        return "The day of the week is not unusual for this account."

    return "This factor was taken into account in the score."


def build_plain_language_reasons(feature_row: dict, shap_values: np.ndarray, top_n: int = 4) -> list:
    """feature_row: {feature_name: raw_value} for the 17 model features (unscaled,
    real-world units — e.g. log_amount still in log space, used internally to
    recover the real dollar amount for the sentence).
    shap_values: array aligned with FEATURE_COLS, the SHAP contribution of each
    raw feature to this one prediction.
    Returns a ranked list of {icon, label, sentence, pct} dicts, most important
    group first, where pct is that group's share of total |SHAP| across all
    groups (so the percentages shown always sum to 100 across everything shown)."""
    shap_by_feature = dict(zip(FEATURE_COLS, shap_values))

    group_scores = []
    for icon, label, feats, key in FEATURE_GROUPS:
        total_shap = sum(shap_by_feature.get(f, 0.0) for f in feats)
        magnitude = sum(abs(shap_by_feature.get(f, 0.0)) for f in feats)
        group_scores.append({'icon': icon, 'label': label, 'feats': feats, 'key': key,
                               'signed': total_shap, 'magnitude': magnitude})

    total_magnitude = sum(g['magnitude'] for g in group_scores) or 1e-9
    group_scores.sort(key=lambda g: g['magnitude'], reverse=True)

    reasons = []
    for g in group_scores[:top_n]:
        if g['magnitude'] <= 0:
            continue
        values = {f: feature_row.get(f) for f in g['feats']}
        direction = 'up' if g['signed'] > 0 else 'down'
        sentence = _sentence_for_group(g['key'], values)
        pct = 100 * g['magnitude'] / total_magnitude
        reasons.append({'icon': g['icon'], 'label': g['label'], 'sentence': sentence,
                          'pct': pct, 'direction': direction})
    return reasons
