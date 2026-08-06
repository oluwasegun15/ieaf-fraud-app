"""
feature_engineering.py
Reproduces, exactly, the feature-engineering method used throughout the IEAF
dissertation (Chapter Three, Sections 3.8-3.9) and the reproducible notebooks.
Given a raw transactions dataframe (one row per transaction, with account_id,
timestamp, amount, merchant_category, channel, distance_from_home), this
builds the same 17 engineered features the model was trained on.
"""
import numpy as np
import pandas as pd

FEATURE_COLS = [
    'log_amount', 'distance_from_home', 'is_night', 'day_of_week', 'hour',
    'merchant_freq', 'ch_mobile', 'ch_online', 'ch_pos',
    'txn_count_1d', 'txn_count_7d', 'txn_count_30d',
    'amount_mean_7d', 'amount_std_7d',
    'velocity', 'time_since_last_txn_h', 'amount_zscore_vs_7d',
]

RAW_REQUIRED_COLS = ['account_id', 'timestamp', 'amount', 'merchant_category', 'channel', 'distance_from_home']

# Global merchant-category frequency map learned during training. Used so a
# single uploaded transaction (with no other transactions to compare against)
# still gets a sensible merchant_freq value. If the uploaded file has enough
# volume, its own category frequencies are used instead (closer to how the
# model was actually trained).
DEFAULT_MERCHANT_FREQ = {
    'grocery': 0.0951, 'electronics': 0.0956, 'travel': 0.0984, 'restaurant': 0.0952,
    'fuel': 0.0953, 'online_retail': 0.1058, 'utilities': 0.0951, 'entertainment': 0.0955,
    'jewelry': 0.0966, 'cash_advance': 0.0955,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a raw transactions dataframe and returns it with the 17 engineered
    feature columns added, following the dissertation's exact method: static
    features from the transaction itself, causal rolling behavioural features
    (1/7/30-day windows, computed per account so no future information leaks
    into a feature), and temporal/sequence-derived features."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['account_id', 'timestamp']).reset_index(drop=True)

    # --- Static features ---
    df['log_amount'] = np.log1p(df['amount'].clip(lower=0))
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] >= 22)).astype(int)

    cat_freq_local = df['merchant_category'].value_counts(normalize=True).to_dict()
    # Blend: prefer the uploaded file's own frequencies when there's enough data
    # to trust them (at least 200 rows), otherwise fall back to the training-time values.
    use_local = len(df) >= 200
    df['merchant_freq'] = df['merchant_category'].map(
        cat_freq_local if use_local else DEFAULT_MERCHANT_FREQ
    ).fillna(0.095)

    df = pd.get_dummies(df, columns=['channel'], prefix='ch', drop_first=False)
    for c in ['ch_mobile', 'ch_online', 'ch_pos', 'ch_atm']:
        if c not in df.columns:
            df[c] = 0
    # atm was the training-time reference (dropped) category
    df = df.drop(columns=['ch_atm'], errors='ignore')

    # --- Rolling behavioural features (causal, per account) ---
    df = df.set_index('timestamp')
    out_frames = []
    for acc_id, g in df.groupby('account_id', sort=False):
        g = g.sort_index()
        amt = g['amount']
        cnt_1d = amt.rolling('1D').count()
        cnt_7d = amt.rolling('7D').count()
        cnt_30d = amt.rolling('30D').count()
        mean_7d = amt.rolling('7D').mean()
        std_7d = amt.rolling('7D').std()
        g = g.assign(
            txn_count_1d=cnt_1d.values, txn_count_7d=cnt_7d.values, txn_count_30d=cnt_30d.values,
            amount_mean_7d=mean_7d.values, amount_std_7d=std_7d.values,
        )
        out_frames.append(g)
    df = pd.concat(out_frames).reset_index()
    df = df.sort_values(['account_id', 'timestamp']).reset_index(drop=True)

    df['txn_count_1d'] = (df['txn_count_1d'] - 1).clip(lower=0)
    df['txn_count_7d'] = (df['txn_count_7d'] - 1).clip(lower=0)
    df['txn_count_30d'] = (df['txn_count_30d'] - 1).clip(lower=0)
    df['velocity'] = df['txn_count_1d'] / (df['txn_count_7d'] / 7.0 + 0.01)

    df['time_since_last_txn_h'] = (
        df.groupby('account_id')['timestamp'].diff().dt.total_seconds() / 3600.0
    )
    median_gap = df['time_since_last_txn_h'].median()
    df['time_since_last_txn_h'] = df['time_since_last_txn_h'].fillna(median_gap if pd.notna(median_gap) else 24.0)
    df['amount_std_7d'] = df['amount_std_7d'].fillna(0.0)
    df['amount_mean_7d'] = df['amount_mean_7d'].fillna(df['amount'])
    df['amount_zscore_vs_7d'] = (df['amount'] - df['amount_mean_7d']) / (df['amount_std_7d'] + 1.0)

    # Cast every model feature to a single, uniform numeric dtype. Without this,
    # the columns end up as a mix of float64, int32, int64, and bool (bool for
    # the one-hot channel flags, in particular), and while that mix is harmless
    # for AutoGluon and the manual models, SHAP's numba-compiled masker cannot
    # handle a dataframe whose .values array comes out as dtype=object, which
    # is exactly what a mixed-dtype dataframe produces. This was a real bug,
    # caught by testing the Upload page's "explain this transaction" feature
    # against a real uploaded file, not a hypothetical one.
    df[FEATURE_COLS] = df[FEATURE_COLS].astype('float64')

    return df


def engineer_single_transaction(amount, hour, day_of_week, merchant_category, channel,
                                  distance_from_home, txn_count_1d, txn_count_7d, txn_count_30d,
                                  amount_mean_7d, amount_std_7d, time_since_last_txn_h) -> pd.DataFrame:
    """Builds the 17-feature vector for one manually-specified transaction,
    used by the app's 'Manual Prediction' page. Behavioural features (recent
    transaction counts, recent average amount, etc.) can't be derived from a
    single transaction alone, so they are supplied directly by the user."""
    log_amount = float(np.log1p(max(amount, 0)))
    is_night = int(hour < 6 or hour >= 22)
    merchant_freq = DEFAULT_MERCHANT_FREQ.get(merchant_category, 0.095)
    ch_mobile = int(channel == 'mobile')
    ch_online = int(channel == 'online')
    ch_pos = int(channel == 'pos')
    velocity = txn_count_1d / (txn_count_7d / 7.0 + 0.01)
    amount_zscore_vs_7d = (amount - amount_mean_7d) / (amount_std_7d + 1.0)

    row = {
        'log_amount': log_amount, 'distance_from_home': distance_from_home, 'is_night': is_night,
        'day_of_week': day_of_week, 'hour': hour, 'merchant_freq': merchant_freq,
        'ch_mobile': ch_mobile, 'ch_online': ch_online, 'ch_pos': ch_pos,
        'txn_count_1d': txn_count_1d, 'txn_count_7d': txn_count_7d, 'txn_count_30d': txn_count_30d,
        'amount_mean_7d': amount_mean_7d, 'amount_std_7d': amount_std_7d,
        'velocity': velocity, 'time_since_last_txn_h': time_since_last_txn_h,
        'amount_zscore_vs_7d': amount_zscore_vs_7d,
    }
    return pd.DataFrame([row])[FEATURE_COLS]
