# ml_agent_tsx_fixed.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    mean_squared_error,
)
import joblib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --------------------------
# Config
# --------------------------
TICKERS = ["TD.TO", "BNS.TO", "ENB.TO", "TRP.TO", "BAM.TO", "CP.TO"]  # sample watchlist
START = "2015-01-01"
END = datetime.today().strftime("%Y-%m-%d")
LOOKBACK_DAYS = 252  # features history window (1 year)
PRED_HORIZON = 21    # target horizon in trading days (approx 1 month)
REBALANCE_DAYS = 21
TOP_N = 3
INITIAL_CAPITAL = 100_000
TRANSACTION_COST_PCT = 0.001  # 0.1% per trade

RANDOM_STATE = 42
CLASSIFIER_MODE = True  # True => classification target (up/down); False => regression (forward return)

# --------------------------
# Utilities / Feature functions
# --------------------------
def fetch_adjusted_close(tickers, start, end):
    """
    Fetch close prices for tickers. If a ticker fails (delisted / missing),
    skip it and return a DataFrame with the remaining tickers.
    """
    print("Downloading:", tickers)
    df = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True, progress=False)

    # If yfinance returned multi-index columns (multiple tickers)
    if isinstance(df.columns, pd.MultiIndex):
        # Try to extract Close for each ticker that exists in the response
        close_dict = {}
        for t in tickers:
            try:
                # df[t]['Close'] will raise if t not in returned columns
                close_series = df[t]['Close'].copy()
                if close_series.dropna().shape[0] > 0:
                    close_dict[t] = close_series
                else:
                    print(f"Warning: no data for {t}, skipping.")
            except Exception:
                print(f"Warning: {t} not found in downloaded data, skipping.")
        if not close_dict:
            raise ValueError("No tickers returned from yfinance. Check tickers or network.")
        close = pd.DataFrame(close_dict)
    else:
        # Single ticker case (or yfinance returned a flat frame)
        # Determine the symbol that has data among provided tickers by inspecting columns
        # If there is exactly one ticker in the user list, use that
        if len(tickers) == 1:
            # df likely contains columns like 'Open','High',..., 'Close'
            if 'Close' in df.columns:
                close = pd.DataFrame({tickers[0]: df['Close']})
            else:
                raise ValueError("Single-ticker download didn't return 'Close' column.")
        else:
            # Unexpected flat return for multiple tickers — try to find column names that equal tickers
            possible = {}
            for c in df.columns:
                if c in tickers:
                    possible[c] = df[c]['Close'] if isinstance(df[c], pd.DataFrame) and 'Close' in df[c].columns else df[c]
            if possible:
                close = pd.DataFrame(possible)
            else:
                raise ValueError("Unexpected yfinance response format; could not build close DataFrame.")
    close = close.sort_index().ffill().dropna(how='all')
    return close

def compute_features(close):
    """
    Input: close prices DataFrame (dates x tickers)
    Output: features DataFrame keyed by MultiIndex (date, ticker) and columns = features
    """
    returns = close.pct_change().fillna(0)
    features = []
    idx = []
    for t in close.columns:
        s = close[t].dropna()
        if len(s) < LOOKBACK_DAYS:
            # Skip tickers without sufficient history
            continue
        # 1) momentum: returns over 21/63/126/252 days
        mom21 = s.pct_change(21)
        mom63 = s.pct_change(63)
        mom126 = s.pct_change(126)
        mom252 = s.pct_change(252)
        # 2) volatility: rolling std of daily returns
        vol21 = returns[t].rolling(21).std()
        vol63 = returns[t].rolling(63).std()
        # 3) moving averages and MA spreads
        ma21 = s.rolling(21).mean()
        ma63 = s.rolling(63).mean()
        ma_spread_21_63 = (ma21 - ma63) / ma63
        # 4) recent skew/kurt (simple)
        roll_skew21 = returns[t].rolling(21).skew()
        roll_kurt21 = returns[t].rolling(21).kurt()
        # 5) RSI (simple implementation)
        delta = s.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -delta.clip(upper=0).rolling(14).mean()
        rsi = 100 - 100 / (1 + up / down)
        df_feat = pd.DataFrame({
            "close": s,
            "mom21": mom21,
            "mom63": mom63,
            "mom126": mom126,
            "mom252": mom252,
            "vol21": vol21,
            "vol63": vol63,
            "ma21": ma21,
            "ma63": ma63,
            "ma_spread_21_63": ma_spread_21_63,
            "skew21": roll_skew21,
            "kurt21": roll_kurt21,
            "rsi14": rsi,
        })
        for date, row in df_feat.iterrows():
            idx.append((date, t))
            features.append(row.to_dict())

    feat_df = pd.DataFrame(features, index=pd.MultiIndex.from_tuples(idx, names=["date", "ticker"]))
    feat_df = feat_df.dropna(thresh=int(len(feat_df.columns) * 0.6))
    return feat_df

def build_targets(close, horizon=PRED_HORIZON, classifier=True):
    """
    Returns target Series aligned to the same MultiIndex (date, ticker) used for features.
    For classification: 1 if forward horizon return > 0, else 0.
    """
    idx = []
    values = []
    for t in close.columns:
        s = close[t].dropna()
        if s.empty:
            continue
        fwd_ret = s.shift(-horizon) / s - 1.0  # forward horizon returns
        for date in s.index:
            idx.append((date, t))
            values.append(fwd_ret.get(date, np.nan))
    targ = pd.Series(values, index=pd.MultiIndex.from_tuples(idx, names=["date", "ticker"]))
    if classifier:
        return (targ > 0).astype(int)
    else:
        return targ

# --------------------------
# Prepare data
# --------------------------
print("Fetching price data...")
close = fetch_adjusted_close(TICKERS, START, END)
print("Tickers used:", list(close.columns))
print("Computing features (this may take a moment)...")
feat = compute_features(close)
print("Building targets...")
target = build_targets(close, horizon=PRED_HORIZON, classifier=CLASSIFIER_MODE)

# Align features and target
common_index = feat.index.intersection(target.index)
feat = feat.loc[common_index].sort_index()
target = target.loc[common_index].sort_index()

# drop rows with NaNs
mask = feat.notna().all(axis=1) & target.notna()
feat = feat[mask]
target = target[mask]

# Create X, y and a 'date' index to use for time-splits
X_meta = feat.reset_index()  # keeps date and ticker columns
y = target.reset_index(drop=True)
dates = X_meta['date'].values
tickers_col = X_meta['ticker'].values
X = X_meta.drop(columns=['date', 'ticker']).astype(float)

print(f"Prepared dataset with {X.shape[0]} rows and {X.shape[1]} features.")

# --------------------------
# Time-aware CV + pipeline + hyperparameter search
# --------------------------
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

if CLASSIFIER_MODE:
    model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    param_dist = {
        "estimator__max_iter": [100, 250, 500],
        "estimator__max_depth": [3, 5, 8, None],
        "estimator__learning_rate": [0.01, 0.05, 0.1],
        "estimator__max_leaf_nodes": [10, 31, 63, None],
    }
else:
    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    param_dist = {
        "estimator__max_iter": [100, 250, 500],
        "estimator__max_depth": [3, 5, 8, None],
        "estimator__learning_rate": [0.01, 0.05, 0.1],
        "estimator__max_leaf_nodes": [10, 31, 63, None],
    }

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("estimator", model)
])

n_iter_search = 25
scoring = "roc_auc" if CLASSIFIER_MODE else "neg_root_mean_squared_error"

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=n_iter_search,
    cv=tscv,
    scoring=scoring,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

print("Running hyperparameter search (time-series aware)...")
search.fit(X, y)
print("Best params:", search.best_params_)
print("Best CV score:", search.best_score_)

best_model = search.best_estimator_
joblib.dump(best_model, "best_ml_agent.joblib")
print("Saved best model to best_ml_agent.joblib")

# --------------------------
# Backtest rules
# --------------------------
# Reconstruct dataframe with predictions
X_with_meta = X_meta.copy()
X_only = X_with_meta.drop(columns=['date', 'ticker']).astype(float)
if CLASSIFIER_MODE:
    probs = best_model.predict_proba(X_only)[:, 1]
else:
    # for regression we want larger = better, so keep predicted returns
    probs = best_model.predict(X_only)

pred_df = X_with_meta[['date', 'ticker']].copy()
pred_df['score'] = probs
pred_df['target'] = y.values

# Build price_long with columns ['date','ticker','price']
close_reset = close.reset_index()
# Normalize date column name to 'date'
date_col = close_reset.columns[0]
close_reset = close_reset.rename(columns={date_col: 'date'})
price_long = close_reset.melt(id_vars='date', var_name='ticker', value_name='price')

# Drop rows with missing price (can't trade them)
price_long = price_long.dropna(subset=['price']).copy()
price_long['date'] = pd.to_datetime(price_long['date'])

# Ensure pred_df has 'date' column as datetime
if 'date' not in pred_df.columns:
    pred_df = pred_df.reset_index().rename(columns={'index': 'date'})
pred_df['date'] = pd.to_datetime(pred_df['date'])

# Merge predictions onto price_long
merged = price_long.merge(pred_df, on=['date', 'ticker'], how='left').sort_values(['date', 'ticker'])

# forward fill last available score per ticker (so we can pick on days where model had a previous score)
merged['score'] = merged.groupby('ticker')['score'].ffill()

# simple backtest: on each rebalance day choose top-N tickers by score, equal weight, buy and hold until next rebalance
dates_unique = merged['date'].drop_duplicates().sort_values().reset_index(drop=True)
cash = INITIAL_CAPITAL
portfolio_values = []
last_rebalance_index = 0
holdings = {}  # ticker -> shares

for i, cur_date in enumerate(dates_unique):
    # on rebalance days (start or every REBALANCE_DAYS), compute picks using the latest available scores
    if i == 0 or (i - last_rebalance_index) % REBALANCE_DAYS == 0:
        last_rebalance_index = i
        day_slice = merged[merged['date'] <= cur_date]
        # take latest row per ticker as of cur_date (so we get latest score and price as of each ticker)
        latest_scores = day_slice.groupby('ticker').apply(lambda g: g.loc[g['date'].idxmax()]).reset_index(drop=True)
        latest_scores = latest_scores.dropna(subset=['score'])  # require a score
        picks = latest_scores.sort_values('score', ascending=False).head(TOP_N)
        pick_list = picks['ticker'].tolist()

        # compute current portfolio value using current market prices (cur_date)
        cur_prices = merged[merged['date'] == cur_date].set_index('ticker')['price'].to_dict()
        total_value = cash + sum(holdings.get(t, 0) * cur_prices.get(t, 0) for t in holdings)

        # equal-weight allocate to picks
        if len(pick_list) > 0:
            alloc = total_value / len(pick_list)
            new_holdings = {}
            for t in pick_list:
                price = cur_prices.get(t, np.nan)
                if pd.isna(price) or price <= 0:
                    # skip buying this ticker due to missing price
                    continue
                # compute shares as integer
                shares = int((alloc * (1 - TRANSACTION_COST_PCT)) / price)
                if shares <= 0:
                    continue
                new_holdings[t] = shares
            # update holdings and cash
            holdings = new_holdings
            spent = sum(holdings[t] * cur_prices.get(t, 0) for t in holdings)
            # apply transaction cost as a fraction of trade value
            cash = total_value - spent - TRANSACTION_COST_PCT * spent
        else:
            # no picks -> hold cash, keep old holdings (or liquidate depending on desired rule). Here we liquidate.
            holdings = {}
            # cash remains total_value (already in cash variable)

    # compute portfolio value at cur_date
    cur_prices = merged[merged['date'] == cur_date].set_index('ticker')['price'].to_dict()
    pv = cash + sum(holdings.get(t, 0) * cur_prices.get(t, 0) for t in holdings)
    portfolio_values.append((cur_date, pv))

pf = pd.DataFrame(portfolio_values, columns=['date', 'value']).set_index('date').sort_index()

# Performance metrics
def performance_stats(pf_series):
    rets = pf_series.pct_change().dropna()
    total_return = pf_series.iloc[-1] / pf_series.iloc[0] - 1
    ann_return = (1 + total_return) ** (252 / len(rets)) - 1 if len(rets) > 0 else np.nan
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    dd = (pf_series.cummax() - pf_series) / pf_series.cummax()
    max_dd = dd.max()
    return {
        "start": float(pf_series.iloc[0]),
        "end": float(pf_series.iloc[-1]),
        "total_return": float(total_return),
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd)
    }

stats = performance_stats(pf['value'])
print("Backtest stats:", stats)

# Plot equity curve
plt.figure(figsize=(10,5))
plt.plot(pf['value'])
plt.title("ML Agent — Equity Curve (fixed)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.show()

# If classifier: quick classification metrics
if CLASSIFIER_MODE:
    pred_labels = (pred_df['score'] > np.nanmedian(pred_df['score'])).astype(int)
    y_true = pred_df['target']
    mask = ~np.isnan(pred_df['score']) & ~np.isnan(y_true)
    if mask.any():
        print("Classification diagnostics on available scored rows:")
        print("Accuracy:", accuracy_score(y_true[mask], pred_labels[mask]))
        try:
            print("AUC:", roc_auc_score(y_true[mask], pred_df['score'][mask]))
        except Exception:
            pass
        print("Precision:", precision_score(y_true[mask], pred_labels[mask], zero_division=0))
        print("Recall:", recall_score(y_true[mask], pred_labels[mask], zero_division=0))
    else:
        print("No scored rows available for classification diagnostics.")
else:
    preds = best_model.predict(X_only)
    print("Regression RMSE:", np.sqrt(mean_squared_error(y.values, preds)))

print("Done.")
