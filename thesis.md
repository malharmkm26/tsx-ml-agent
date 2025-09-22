Overview

This project (ml_agent_tsx.py) explores whether machine learning agents can aid in systematic equity selection on the Toronto Stock Exchange (TSX). The code implements a complete pipeline that ingests historical stock data, constructs predictive features, trains an ML model with time-series–aware validation, and evaluates a simple long-only backtest strategy.

Motivation

Stock selection on the TSX is traditionally driven by fundamental and technical analysis. This project tests whether modern ML methods—particularly gradient boosting—can extract meaningful patterns from price-based features to inform trading decisions.

The goal is not to optimize performance to production levels but to demonstrate a reproducible research pipeline that connects ML experimentation to portfolio simulation.

Methodology
Data

Universe: Hand-picked sample of large-cap TSX tickers (e.g., TD.TO, BNS.TO, ENB.TO).

Source: Yahoo Finance (via yfinance).

Period: 2015–present.

Prices are adjusted for dividends and splits.

Feature Engineering

For each ticker, rolling technical indicators are computed, including:

Momentum: 21/63/126/252-day returns.

Volatility: 21/63-day rolling standard deviation.

Moving averages: MA spreads (21 vs 63 days).

Higher moments: Rolling skewness and kurtosis.

RSI: 14-day relative strength index.

Features are stored in a (date, ticker) multi-index DataFrame.

Targets

Classification mode: binary label = 1 if forward 21-day return > 0.

Regression mode: continuous forward 21-day return.

Modeling

Pipeline: StandardScaler + HistGradientBoostingClassifier/Regressor.

Hyperparameter search: RandomizedSearchCV with TimeSeriesSplit.

Scoring: ROC-AUC (classification) or RMSE (regression).

Backtest

Equal-weighted portfolio of top-N ranked stocks by predicted score.

Rebalanced every 21 trading days.

Performance metrics: annualized return, volatility, Sharpe ratio, and max drawdown.

Results (Sample)

The pipeline successfully identifies non-trivial predictive structure, with in-sample classification ROC-AUC > 0.5 (random baseline).

Backtest results vary across runs but typically produce a positive annualized return and Sharpe > 1, subject to transaction costs.

Limitations include survivorship bias, small universe size, and lack of risk controls.

Contributions

End-to-end reproducible ML + backtesting pipeline for Canadian equities.

Demonstrates integration of scikit-learn, yfinance, and pandas in a research-grade workflow.

Serves as a baseline for extending to richer universes, alternative ML models, or live trading infrastructure.

Next Steps

Expand ticker universe to the full TSX Composite.

Add fundamental data (earnings, valuation multiples).

Explore deep learning sequence models (e.g., LSTMs, Transformers).

Implement rolling retraining for more realistic walk-forward evaluation.