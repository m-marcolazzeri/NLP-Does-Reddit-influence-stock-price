#!/usr/bin/env python3
"""
run_financial_analysis.py
─────────────────────────
Logistic regression analysis of LDA topic influence on next-day stock activity.

Loads daily topic distributions from the LDA pipeline output (results_v1.csv),
downloads intraday price data from Yahoo Finance, and fits three logistic
regression models to predict high-activity days (top-25% absolute return):
    1. Market-only baseline   (return, volatility, SPY controls)
    2. Topics-only            (topic_1 … topic_{K-1}; topic_0 as reference)
    3. Combined               (market controls + topics)

Outputs per ticker:
    data/financial/results_{TICKER}.json    AUC, Gini, odds ratios, p-values
    data/financial/plots/results_{TICKER}.png  coefficient bar chart + AUC comparison

K is read automatically from src/topic_modeling/config_lda.py (K_FINAL).
To analyse a different stock, change TICKER and STOCK_NAME at the top of
the configuration block.

Usage:
    python src/financial/run_financial_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

# Load K_FINAL from the topic modeling config (single source of truth).
sys.path.insert(0, str(BASE_DIR / "src" / "topic_modeling"))
from config_lda import K_FINAL  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — change this block to analyse a different stock
# ---------------------------------------------------------------------------

TICKER      = "AMD"
STOCK_NAME  = "AMD"   # must match the 'stock' column in results_v1.csv

CSV_PATH    = BASE_DIR / "data" / "topic_modeling" / "results_v1.csv"
OUTPUT_DIR  = BASE_DIR / "data" / "financial"
PLOT_DIR    = OUTPUT_DIR / "plots"

DATE_START  = "2025-01-01"
DATE_END    = "2025-12-31"

# topic_0 is the catch-all / noise topic and serves as reference category.
TOPIC_COLS     = [f"topic_{i}" for i in range(K_FINAL)]
TOPIC_FEATURES = [f"topic_{i}" for i in range(1, K_FINAL)]

MARKET_FEATURES = ["return_t", "volatility_t", "spy_return", "spy_volatility"]
ALL_FEATURES    = TOPIC_FEATURES + MARKET_FEATURES

# Activity threshold: top quantile of |return| is labelled "high activity".
ACTIVITY_QUANTILE = 0.75
CV_SPLITS         = 5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def weighted_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate chunk-level topic scores to daily level, weighted by n_messages."""
    records = []
    for date, group in df.groupby("date"):
        total = group["n_messages"].sum()
        row = {"date": date, "n_messages": total}
        for col in TOPIC_COLS:
            row[col] = (group[col] * group["n_messages"]).sum() / total
        records.append(row)
    return pd.DataFrame(records)


def run_cv(name: str, X: pd.DataFrame, y: pd.Series,
           cv: TimeSeriesSplit) -> tuple[float, float]:
    """Fit a logistic regression with time-series CV; return (AUC, Gini)."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)),
    ])
    scores = cross_validate(pipe, X, y, cv=cv,
                            scoring=["accuracy", "f1", "roc_auc"],
                            return_train_score=False)
    auc  = float(scores["test_roc_auc"].mean())
    gini = 2 * auc - 1

    print(f"\n{'='*45}")
    print(f"MODEL: {name}")
    print(f"{'='*45}")
    print(f"AUC:               {auc:.3f} ± {scores['test_roc_auc'].std():.3f}")
    print(f"Gini:              {gini:.3f}")
    print(f"F1:                {scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}")
    print(f"Accuracy:          {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}")
    print(f"Misclassification: {1 - scores['test_accuracy'].mean():.3f}")
    print("AUC per fold:")
    for i, v in enumerate(scores["test_roc_auc"]):
        print(f"  Fold {i+1}: {v:.3f}")
    return auc, gini


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # -- Load and aggregate sentiment data -----------------------------------
    raw = pd.read_csv(CSV_PATH)
    raw = raw[raw["stock"] == STOCK_NAME].copy()
    raw["date"] = pd.to_datetime(raw["date_end"], format="%Y-%m-%d").dt.date

    daily = weighted_daily(raw)
    daily["date"] = pd.to_datetime(daily["date"])

    print(f"Sentiment data: {len(daily)} trading days for {STOCK_NAME}")

    # -- Download price data -------------------------------------------------
    stock_raw = yf.download(TICKER, start=DATE_START, end=DATE_END, auto_adjust=True)
    # Flatten MultiIndex columns produced by recent yfinance versions.
    if isinstance(stock_raw.columns, pd.MultiIndex):
        stock_raw.columns = stock_raw.columns.get_level_values(0)
    stock_raw = stock_raw[["Open", "Close"]].copy()
    stock_raw.columns = ["open", "close"]
    stock_raw = stock_raw.reset_index()
    stock_raw = stock_raw.rename(columns={stock_raw.columns[0]: "date"})
    stock_raw["date"] = pd.to_datetime(stock_raw["date"])

    spy_raw = yf.download("SPY", start=DATE_START, end=DATE_END, auto_adjust=True)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy_raw["spy_return"]     = (spy_raw["Close"] - spy_raw["Open"]) / spy_raw["Open"]
    spy_raw["spy_volatility"] = spy_raw["spy_return"].abs()
    spy_raw = spy_raw[["spy_return", "spy_volatility"]].reset_index()
    spy_raw = spy_raw.rename(columns={spy_raw.columns[0]: "date"})
    spy_raw["date"] = pd.to_datetime(spy_raw["date"])

    # -- Merge ---------------------------------------------------------------
    df = pd.merge(daily, stock_raw, on="date", how="inner")
    df = pd.merge(df, spy_raw, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    # -- Features and target -------------------------------------------------
    df["return_t"]     = (df["close"] - df["open"]) / df["open"]
    df["volatility_t"] = df["return_t"].abs()

    threshold    = df["return_t"].abs().quantile(ACTIVITY_QUANTILE)
    df["target"] = (df["return_t"].abs() > threshold).astype(int).shift(-1)
    df           = df.dropna()
    df["target"] = df["target"].astype(int)

    print(f"Final dataset:      {len(df)} trading days")
    print(f"High activity days: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
    print(f"Low activity days:  {(1-df['target']).sum()} ({(1-df['target']).mean()*100:.1f}%)")

    X_market = df[MARKET_FEATURES]
    X_topics = df[TOPIC_FEATURES]
    X_all    = df[ALL_FEATURES]
    y        = df["target"]

    # -- Cross-validated model comparison ------------------------------------
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    auc_market,   gini_market   = run_cv("1. Market only (baseline)",     X_market, y, tscv)
    auc_topics,   gini_topics   = run_cv("2. Topics only (ref: topic_0)", X_topics, y, tscv)
    auc_combined, gini_combined = run_cv("3. Combined (market + topics)",  X_all,   y, tscv)

    print(f"\n-- Summary for {TICKER} --")
    print(f"Market only:  AUC = {auc_market:.3f}   Gini = {gini_market:.3f}")
    print(f"Topics only:  AUC = {auc_topics:.3f}   Gini = {gini_topics:.3f}")
    print(f"Combined:     AUC = {auc_combined:.3f}   Gini = {gini_combined:.3f}")
    print(f"Improvement:  +{auc_combined - auc_market:.3f} AUC over market baseline")

    # -- Statsmodels logit (full table with p-values) ------------------------
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    X_sm     = sm.add_constant(pd.DataFrame(X_scaled, columns=ALL_FEATURES))

    result = sm.Logit(y.reset_index(drop=True), X_sm).fit()
    print(result.summary())

    odds_df = pd.DataFrame({
        "feature":     ALL_FEATURES,
        "coefficient": result.params[1:].values,
        "odds_ratio":  np.exp(result.params[1:].values),
        "p_value":     result.pvalues[1:].values,
    }).sort_values("odds_ratio", ascending=False).reset_index(drop=True)

    print(f"\n-- Odds ratios for {TICKER} (reference: topic_0) --")
    print(odds_df.to_string(index=False))

    # -- Save JSON results ---------------------------------------------------
    output = {
        "ticker": TICKER,
        "K": K_FINAL,
        "n_days": len(df),
        "models": {
            "market_only": {"auc": round(auc_market,   4), "gini": round(gini_market,   4)},
            "topics_only": {"auc": round(auc_topics,   4), "gini": round(gini_topics,   4)},
            "combined":    {"auc": round(auc_combined, 4), "gini": round(gini_combined, 4)},
        },
        "auc_improvement_over_market": round(auc_combined - auc_market, 4),
        "odds_ratios": odds_df.to_dict(orient="records"),
    }
    json_path = OUTPUT_DIR / f"results_{TICKER}.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {json_path}")

    # -- Plots ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["red" if c < 0 else "steelblue" for c in odds_df["coefficient"]]
    axes[0].barh(odds_df["feature"], odds_df["coefficient"], color=colors)
    axes[0].axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("Coefficient (log-odds)")
    axes[0].set_title(f"{TICKER} — Coefficients (ref: topic_0)")
    axes[0].invert_yaxis()

    models = ["Market\nOnly", "Topics\nOnly", "Combined"]
    aucs   = [auc_market, auc_topics, auc_combined]
    bar_colors = ["#d3d3d3", "#a8c5e8", "#2196F3"]
    bars = axes[1].bar(models, aucs, color=bar_colors, edgecolor="black", linewidth=0.8)
    axes[1].axhline(y=0.5, color="red", linestyle="--", linewidth=0.8, label="Random baseline")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title(f"{TICKER} — Model Comparison")
    axes[1].set_ylim(0.4, 0.8)
    axes[1].legend()
    for bar, auc in zip(bars, aucs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{auc:.3f}", ha="center", va="bottom", fontweight="bold")

    plt.suptitle(f"Results for {TICKER}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = PLOT_DIR / f"results_{TICKER}.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
