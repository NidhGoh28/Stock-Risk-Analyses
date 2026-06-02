"""
Stock Risk & Portfolio Analyser
================================
Author  : [Your Name]
Degree  : B.Sc. Applied Mathematics
Purpose : Quantitative Finance internship portfolio project

Models implemented
──────────────────
1. Geometric Brownian Motion  — price path simulation
2. Monte Carlo Simulation     — 500 paths × 90-day horizon
3. Value at Risk (VaR)        — 95th & 99th percentile
4. Conditional VaR (CVaR)     — Expected Shortfall
5. Maximum Drawdown           — peak-to-trough decline
6. Sharpe Ratio               — risk-adjusted return
7. Markowitz Efficient Frontier — portfolio optimisation
8. Pearson Correlation Matrix  — diversification analysis

Run
───
    python stock_analyser.py
Outputs: stock_data.json  (consumed by the HTML dashboard)
"""

import numpy as np
import json
from scipy import stats


# ── 1. Asset parameters ──────────────────────────────────────────────────────
# In production: replace with real historical data from yfinance or Alpha Vantage
STOCKS = {
    "AAPL":    {"mu": 0.28, "sigma": 0.22, "S0": 185.0,   "color": "#378ADD"},
    "GOOGL":   {"mu": 0.24, "sigma": 0.25, "S0": 140.0,   "color": "#1D9E75"},
    "TSLA":    {"mu": 0.35, "sigma": 0.55, "S0": 210.0,   "color": "#E24B4A"},
    "JPM":     {"mu": 0.18, "sigma": 0.18, "S0": 195.0,   "color": "#BA7517"},
    "BTC-USD": {"mu": 0.60, "sigma": 0.80, "S0": 42000.0, "color": "#7F77DD"},
}

T       = 252    # trading days in 1 year
dt      = 1/252  # time step
N_SIM   = 500    # Monte Carlo paths
HORIZON = 90     # forecast horizon (days)
SEED    = 42

np.random.seed(SEED)


# ── 2. Geometric Brownian Motion ─────────────────────────────────────────────
def simulate_gbm(mu: float, sigma: float, S0: float,
                 n_days: int, dt: float) -> np.ndarray:
    """
    Exact GBM discretisation:
        S(t+dt) = S(t) * exp((mu - sigma²/2)*dt + sigma*sqrt(dt)*Z)
    where Z ~ N(0,1).
    Returns price array of length n_days+1.
    """
    log_returns = np.random.normal(
        (mu - 0.5 * sigma**2) * dt,
        sigma * np.sqrt(dt),
        n_days
    )
    prices = S0 * np.exp(np.cumsum(log_returns))
    return np.insert(prices, 0, S0)


# ── 3. Risk metrics ───────────────────────────────────────────────────────────
def compute_risk_metrics(prices: np.ndarray) -> dict:
    """Compute standard quantitative risk metrics from a price series."""
    log_ret = np.diff(np.log(prices))

    ann_return = float(np.mean(log_ret) * 252)
    ann_vol    = float(np.std(log_ret) * np.sqrt(252))
    sharpe     = ann_return / ann_vol if ann_vol > 0 else 0.0

    var_95  = float(np.percentile(log_ret, 5))    # 1-day VaR at 95% confidence
    var_99  = float(np.percentile(log_ret, 1))    # 1-day VaR at 99% confidence
    cvar_95 = float(np.mean(log_ret[log_ret <= var_95]))  # Expected Shortfall

    # Maximum drawdown
    peak    = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    max_dd  = float(np.min(drawdown))

    # Higher moments
    skewness = float(stats.skew(log_ret))
    kurtosis = float(stats.kurtosis(log_ret))   # excess kurtosis

    return {
        "ann_return": round(ann_return * 100, 2),
        "ann_vol":    round(ann_vol    * 100, 2),
        "sharpe":     round(sharpe,           3),
        "var_95":     round(var_95     * 100, 3),
        "var_99":     round(var_99     * 100, 3),
        "cvar_95":    round(cvar_95    * 100, 3),
        "max_dd":     round(max_dd     * 100, 2),
        "skew":       round(skewness,         3),
        "kurtosis":   round(kurtosis,         3),
    }


# ── 4. Monte Carlo forecast ───────────────────────────────────────────────────
def monte_carlo(last_price: float, mu: float, sigma: float,
                horizon: int, n_sim: int, dt: float) -> dict:
    """
    Simulate n_sim price paths forward for `horizon` days.
    Returns percentile distribution and sampled paths.
    """
    sim_returns = np.random.normal(
        (mu - 0.5 * sigma**2) * dt,
        sigma * np.sqrt(dt),
        (horizon, n_sim)
    )
    sim_prices   = last_price * np.exp(np.cumsum(sim_returns, axis=0))
    final_prices = sim_prices[-1]

    return {
        "mean": round(float(np.mean(final_prices)),            2),
        "p5":   round(float(np.percentile(final_prices,  5)),  2),
        "p25":  round(float(np.percentile(final_prices, 25)),  2),
        "p50":  round(float(np.median(final_prices)),          2),
        "p75":  round(float(np.percentile(final_prices, 75)),  2),
        "p95":  round(float(np.percentile(final_prices, 95)),  2),
    }, sim_prices[:, :40].tolist()   # return 40 sample paths for charting


# ── 5. Efficient frontier ─────────────────────────────────────────────────────
def efficient_frontier(tickers: list, stocks: dict, n_portfolios: int = 1800) -> list:
    """
    Generate random portfolios and record risk/return/Sharpe.
    Simplified: diagonal covariance (no cross-asset correlations).
    Extend with np.cov(returns_matrix) for full Markowitz.
    """
    means = np.array([stocks[t]["mu"]    for t in tickers])
    vols  = np.array([stocks[t]["sigma"] for t in tickers])
    n     = len(tickers)
    ports = []

    for _ in range(n_portfolios):
        w  = np.random.dirichlet(np.ones(n))
        pr = float(np.dot(w, means))
        pv = float(np.sqrt(np.dot(w**2, vols**2)))
        ps = pr / pv if pv > 0 else 0.0
        ports.append({
            "r": round(pr * 100, 2),
            "v": round(pv * 100, 2),
            "s": round(ps,       3),
            "w": {tickers[i]: round(float(w[i]), 3) for i in range(n)},
        })
    return ports


# ── 6. Correlation matrix ──────────────────────────────────────────────────────
def correlation_matrix(tickers: list, stocks: dict, n_days: int, dt: float) -> dict:
    """Pearson correlation of simulated daily log-returns."""
    all_lr = {}
    for t, p in stocks.items():
        lr = np.random.normal(
            (p["mu"] - 0.5 * p["sigma"]**2) * dt,
            p["sigma"] * np.sqrt(dt),
            n_days
        )
        all_lr[t] = lr

    corr = np.corrcoef([all_lr[t] for t in tickers])
    return {
        "tickers": tickers,
        "matrix": [[round(float(v), 3) for v in row] for row in corr],
    }


# ── 7. Main pipeline ──────────────────────────────────────────────────────────
def main():
    tickers = list(STOCKS.keys())
    output  = {}

    for ticker, p in STOCKS.items():
        print(f"Processing {ticker}...")
        prices = simulate_gbm(p["mu"], p["sigma"], p["S0"], T, dt)
        metrics = compute_risk_metrics(prices)
        mc_stats, mc_paths = monte_carlo(
            prices[-1], p["mu"], p["sigma"], HORIZON, N_SIM, dt
        )

        # Downsample price series for charting (max 126 points)
        step = max(1, len(prices) // 126)
        price_series = [round(float(x), 4) for x in prices[::step]]

        output[ticker] = {
            "color":        p["color"],
            "S0":           p["S0"],
            "price_series": price_series,
            "last_price":   round(float(prices[-1]), 2),
            **metrics,
            "mc":           mc_stats,
            "mc_paths":     [[round(float(v), 2) for v in col]
                             for col in mc_paths],
        }

    result = {
        "stocks":   output,
        "corr":     correlation_matrix(tickers, STOCKS, T, dt),
        "frontier": efficient_frontier(tickers, STOCKS),
    }

    with open("stock_data.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n✓ stock_data.json written")
    print("\nSummary:")
    for t, d in output.items():
        print(f"  {t:10s}  ret={d['ann_return']:6.1f}%  "
              f"vol={d['ann_vol']:5.1f}%  sharpe={d['sharpe']:.3f}")


if __name__ == "__main__":
    main()
