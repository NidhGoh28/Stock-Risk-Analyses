# 📊 Stock Risk & Portfolio Analyser

> Quantitative finance tool built with Python and an interactive HTML dashboard — Monte Carlo simulation, Value at Risk, and Markowitz Portfolio Optimisation.

**Live demo:** [View Dashboard](https://nidhgoh28.github.io/Stock-Risk-Analyses/stock_risk_analyser.html)

---

## What it does

This tool models equity risk and optimises portfolio allocation using core quantitative finance techniques:

| Model | Description |
|-------|-------------|
| Geometric Brownian Motion | Simulates 1-year price paths using exact GBM discretisation |
| Monte Carlo Simulation | Projects 500 price paths 90 days forward per asset |
| Value at Risk (VaR) | Computes 1-day VaR at 95th and 99th percentile |
| Conditional VaR (CVaR) | Expected Shortfall — average loss in worst 5% of days |
| Maximum Drawdown | Peak-to-trough decline across simulated history |
| Sharpe Ratio | Risk-adjusted return: E[R] / σ |
| Markowitz Efficient Frontier | 1,800 random portfolios mapped to identify max-Sharpe allocation |
| Pearson Correlation Matrix | Pairwise correlation of daily log-returns for diversification analysis |

---

## Assets covered

AAPL · GOOGL · TSLA · JPM · BTC-USD

---

## Tech stack

- **Python** — NumPy, SciPy (core computation & simulation)
- **Stochastic calculus** — GBM model for price dynamics
- **HTML / CSS / JavaScript** — interactive dashboard
- **Chart.js** — all visualisations (line, bar, scatter)
- **JSON pipeline** — Python generates data, inlined into dashboard

---

## How to run

```bash
# Generate fresh simulation data
python stock_analyser.py

# Then open stock_risk_analyser.html in any browser
```

No dependencies to install beyond NumPy and SciPy.

---

## Dashboard tabs

- **Overview** — 1-year simulated price history + key stats
- **Risk Metrics** — VaR, CVaR, drawdown, skewness, kurtosis
- **Monte Carlo** — 40 visualised paths + bear/bull price distribution
- **Portfolio & Frontier** — efficient frontier scatter + max-Sharpe weights
- **Correlation** — heatmap of pairwise return correlation
- **How It Works** — models explained + interview talking points

---

## Author

**Nidhi Gohel** — B.Sc. Applied Mathematics, York University (Dec 2026)  
[LinkedIn](https://linkedin.com/in/nidhi28) · [GitHub](https://github.com/NidhGoh28)
