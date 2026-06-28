# TradingView Strategy Risk Lab

Model-training lab for TradingView Strategy Tester exports. It trains a bad-trade risk model, exposes a Litestar API, and defines the SHAP/DiCE boundary for explaining why a trade setup looks fragile.

```mermaid
flowchart LR
    A[TradingView trades CSV] --> B[Pydantic trade validation]
    B --> C[Feature builder]
    C --> D[RandomForest risk model]
    D --> E[Holdout ROC-AUC]
    D --> F[SHAP global drivers]
    D --> G[DiCE counterfactual trade conditions]
    E --> H[Strategy risk report]
    F --> H
    G --> H
```

## What It Solves

TradingView exports show trades, but they do not explain which conditions make a strategy fragile. This repo converts trade history into a risk model that can be inspected before a strategy is trusted.

## Sample Output

The committed sample report is generated from Kaggle AAPL market data converted into a
TradingView-style trade export.

![Strategy risk report preview](docs/assets/strategy-risk-report-preview.png)

- [PDF report](docs/samples/strategy_risk_report.pdf)
- [Prepared trade export](examples/trades.csv)

## Stack

Litestar, Pydantic v2, pandas, scikit-learn, SHAP, DiCE, Typer, pytest, ruff.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
tv-risk-prepare-kaggle --out examples/trades.csv
python -m pytest
python -m ruff check .
tv-risk-lab train
```

Run the API:

```bash
uvicorn tradingview_strategy_risk_lab.api:app --reload
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/train` | Train a strategy-risk model from a local CSV path |
| `GET /v1/explainability-stack` | Show how SHAP and DiCE are used |
| `GET /health` | Health check |

## Notes

This is a strategy review tool, not an execution system. The public sample is prepared from
Kaggle AAPL OHLC data. The model uses only setup fields known before exit; realized PnL is used
only as the label.

See [Leakage Policy](docs/leakage_policy.md) for the zero-lookahead contract.

## Evidence Contract

- Features are built only from setup fields known before trade exit.
- Realized PnL is used as the label, not as an input feature.
- SHAP explains model drivers; DiCE is constrained to actionable setup fields.
- Holdout metrics and the leakage policy are part of the committed report workflow.
