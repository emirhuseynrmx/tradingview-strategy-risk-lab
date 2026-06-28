from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

DATASET_SLUG = "serkanp/algorithmic-trading-strategy"
RAW_FILENAME = "AAPL.csv"

app = typer.Typer(help="Download Kaggle market data and prepare a strategy-risk trade export.")


def download_market_dataset(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET_SLUG,
            "-p",
            str(data_dir),
            "--unzip",
        ],
        check=True,
    )
    raw_path = data_dir / RAW_FILENAME
    if not raw_path.exists():
        raise FileNotFoundError(f"Kaggle download completed but {RAW_FILENAME} was not found.")
    return raw_path


def build_strategy_risk_export(raw: pd.DataFrame, max_trades: int = 240) -> pd.DataFrame:
    prices = raw.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values("Date").reset_index(drop=True)
    prices["fast_ma"] = prices["Close Price"].rolling(8).mean()
    prices["slow_ma"] = prices["Close Price"].rolling(21).mean()
    prices["atr_pct"] = (
        (prices["High Price"] - prices["Low Price"]) / prices["Close Price"]
    ).fillna(0)
    prices["volume_z"] = (
        (prices["Volume"] - prices["Volume"].rolling(20).mean())
        / prices["Volume"].rolling(20).std()
    ).fillna(0)
    prices["trend_score"] = (
        (prices["fast_ma"] - prices["slow_ma"]) / prices["Close Price"] * 100
    ).fillna(0)

    trades: list[dict[str, object]] = []
    for idx in range(25, len(prices) - 6, 5):
        row = prices.iloc[idx]
        exit_row = prices.iloc[idx + 5]
        side = "long" if row["fast_ma"] >= row["slow_ma"] else "short"
        direction = 1 if side == "long" else -1
        entry = float(row["Close Price"])
        exit_price = float(exit_row["Close Price"])
        pnl_r = ((exit_price - entry) / entry * direction) / max(float(row["atr_pct"]), 0.003)
        trades.append(
            {
                "trade_id": f"AAPL-{idx:04d}",
                "side": side,
                "entry_price": round(entry, 4),
                "exit_price": round(exit_price, 4),
                "pnl_r": round(float(pnl_r), 4),
                "bars_held": 5,
                "atr_pct": round(float(row["atr_pct"]), 5),
                "volume_z": round(float(row["volume_z"]), 4),
                "trend_score": round(float(max(min(row["trend_score"], 5), -5)), 4),
                "session": "us_regular",
            }
        )
        if len(trades) >= max_trades:
            break
    return pd.DataFrame(trades)


def prepare_strategy_risk(raw_path: Path, output_path: Path) -> Path:
    trades = build_strategy_risk_export(pd.read_csv(raw_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_path, index=False, float_format="%.5f")
    return output_path


@app.command()
def prepare(
    data_dir: Annotated[Path, typer.Option(help="Directory for the Kaggle download.")] = Path(
        "data/raw/kaggle/aapl"
    ),
    out: Annotated[Path, typer.Option(help="Prepared strategy-risk CSV path.")] = Path(
        "examples/trades.csv"
    ),
    skip_download: Annotated[
        bool,
        typer.Option(help="Use an existing raw Kaggle CSV in data_dir."),
    ] = False,
) -> None:
    raw_path = data_dir / RAW_FILENAME if skip_download else download_market_dataset(data_dir)
    prepared = prepare_strategy_risk(raw_path, out)
    typer.echo(f"Prepared Kaggle strategy-risk export at {prepared}")
