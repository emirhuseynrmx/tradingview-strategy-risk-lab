# Leakage Policy

This repo treats lookahead bias as a hard failure.

The model may use only setup fields known before the trade exits:

- `side`
- `atr_pct`
- `volume_z`
- `trend_score`

The model must not use fields created after the trade outcome is known:

- `exit_price`
- `pnl_r`
- realized return
- max favorable excursion after entry
- max adverse excursion after entry
- any manually labeled win/loss reason

`pnl_r` is used only to build the supervised label. It is never included in
`FEATURE_COLUMNS`.

The report is a completed-trade risk audit. It is not a live signal engine and it does not claim
future profitability.
