# Checkpoint

## Already Working
- Equal-weight backtest via `run_backtest.py` and the Flask web UI, consuming local CSVs under `stocks/`.
- Preset portfolios (Single - NVDA/TSLA/BOOT, Small 5, Large 60, Very Large 120, Company List 70) auto-populate tickers.
- Date window presets (1d, 1w, 1m, etc., max) apply in both CLI and web UI.
- External comparison against the remote strategy service plus dynamic link to open the remote web platform with matching tickers/dates.
- Plotly charts rendered with transparent backgrounds and a single-column layout.

## Planned / Upcoming
- Add a dedicated custom-weight mode/tab in the web UI so users can input `ticker:weight` pairs and run weighted portfolios.
- Update the remote-link helper to ensure it reflects the weighted configuration once the new mode is available.

## Outstanding Issues
- Custom-weight workflow is not implemented yet; current interface only handles equal weights.
- Need to re-run verification after the weighted workflow is complete.
