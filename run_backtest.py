from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from backtester.comparison import (
    ComparisonResult,
    RemoteBacktestError,
    compare_metrics,
    fetch_remote_metrics,
)
from backtester.engine import BacktestConfig, BacktestEngine
from backtester.report import ReportBuilder


def default_company_list_path() -> Path | None:
    candidate = Path('company_list.csv')
    if candidate.exists():
        return candidate
    candidate = Path('backtest_sims/company_list.csv')
    if candidate.exists():
        return candidate
    return None


def read_company_list(path: Path | None) -> List[str]:
    if not path or not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    for column in df.columns:
        if column.lower() in {'ticker', 'symbol', 'company'}:
            return [str(value).strip().upper() for value in df[column].dropna().tolist() if str(value).strip()]
    return []


def collect_available_tickers(data_dir: Path, company_list: Sequence[str]) -> List[str]:
    tickers = set(t.upper() for t in company_list)
    if data_dir.exists():
        for file in data_dir.glob('*.csv'):
            tickers.add(file.stem.upper())
    else:
        data_dir.mkdir(parents=True, exist_ok=True)
    return sorted(tickers)


def parse_ticker_selection(value: str | None, available: Sequence[str]) -> List[str]:
    if value is None:
        return list(available)
    value = value.strip()
    if not value or value.lower() == 'all':
        return list(available)
    if value.startswith('@'):
        path = Path(value[1:])
        if not path.exists():
            raise FileNotFoundError(f"Ticker list file not found: {path}")
        tickers: List[str] = []
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                token = line.strip()
                if token:
                    tickers.append(token.upper())
        return _validate_selection(tickers, available)

    tokens = [token.strip().upper() for token in value.replace('\n', ',').replace('\r', ',').split(',') if token.strip()]
    return _validate_selection(tokens, available)


def _validate_selection(selection: Iterable[str], available: Sequence[str]) -> List[str]:
    available_set = set(t.upper() for t in available)
    missing = [ticker for ticker in selection if ticker.upper() not in available_set]
    if missing:
        raise ValueError(
            "The following tickers are not available in company list or price cache: " + ', '.join(missing)
        )
    unique_ordered = []
    seen = set()
    for ticker in selection:
        ticker = ticker.upper()
        if ticker not in seen:
            seen.add(ticker)
            unique_ordered.append(ticker)
    if not unique_ordered:
        raise ValueError("No valid tickers were selected.")
    return unique_ordered


def format_metric_value(key: str, value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    if key == "final_value":
        return f"USD {value:,.2f}"
    if key in {"total_return", "annualized_return", "max_drawdown", "volatility", "win_rate"}:
        return f"{value:.2%}"
    return f"{value:.4f}"


def format_metric_diff(key: str, value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    prefix = "+" if value >= 0 else "-"
    magnitude = abs(value)
    if key == "final_value":
        return f"{prefix}${magnitude:,.2f}"
    if key in {"total_return", "annualized_return", "max_drawdown", "volatility", "win_rate"}:
        return f"{prefix}{magnitude:.2%}"
    return f"{prefix}{magnitude:.4f}"


def safe_console_text(message: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return message.encode(encoding, errors="replace").decode(encoding, errors="replace")


WINDOW_CHOICES = ["custom", "1d", "1w", "1m", "3m", "6m", "1y", "3y", "5y", "10y", "max"]
WINDOW_LABELS = {
    "custom": "Custom",
    "1d": "1 day",
    "1w": "1 week",
    "1m": "1 month",
    "3m": "3 months",
    "6m": "6 months",
    "1y": "1 year",
    "3y": "3 years",
    "5y": "5 years",
    "10y": "10 years",
    "max": "Max available",
}


def resolve_window_dates(window: str, start: str | None, end: str | None) -> tuple[str, str]:
    window = (window or "custom").lower()
    if end is None:
        raise ValueError("End date is required when applying a date window.")
    end_ts = pd.Timestamp(end)

    if window == "custom":
        if not start:
            raise ValueError("Start date is required when window is 'custom'.")
        return pd.Timestamp(start).strftime('%Y-%m-%d'), end_ts.strftime('%Y-%m-%d')

    if window == "1d":
        start_ts = end_ts
    elif window == "1w":
        start_ts = end_ts - pd.Timedelta(days=6)
    elif window == "1m":
        start_ts = end_ts - pd.DateOffset(months=1)
    elif window == "3m":
        start_ts = end_ts - pd.DateOffset(months=3)
    elif window == "6m":
        start_ts = end_ts - pd.DateOffset(months=6)
    elif window == "1y":
        start_ts = end_ts - pd.DateOffset(years=1)
    elif window == "3y":
        start_ts = end_ts - pd.DateOffset(years=3)
    elif window == "5y":
        start_ts = end_ts - pd.DateOffset(years=5)
    elif window == "10y":
        start_ts = end_ts - pd.DateOffset(years=10)
    elif window == "max":
        start_ts = pd.Timestamp('1970-01-01')
    else:
        raise ValueError(f"Unsupported window value: {window}")

    if start_ts > end_ts:
        start_ts = end_ts

    return start_ts.strftime('%Y-%m-%d'), end_ts.strftime('%Y-%m-%d')


def run_interactive_wizard(args, available: Sequence[str]) -> None:
    print("Interactive backtest configuration\n------------------------------")
    print(f"Available tickers: {len(available)} symbols")
    if available:
        preview_list = list(available)[:20]
        preview = ", ".join(preview_list)
        if len(available) > 20:
            preview += ', ...'
        print(f"Example symbols: {preview}")
    print("Enter comma-separated tickers, '@file.txt' to read from file, or press Enter to keep current selection.")
    raw_tickers = input(f"Tickers [{args.tickers}]: ").strip()
    if raw_tickers:
        args.tickers = raw_tickers

    print("Enter dates in YYYY-MM-DD format. Leave blank to keep the existing value.")
    raw_start = input(f"Start date [{args.start}]: ").strip()
    if raw_start:
        args.start = raw_start
    raw_end = input(f"End date [{args.end}]: ").strip()
    if raw_end:
        args.end = raw_end

    options_display = ', '.join(f"{key} ({WINDOW_LABELS[key]})" for key in WINDOW_CHOICES)
    print(f"Available date windows: {options_display}")
    raw_window = input(f"Date window [{args.window}]: ").strip().lower()
    if raw_window:
        if raw_window in WINDOW_CHOICES:
            args.window = raw_window
        else:
            print(f"Unrecognized window '{raw_window}', keeping {args.window}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an equal-weight buy-and-hold backtest.")
    parser.add_argument(
        "--start",
        required=True,
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--tickers",
        default='all',
        help="Ticker selection. Use comma-separated symbols, '@file.txt' to read from file, or 'all' (default).",
    )
    parser.add_argument(
        "--window",
        default='custom',
        choices=WINDOW_CHOICES,
        help="Preset date window (custom, 1d, 1w, 1m, 3m, 6m, 1y, 3y, 5y, 10y, max).",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1_000_000.0,
        help="Initial capital deployed across the portfolio (default: 1,000,000).",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Annualized risk-free rate used for Sharpe/Sortino calculations (default: 0).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path('stocks'),
        help="Directory where price history CSVs are cached or stored.",
    )
    parser.add_argument(
        "--company-list",
        type=Path,
        default=default_company_list_path(),
        help="CSV containing available tickers (column named Company/Ticker/Symbol).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path('reports/AIL_backtest.html'),
        help="Path of the HTML report to write (default: reports/AIL_backtest.html).",
    )
    parser.add_argument(
        "--remote-base",
        default="http://172.23.22.100:7000",
        help="Base URL for the external backtest service (default: http://172.23.22.100:7000).",
    )
    parser.add_argument(
        "--remote-strategy",
        default="BuyAndHold",
        help="Strategy name to request from the external service (default: BuyAndHold).",
    )
    parser.add_argument(
        "--remote-period",
        default="d",
        help="Sampling period to request from the external service (default: d).",
    )
    parser.add_argument(
        "--remote-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for the external backtest request (default: 30).",
    )
    parser.add_argument(
        "--no-remote",
        action='store_true',
        help="Skip calling the external backtest service for comparison.",
    )
    parser.add_argument(
        "--interactive",
        action='store_true',
        help="Launch an interactive prompt to choose tickers and date range.",
    )
    return parser


def resolve_output_path(path: Path) -> Path:
    if path.suffix:
        output_path = path
    else:
        output_path = path / 'AIL_backtest.html'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    company_list = read_company_list(args.company_list)
    available = collect_available_tickers(data_dir, company_list)
    if not available:
        raise RuntimeError(
            "No available tickers found. Provide a company list or populate the data directory with price CSVs."
        )

    if args.interactive:
        run_interactive_wizard(args, available)

    tickers = parse_ticker_selection(args.tickers, available)

    try:
        args.start, args.end = resolve_window_dates(args.window, args.start, args.end)
    except ValueError as exc:
        raise SystemExit(str(exc))

    print(f"Selected {len(tickers)} tickers out of {len(available)} available symbols.")
    print(f"Date range: {args.start} to {args.end} (window: {WINDOW_LABELS.get(args.window, args.window)})")

    config = BacktestConfig(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.initial_capital,
        risk_free_rate=args.risk_free_rate,
        data_dir=str(data_dir),
    )

    engine = BacktestEngine(config)
    result = engine.run()

    comparison_result: Optional[ComparisonResult] = None
    if not args.no_remote:
        try:
            remote_metrics = fetch_remote_metrics(
                result,
                base_url=args.remote_base,
                strategy=args.remote_strategy,
                period=args.remote_period,
                timeout=args.remote_timeout,
            )
            comparison_result = compare_metrics(result, remote_metrics)
            if comparison_result.metrics:
                print("External comparison (local vs remote):")
                for entry in comparison_result.metrics:
                    status = "OK" if entry.within_tolerance else "DIFF"
                    rel_display = f"{entry.rel_diff:.2%}" if entry.rel_diff is not None else "N/A"
                    local_display = format_metric_value(entry.key, entry.local_value)
                    remote_display = format_metric_value(entry.key, entry.remote_value)
                    diff_display = format_metric_diff(entry.key, entry.abs_diff)
                    print(
                        f"  {entry.label}: local={local_display} remote={remote_display} diff={diff_display} rel={rel_display} [{status}]"
                    )
            else:
                print("External comparison available but returned no comparable metrics.")
        except RemoteBacktestError as exc:
            print("External comparison skipped: " + safe_console_text(str(exc)))
        except Exception as exc:
            print("External comparison failed: " + safe_console_text(str(exc)))

    output_path = resolve_output_path(args.output)
    builder = ReportBuilder(result, output_path, comparison=comparison_result)
    builder.build()

    metrics = result.metrics
    sortino_display = f"{metrics.sortino_ratio:.2f}" if math.isfinite(metrics.sortino_ratio) else "N/A"
    calmar_display = f"{metrics.calmar_ratio:.2f}" if math.isfinite(metrics.calmar_ratio) else "N/A"
    final_value_display = f"USD {metrics.final_value:,.2f}"
    total_return_display = f"{metrics.total_return:.2%}"
    annualized_display = f"{metrics.annualized_return:.2%}"
    max_drawdown_display = f"{metrics.max_drawdown:.2%}"
    sharpe_display = f"{metrics.sharpe_ratio:.2f}"
    volatility_display = f"{metrics.volatility:.2%}"
    avg_daily_display = f"{metrics.average_daily_return:.3%}"
    best_day_display = f"{metrics.best_day:.2%}"
    worst_day_display = f"{metrics.worst_day:.2%}"
    win_rate_display = f"{metrics.win_rate:.2%}"

    print("Backtest completed. Key metrics:")
    print(f"  Final value: {final_value_display}")
    print(f"  Total return: {total_return_display}")
    print(f"  Annualized return: {annualized_display}")
    print(f"  Max drawdown: {max_drawdown_display}")
    print(f"  Volatility: {volatility_display}")
    print(f"  Sharpe ratio: {sharpe_display}")
    print(f"  Sortino ratio: {sortino_display}")
    print(f"  Calmar ratio: {calmar_display}")
    print(f"  Average daily return: {avg_daily_display}")
    print(f"  Best day: {best_day_display}")
    print(f"  Worst day: {worst_day_display}")
    print(f"  Win rate: {win_rate_display}")

    if metrics.recovery_time_trading_days:
        trading_days = metrics.recovery_time_trading_days
        calendar_days = metrics.recovery_time_calendar_days
        print(f"  Recovery time: {trading_days} trading days ({calendar_days} calendar days)")
    else:
        print("  Recovery time: Not recovered during test window")

    print(f"Report saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
