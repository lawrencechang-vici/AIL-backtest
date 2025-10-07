from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import math
from pathlib import Path

import requests

from .engine import BacktestResult

DEFAULT_REMOTE_BASE = "http://172.23.22.100:7000"
DEFAULT_STRATEGY = "BuyAndHold"
DEFAULT_PERIOD = "d"


@dataclass
class RemoteBacktestMetrics:
    initial_value: Optional[float]
    final_value: Optional[float]
    total_return: Optional[float]
    annualized_return: Optional[float]
    max_drawdown: Optional[float]
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    calmar_ratio: Optional[float]
    volatility: Optional[float]
    win_rate: Optional[float]
    profit_factor: Optional[float]
    raw: Dict[str, object]
    failed_tickers: Optional[List[str]]
    failed_ticker_errors: Optional[Dict[str, str]]


@dataclass
class MetricComparison:
    key: str
    label: str
    local_value: float
    remote_value: float
    abs_diff: float
    rel_diff: Optional[float]
    within_tolerance: bool


@dataclass
class ComparisonResult:
    remote: RemoteBacktestMetrics
    metrics: List[MetricComparison]
    rel_tolerance: float
    abs_tolerance: float


class RemoteBacktestError(RuntimeError):
    """Raised when the remote service cannot provide comparable results."""


def fetch_remote_metrics(
    result: BacktestResult,
    base_url: str = DEFAULT_REMOTE_BASE,
    strategy: str = DEFAULT_STRATEGY,
    period: str = DEFAULT_PERIOD,
    timeout: float = 30.0,
) -> RemoteBacktestMetrics:
    tickers = ",".join(result.config.tickers)
    if not tickers:
        raise RemoteBacktestError("No tickers supplied for remote comparison.")

    base = base_url.rstrip("/")
    url = f"{base}/run_strategy/{strategy}/{tickers}/{period}"
    params = {
        "start": result.config.start_date,
        "end": result.config.end_date,
    }

    response = requests.put(url, params=params, timeout=timeout)
    if response.status_code != 200:
        raise RemoteBacktestError(f"Remote service returned {response.status_code}: {response.text[:200]}")

    payload = response.json()

    failed = payload.get("failed_tickers") or []
    if failed:
        # If every ticker failed, the payload will carry no useful numbers.
        if payload.get("successful_ticker_count", 0) == 0:
            raise RemoteBacktestError(
                "Remote service could not process any tickers: "
                + ", ".join(failed)
            )

    def _to_float(key: str) -> Optional[float]:
        value = payload.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    max_drawdown = _to_float("max_drawdown")
    if max_drawdown is not None:
        max_drawdown = abs(max_drawdown)
        # Remote API may return percentage values (e.g. 35.5). Normalise to fraction if necessary.
        if max_drawdown > 1.0:
            max_drawdown = max_drawdown / 100.0

    volatility = _to_float("standard_deviation")

    remote_metrics = RemoteBacktestMetrics(
        initial_value=_to_float("initial_value"),
        final_value=_to_float("result_value"),
        total_return=_to_float("total_return"),
        annualized_return=_to_float("annualized_return"),
        max_drawdown=max_drawdown,
        sharpe_ratio=_to_float("sharpe_ratio"),
        sortino_ratio=_to_float("sortino_ratio"),
        calmar_ratio=_to_float("calmar_ratio"),
        volatility=volatility,
        win_rate=_to_float("win_rate"),
        profit_factor=_to_float("profit_factor"),
        raw=payload,
        failed_tickers=failed or None,
        failed_ticker_errors=payload.get("failed_ticker_errors"),
    )
    if (
        (remote_metrics.final_value in (None, 0.0))
        and (remote_metrics.total_return in (None, 0.0))
        and payload.get("successful_ticker_count", 0) < len(result.config.tickers)
    ):
        details = remote_metrics.failed_ticker_errors or {}
        if remote_metrics.failed_tickers:
            messages = [f"{ticker}: {details.get(ticker, 'no diagnostic provided')}" for ticker in remote_metrics.failed_tickers]
            info = '; '.join(messages)
        else:
            info = 'remote service returned zero values without diagnostics.'
        raise RemoteBacktestError(
            "Remote service could not compute usable metrics for the requested window. " + info
        )

    return remote_metrics


def compare_metrics(
    result: BacktestResult,
    remote: RemoteBacktestMetrics,
    rel_tol: float = 1e-2,
    abs_tol: float = 1e-4,
) -> ComparisonResult:
    comparisons: List[MetricComparison] = []

    def add_entry(key: str, label: str, local_value: Optional[float], remote_value: Optional[float]) -> None:
        if local_value is None or remote_value is None:
            return
        abs_diff = float(local_value) - float(remote_value)
        abs_diff_value = abs(abs_diff)
        denom = max(abs(float(remote_value)), abs(float(local_value)), abs_tol)
        rel_diff = abs_diff_value / denom if denom else None
        within = abs_diff_value <= abs_tol or (rel_diff is not None and rel_diff <= rel_tol)
        comparisons.append(
            MetricComparison(
                key=key,
                label=label,
                local_value=float(local_value),
                remote_value=float(remote_value),
                abs_diff=abs_diff,
                rel_diff=rel_diff,
                within_tolerance=within,
            )
        )

    metrics = result.metrics

    add_entry("final_value", "Final Value", metrics.final_value, remote.final_value)
    add_entry("total_return", "Total Return", metrics.total_return, remote.total_return)
    add_entry("annualized_return", "Annualized Return", metrics.annualized_return, remote.annualized_return)
    add_entry("max_drawdown", "Max Drawdown", metrics.max_drawdown, remote.max_drawdown)
    add_entry("sharpe_ratio", "Sharpe Ratio", metrics.sharpe_ratio, remote.sharpe_ratio)
    add_entry("sortino_ratio", "Sortino Ratio", metrics.sortino_ratio, remote.sortino_ratio)
    add_entry("calmar_ratio", "Calmar Ratio", metrics.calmar_ratio, remote.calmar_ratio)
    add_entry("volatility", "Volatility", metrics.volatility, remote.volatility)
    add_entry("win_rate", "Win Rate", metrics.win_rate, remote.win_rate)

    return ComparisonResult(
        remote=remote,
        metrics=comparisons,
        rel_tolerance=rel_tol,
        abs_tolerance=abs_tol,
    )
