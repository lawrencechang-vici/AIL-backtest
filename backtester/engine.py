from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestConfig:
    tickers: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 1_000_000.0
    risk_free_rate: float = 0.0  # Annualized rate, e.g. 0.02 for 2%
    weights: Optional[Dict[str, float]] = None
    data_dir: Optional[str] = None  # Folder used to cache historical prices


@dataclass
class BacktestMetrics:
    final_value: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    volatility: float
    average_daily_return: float
    best_day: float
    worst_day: float
    win_rate: float
    recovery_time_trading_days: Optional[int]
    recovery_time_calendar_days: Optional[int]


@dataclass
class BacktestResult:
    config: BacktestConfig
    metrics: BacktestMetrics
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    daily_returns: pd.Series
    asset_prices: pd.DataFrame
    asset_returns: pd.DataFrame
    asset_weights: pd.Series


class BacktestEngine:
    def __init__(self, config: BacktestConfig) -> None:
        if not config.tickers:
            raise ValueError("At least one ticker is required")

        self.config = config
        self.data_dir = Path(config.data_dir or "data/prices")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.start_ts = pd.to_datetime(config.start_date).tz_localize(None)
        self.end_ts = pd.to_datetime(config.end_date).tz_localize(None)
        if self.end_ts <= self.start_ts:
            raise ValueError("End date must be after start date")

    def run(self) -> BacktestResult:
        prices = self._download_price_history()
        weights = self._resolve_weights(prices.columns)
        equity_curve = self._build_equity_curve(prices, weights)
        daily_returns = equity_curve.pct_change().dropna()
        drawdown_curve = self._compute_drawdown_curve(equity_curve)
        metrics = self._compute_metrics(equity_curve, daily_returns, drawdown_curve)
        asset_returns = prices.pct_change().dropna()

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            daily_returns=daily_returns,
            asset_prices=prices,
            asset_returns=asset_returns,
            asset_weights=weights,
        )

    def _download_price_history(self) -> pd.DataFrame:
        tickers = list(dict.fromkeys(self.config.tickers))
        series: List[pd.Series] = []
        for ticker in tickers:
            series.append(self._load_single_ticker_series(ticker))

        prices = pd.concat(series, axis=1)
        prices = prices.sort_index().ffill().dropna(how="any")

        if prices.empty:
            raise RuntimeError("Price data became empty after cleaning. Consider a different period or ticker set.")

        return prices

    def _load_single_ticker_series(self, ticker: str) -> pd.Series:
        ticker = ticker.upper()
        cache_path = self._cache_path_for(ticker)

        cached = self._read_cached_prices(cache_path)
        cache_missing = cached.empty

        start_needed = self.start_ts
        end_needed = self.end_ts

        if not cache_missing:
            cache_start = cached.index.min()
            cache_end = cached.index.max()
            cache_missing = start_needed < cache_start or end_needed > cache_end

        if cache_missing:
            fetch_start = start_needed if cached.empty else min(start_needed, cached.index.min())
            fetch_end = end_needed if cached.empty else max(end_needed, cached.index.max())
            fetched = self._download_from_source(ticker, fetch_start, fetch_end)
            if cached.empty:
                combined = fetched
            else:
                combined = pd.concat([cached, fetched])
                combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            self._write_cache(cache_path, combined)
            cached = combined

        sliced = cached.loc[(cached.index >= start_needed) & (cached.index <= end_needed)].copy()
        if sliced.empty:
            raise RuntimeError(f"No cached prices available for {ticker} covering {self.config.start_date} to {self.config.end_date}.")

        sliced.name = ticker
        return sliced

    def _read_cached_prices(self, cache_path: Path) -> pd.Series:
        if not cache_path.exists():
            return pd.Series(dtype=float)

        df = pd.read_csv(cache_path)
        if df.empty:
            return pd.Series(dtype=float)

        if "date" in df.columns:
            date_col = "date"
        elif "Date" in df.columns:
            date_col = "Date"
        else:
            date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        candidate_cols = [
            col
            for col in df.columns
            if col.lower() in {"price", "close", "adj close", "adjusted close"}
        ]
        price_col = candidate_cols[0] if candidate_cols else df.columns[0]

        series = df[price_col].astype(float)
        series.index = series.index.tz_localize(None)
        series = series[~series.index.duplicated(keep="last")]
        series = series.dropna()
        if series.empty:
            return pd.Series(dtype=float)
        series.name = cache_path.stem.upper()
        return series.sort_index()

    def _write_cache(self, cache_path: Path, series: pd.Series | pd.DataFrame) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(series, pd.DataFrame):
            if "price" in series.columns:
                data = series["price"]
            else:
                data = series.iloc[:, 0]
        else:
            data = series
        df = data.sort_index().to_frame(name="price")
        df.index.name = "date"
        df.to_csv(cache_path)

    def _download_from_source(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        start_str = start.strftime("%Y-%m-%d")
        # yfinance treats the end date as exclusive; add one day to make it inclusive in practice
        end_str = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        data = yf.download(ticker, start=start_str, end=end_str, auto_adjust=True, progress=False)
        if data.empty:
            raise RuntimeError(f"No data returned for {ticker} when requesting {start_str} to {end_str}.")

        if "Adj Close" in data.columns:
            prices = data["Adj Close"]
        elif "Close" in data.columns:
            prices = data["Close"]
        else:
            raise RuntimeError(f"Ticker {ticker} missing Close/Adj Close columns in downloaded data.")

        if isinstance(prices, pd.DataFrame):
            if ticker in prices.columns:
                prices = prices[ticker]
            else:
                prices = prices.iloc[:, 0]

        prices.index = prices.index.tz_localize(None)
        prices = prices.astype(float)
        prices.name = ticker
        return prices

    def _cache_path_for(self, ticker: str) -> Path:
        filename = f"{ticker.upper()}.csv"
        return self.data_dir / filename

    def _resolve_weights(self, assets: pd.Index) -> pd.Series:
        if self.config.weights:
            weights = pd.Series(self.config.weights, dtype=float)
            weights.index = weights.index.str.upper()
            missing = [asset for asset in assets if asset not in weights.index]
            if missing:
                raise ValueError(f"Weights missing for tickers: {missing}")
            weights = weights.reindex(assets)
        else:
            weights = pd.Series(1.0 / len(assets), index=assets)

        if not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()
        return weights

    def _build_equity_curve(self, prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
        normalized = prices / prices.iloc[0]
        portfolio_index = (normalized * weights).sum(axis=1)
        return portfolio_index * self.config.initial_capital

    @staticmethod
    def _compute_drawdown_curve(equity_curve: pd.Series) -> pd.Series:
        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1.0
        return drawdown.fillna(0.0)

    def _compute_metrics(
        self,
        equity_curve: pd.Series,
        daily_returns: pd.Series,
        drawdown_curve: pd.Series,
    ) -> BacktestMetrics:
        final_value = float(equity_curve.iloc[-1])
        total_return = final_value / self.config.initial_capital - 1.0
        trading_days = len(equity_curve)
        years = trading_days / TRADING_DAYS_PER_YEAR
        annualized_return = self._annualize_return(total_return, years)
        max_drawdown = abs(float(drawdown_curve.min()))
        sharpe = self._sharpe_ratio(daily_returns)
        sortino = self._sortino_ratio(daily_returns)
        calmar = annualized_return / max_drawdown if max_drawdown > 0 else np.nan
        volatility = daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) if not daily_returns.empty else 0.0
        avg_daily = daily_returns.mean() if not daily_returns.empty else 0.0
        best_day = daily_returns.max() if not daily_returns.empty else 0.0
        worst_day = daily_returns.min() if not daily_returns.empty else 0.0
        win_rate = float((daily_returns > 0).mean()) if not daily_returns.empty else 0.0
        recovery_trading, recovery_calendar = self._recovery_time(equity_curve)

        return BacktestMetrics(
            final_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            volatility=volatility,
            average_daily_return=avg_daily,
            best_day=best_day,
            worst_day=worst_day,
            win_rate=win_rate,
            recovery_time_trading_days=recovery_trading,
            recovery_time_calendar_days=recovery_calendar,
        )

    def _annualize_return(self, total_return: float, years: float) -> float:
        if years <= 0:
            return 0.0
        return (1.0 + total_return) ** (1.0 / years) - 1.0

    def _sharpe_ratio(self, daily_returns: pd.Series) -> float:
        if daily_returns.empty:
            return 0.0
        mean = daily_returns.mean()
        std = daily_returns.std(ddof=1)
        if std == 0:
            return 0.0
        return (mean * TRADING_DAYS_PER_YEAR) / (std * np.sqrt(TRADING_DAYS_PER_YEAR))

    def _sortino_ratio(self, daily_returns: pd.Series) -> float:
        if daily_returns.empty:
            return 0.0
        rf_daily = (1.0 + self.config.risk_free_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
        excess = daily_returns - rf_daily
        downside = excess[excess < 0]
        if downside.empty:
            return float("inf")
        downside_std = downside.std(ddof=1)
        if downside_std == 0:
            return float("inf")
        return excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)

    def _recovery_time(self, equity_curve: pd.Series) -> Tuple[Optional[int], Optional[int]]:
        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1.0
        trough_date = drawdown.idxmin()
        if pd.isna(trough_date):
            return None, None
        peak_value = running_max.loc[trough_date]
        trough_loc = equity_curve.index.get_loc(trough_date)
        if isinstance(trough_loc, slice):
            trough_loc = trough_loc.stop - 1
        post_trough = equity_curve.iloc[trough_loc + 1 :]
        recovery = post_trough[post_trough >= peak_value]
        if recovery.empty:
            return None, None
        recovery_date = recovery.index[0]
        trading_days = len(
            equity_curve.loc[(equity_curve.index > trough_date) & (equity_curve.index <= recovery_date)]
        )
        calendar_days = (recovery_date - trough_date).days
        return trading_days if trading_days > 0 else None, calendar_days if calendar_days > 0 else None
