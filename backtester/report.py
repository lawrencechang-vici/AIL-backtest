from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .comparison import ComparisonResult
from .engine import BacktestResult, TRADING_DAYS_PER_YEAR


class ReportBuilder:
    def __init__(
        self,
        result: BacktestResult,
        output_path: Path,
        comparison: Optional[ComparisonResult] = None,
    ) -> None:
        self.result = result
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.comparison = comparison
        self.asset_metrics = self._compute_asset_metrics()

    def build(self) -> None:
        html = self.render_html()
        self.output_path.write_text(html, encoding="utf-8")

    def render_html(self) -> str:
        charts = self._build_interactive_charts()
        return self._render_html(charts)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _compute_asset_metrics(self) -> pd.DataFrame:
        prices = self.result.asset_prices
        returns = self.result.asset_returns
        weights = self.result.asset_weights.reindex(prices.columns).fillna(0.0)

        trading_days = len(prices)
        years = trading_days / TRADING_DAYS_PER_YEAR if trading_days else 0.0

        total_return = prices.iloc[-1] / prices.iloc[0] - 1.0
        annualized = (1.0 + total_return).pow(1.0 / years) - 1.0 if years > 0 else pd.Series(0.0, index=total_return.index)
        avg_daily = returns.mean()
        volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = (avg_daily / returns.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = sharpe.replace([np.inf, -np.inf], np.nan)
        downside = returns.where(returns < 0.0).std(ddof=1)
        sortino = (avg_daily / downside) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sortino = sortino.replace([np.inf, -np.inf], np.nan)
        contribution = weights * total_return

        metrics = pd.DataFrame(
            {
                "Ticker": total_return.index,
                "Weight": weights.values,
                "Total Return": total_return.values,
                "Annualized Return": annualized.values,
                "Volatility": volatility.values,
                "Sharpe": sharpe.values,
                "Sortino": sortino.values,
                "Avg Daily Return": avg_daily.values,
                "Contribution": contribution.values,
            }
        )
        metrics = metrics.replace([np.inf, -np.inf], np.nan)
        metrics = metrics.sort_values(by="Contribution", ascending=False).reset_index(drop=True)
        return metrics

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    def _apply_chart_theme(self, fig: go.Figure) -> go.Figure:
        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font=dict(color="#e0e6ed"),
        )
        fig.update_xaxes(
            color="#e0e6ed",
            gridcolor="rgba(255, 255, 255, 0.08)",
            zerolinecolor="rgba(255, 255, 255, 0.15)",
        )
        fig.update_yaxes(
            color="#e0e6ed",
            gridcolor="rgba(255, 255, 255, 0.08)",
            zerolinecolor="rgba(255, 255, 255, 0.15)",
        )
        return fig


    def _build_interactive_charts(self) -> List[Tuple[str, str]]:
        figures = [
            ("Equity Curve", self._equity_curve_figure()),
            ("Drawdown", self._drawdown_figure()),
            ("Daily Return Distribution", self._return_distribution_figure()),
            ("Rolling 6M Volatility", self._rolling_volatility_figure()),
            ("Rolling 6M Sharpe", self._rolling_sharpe_figure()),
            ("Monthly Return Heatmap", self._monthly_heatmap()),
            ("Total Return by Asset", self._total_return_figure()),
            ("Asset Sharpe Ratios", self._asset_sharpe_figure()),
        ]

        if self.result.asset_returns.shape[1] > 1:
            figures.append(("Asset Return Correlation", self._correlation_heatmap()))

        charts: List[Tuple[str, str]] = []
        plotlyjs = "cdn"
        for title, fig in figures:
            html = fig.to_html(full_html=False, include_plotlyjs=plotlyjs, default_height="440px")
            charts.append((title, html))
            plotlyjs = "none"
        return charts

    def _equity_curve_figure(self) -> go.Figure:
        equity = self.result.equity_curve
        fig = go.Figure(
            go.Scatter(x=equity.index, y=equity.values, mode="lines", line=dict(color="#3a80e9", width=2))
        )
        fig.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Equity (USD)",
            hovermode="x unified",
        )
        return self._apply_chart_theme(fig)

    def _drawdown_figure(self) -> go.Figure:
        drawdown = self.result.drawdown_curve
        fig = go.Figure(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                line=dict(color="#e03f6a"),
                fill="tozeroy",
            )
        )
        fig.update_layout(
            title="Drawdown Curve",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            hovermode="x unified",
            yaxis=dict(tickformat=".0%"),
        )
        return self._apply_chart_theme(fig)

    def _return_distribution_figure(self) -> go.Figure:
        returns = self.result.daily_returns
        fig = go.Figure(go.Histogram(x=returns.values, nbinsx=60, marker_color="#3ac779"))
        fig.update_layout(
            title="Daily Return Distribution",
            xaxis_title="Daily Return",
            yaxis_title="Frequency",
        )
        return self._apply_chart_theme(fig)

    def _rolling_volatility_figure(self) -> go.Figure:
        window = 126
        rolling_vol = self.result.daily_returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        rolling_vol = rolling_vol.dropna()
        fig = go.Figure(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values, mode="lines", line=dict(color="#ff7f0e", width=2))
        )
        fig.update_layout(
            title=f"Rolling {window//21}M Annualized Volatility",
            xaxis_title="Date",
            yaxis_title="Volatility",
            hovermode="x unified",
            yaxis=dict(tickformat=".1%"),
        )
        return self._apply_chart_theme(fig)

    def _rolling_sharpe_figure(self) -> go.Figure:
        window = 126
        rolling_mean = self.result.daily_returns.rolling(window).mean()
        rolling_std = self.result.daily_returns.rolling(window).std(ddof=1)
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
        rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan).dropna()
        fig = go.Figure(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, mode="lines", line=dict(color="#9467bd", width=2))
        )
        fig.update_layout(
            title=f"Rolling {window//21}M Sharpe Ratio",
            xaxis_title="Date",
            yaxis_title="Sharpe",
            hovermode="x unified",
        )
        return self._apply_chart_theme(fig)

    def _monthly_heatmap(self) -> go.Figure:
        returns = self.result.daily_returns
        monthly = (1.0 + returns).resample("ME").prod() - 1.0
        if monthly.empty:
            monthly = pd.Series(dtype=float)
        df = monthly.to_frame("return")
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title="Monthly Return Heatmap")
            return fig
        df["Year"] = df.index.year
        df["Month"] = df.index.strftime("%b")
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        df["Month"] = pd.Categorical(df["Month"], categories=month_order, ordered=True)
        pivot = df.pivot(index="Year", columns="Month", values="return").sort_index()
        fig = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="RdYlGn",
                zmid=0,
                colorbar=dict(title="Return", ticksuffix="%", tickformat=".1%"),
            )
        )
        fig.update_layout(
            title="Monthly Return Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
        )
        return self._apply_chart_theme(fig)

    def _total_return_figure(self) -> go.Figure:
        total_returns = self.asset_metrics[["Ticker", "Total Return"]].set_index("Ticker")["Total Return"].sort_values()
        fig = go.Figure(go.Bar(x=total_returns.index, y=total_returns.values, marker_color="#2ca02c"))
        fig.update_layout(
            title="Total Return by Asset",
            xaxis_title="Ticker",
            yaxis_title="Return",
            hovermode="x",
            yaxis=dict(tickformat=".1%"),
        )
        return self._apply_chart_theme(fig)

    def _asset_sharpe_figure(self) -> go.Figure:
        sharpe = self.asset_metrics[["Ticker", "Sharpe"]].set_index("Ticker")["Sharpe"].sort_values()
        fig = go.Figure(go.Bar(x=sharpe.index, y=sharpe.values, marker_color="#ff7f0e"))
        fig.update_layout(
            title="Asset Sharpe Ratios",
            xaxis_title="Ticker",
            yaxis_title="Sharpe",
            hovermode="x",
        )
        return self._apply_chart_theme(fig)

    def _correlation_heatmap(self) -> go.Figure:
        returns = self.result.asset_returns
        if returns.empty or returns.shape[1] < 2:
            fig = go.Figure()
            fig.update_layout(title="Asset Return Correlation")
            return fig

        top_assets = self.asset_metrics.head(25)["Ticker"].tolist()
        subset = returns[top_assets].corr()
        fig = go.Figure(
            go.Heatmap(
                z=subset.values,
                x=subset.columns,
                y=subset.index,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                colorbar=dict(title="Correlation"),
            )
        )
        fig.update_layout(
            title="Asset Return Correlation (Top 25 by Contribution)",
        )
        return self._apply_chart_theme(fig)

    # ------------------------------------------------------------------
    # HTML rendering
    # ------------------------------------------------------------------
    def _render_html(self, charts: List[Tuple[str, str]]) -> str:
        cfg = self.result.config
        metrics = self.result.metrics

        summary_cards = [
            ("Final Value", f"${metrics.final_value:,.2f}"),
            ("Total Return", self._fmt_pct(metrics.total_return)),
            ("Annualized Return", self._fmt_pct(metrics.annualized_return)),
            ("Max Drawdown", self._fmt_pct(metrics.max_drawdown)),
        ]
        risk_cards = [
            ("Sharpe Ratio", self._fmt_num(metrics.sharpe_ratio)),
            ("Sortino Ratio", self._fmt_num(metrics.sortino_ratio)),
            ("Calmar Ratio", self._fmt_num(metrics.calmar_ratio)),
            ("Volatility", self._fmt_pct(metrics.volatility)),
            ("Average Daily Return", self._fmt_pct(metrics.average_daily_return, digits=3)),
            ("Best Day", self._fmt_pct(metrics.best_day)),
            ("Worst Day", self._fmt_pct(metrics.worst_day)),
            ("Win Rate", self._fmt_pct(metrics.win_rate)),
            (
                "Recovery Time",
                (
                    f"{metrics.recovery_time_trading_days} trading days / {metrics.recovery_time_calendar_days} calendar days"
                    if metrics.recovery_time_trading_days and metrics.recovery_time_calendar_days
                    else "Not recovered"
                ),
            ),
        ]

        top_assets = self.asset_metrics.head(10)
        bottom_assets = self.asset_metrics.tail(10)

        def _cards_html(cards: List[Tuple[str, str]]) -> str:
            parts = []
            for title, value in cards:
                parts.append(
                    f"<div class='card'><p class='card-label'>{title}</p><p class='card-value'>{value}</p></div>"
                )
            return "".join(parts)

        def _asset_table(df: pd.DataFrame) -> str:
            header = "".join(
                "<th>" + col + "</th>" for col in [
                    "Ticker",
                    "Weight",
                    "Total Return",
                    "Annualized Return",
                    "Volatility",
                    "Sharpe",
                    "Sortino",
                    "Avg Daily Return",
                    "Contribution",
                ]
            )

            rows = []
            for _, row in df.iterrows():
                rows.append(
                    "<tr>"
                    + f"<td>{row['Ticker']}</td>"
                    + f"<td>{self._fmt_pct(row['Weight'])}</td>"
                    + f"<td>{self._fmt_pct(row['Total Return'])}</td>"
                    + f"<td>{self._fmt_pct(row['Annualized Return'])}</td>"
                    + f"<td>{self._fmt_pct(row['Volatility'])}</td>"
                    + f"<td>{self._fmt_num(row['Sharpe'])}</td>"
                    + f"<td>{self._fmt_num(row['Sortino'])}</td>"
                    + f"<td>{self._fmt_pct(row['Avg Daily Return'], digits=3)}</td>"
                    + f"<td>{self._fmt_pct(row['Contribution'])}</td>"
                    + "</tr>"
                )
            return "<table><thead><tr>" + header + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"

        comparison_html = self._comparison_section()

        chart_sections = []
        for i in range(0, len(charts), 2):
            group = charts[i : i + 2]
            cards_html = "".join(
                f"<div class='chart-card'><h2>{title}</h2>{html}</div>" for title, html in group
            )
            chart_sections.append(f"<div class='chart-grid'>{cards_html}</div>")

        html_lines = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>AIL Backtest</title>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "<style>",
            self._css(),
            "</style>",
            "</head>",
            "<body>",
            "<div class='container'>",
            "<header>",
            "<p class='label'>AIL Backtesting Suite</p>",
            "<h1>Portfolio Performance Report</h1>",
            f"<p class='meta'>Tickers: {', '.join(cfg.tickers)}</p>",
            f"<p class='meta'>Period: {cfg.start_date} to {cfg.end_date}</p>",
            "</header>",
            f"<section class='summary-grid'>{_cards_html(summary_cards)}</section>",
            f"<section class='summary-grid secondary'>{_cards_html(risk_cards)}</section>",
        ]

        if comparison_html:
            html_lines.append(comparison_html)

        html_lines.extend(
            [
                "<section class='tables'>",
                "<div class='table-wrapper'>",
                "<h2>Top Contributors (by total return * weight)</h2>",
                _asset_table(top_assets),
                "</div>",
                "<div class='table-wrapper'>",
                "<h2>Bottom Contributors</h2>",
                _asset_table(bottom_assets.sort_values(by='Contribution')),
                "</div>",
                "</section>",
            ]
        )

        html_lines.extend(chart_sections)

        html_lines.extend(
            [
                "</div>",
                "</body>",
                "</html>",
            ]
        )
        return "\n".join(html_lines)

    def _comparison_section(self) -> Optional[str]:
        if not self.comparison:
            return None
        comp = self.comparison
        rows = []
        for entry in comp.metrics:
            rows.append(
                "<tr>"
                + f"<td>{entry.label}</td>"
                + f"<td>{self._format_metric(entry.key, entry.local_value)}</td>"
                + f"<td>{self._format_metric(entry.key, entry.remote_value)}</td>"
                + f"<td>{self._format_metric(entry.key, entry.abs_diff)}</td>"
                + f"<td>{self._fmt_pct(entry.rel_diff) if entry.rel_diff is not None else 'N/A'}</td>"
                + f"<td class='status {'ok' if entry.within_tolerance else 'warn'}'>{'Within tolerance' if entry.within_tolerance else 'Diff exceeds tolerance'}</td>"
                + "</tr>"
            )

        note_lines: List[str] = []
        if comp.remote.failed_tickers:
            errors = comp.remote.failed_ticker_errors or {}
            items = []
            for ticker in comp.remote.failed_tickers:
                message = errors.get(ticker, "No diagnostic provided")
                items.append(f"<li><strong>{ticker}</strong>: {message}</li>")
            note_lines.append(
                "<div class='comparison-note'><p>Remote service reported issues for these tickers:</p>"
                + f"<ul>{''.join(items)}</ul></div>"
            )

        table_html = (
            "<section class='comparison'>"
            "<h2>External Backtest Comparison</h2>"
            f"<p class='comparison-tolerance'>Tolerance: absolute ≤ {comp.abs_tolerance:.2e}, relative ≤ {comp.rel_tolerance:.2%}</p>"
            + "".join(note_lines)
            + "<div class='comparison-table-wrapper'>"
            + "<table class='comparison-table'>"
            + "<thead><tr><th>Metric</th><th>Local</th><th>Remote</th><th>Abs. diff</th><th>Rel. diff</th><th>Status</th></tr></thead>"
            + "<tbody>" + "".join(rows) + "</tbody></table></div></section>"
        )
        return table_html

    def _format_metric(self, key: str, value: float) -> str:
        if key == "final_value":
            return self._fmt_currency(value)
        if key in {"total_return", "annualized_return", "max_drawdown", "volatility", "win_rate"}:
            return self._fmt_pct(value)
        return self._fmt_num(value)

    @staticmethod
    def _fmt_pct(value: Optional[float], digits: int = 2) -> str:
        if value is None or not math.isfinite(value):
            return "N/A"
        return f"{value:.{digits}%}"

    @staticmethod
    def _fmt_num(value: Optional[float], digits: int = 2) -> str:
        if value is None or not math.isfinite(value):
            return "N/A"
        return f"{value:.{digits}f}"

    @staticmethod
    def _fmt_currency(value: Optional[float]) -> str:
        if value is None or not math.isfinite(value):
            return "N/A"
        return f"${value:,.2f}"

    def _css(self) -> str:
        return """
        :root {
            color-scheme: dark;
            --bg: #0b1622;
            --panel: #131d2b;
            --accent: #3a80e9;
            --accent-muted: rgba(58, 128, 233, 0.2);
            --text-primary: #e0e6ed;
            --text-secondary: #7f8ea3;
            --positive: #3ac779;
            --negative: #e03f6a;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg);
            color: var(--text-primary);
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 28px 60px;
        }
        header {
            margin-bottom: 28px;
        }
        header h1 {
            margin: 6px 0 10px;
            font-size: 2.1rem;
            font-weight: 600;
        }
        .label {
            text-transform: uppercase;
            letter-spacing: 0.2rem;
            color: var(--accent);
            font-size: 0.85rem;
        }
        .meta {
            margin: 0;
            color: var(--text-secondary);
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }
        .summary-grid.secondary {
            margin-bottom: 32px;
        }
        .card {
            background: var(--panel);
            border-radius: 18px;
            padding: 20px 22px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.25);
        }
        .card-label {
            text-transform: uppercase;
            letter-spacing: 0.08rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin: 0 0 6px;
        }
        .card-value {
            margin: 0;
            font-size: 1.45rem;
            font-weight: 600;
        }
        .comparison {
            background: var(--panel);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 18px;
            padding: 24px 26px;
            margin-bottom: 36px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.22);
        }
        .comparison h2 {
            margin-top: 0;
        }
        .comparison-tolerance {
            margin-top: 6px;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        .comparison-note {
            background: rgba(224, 63, 106, 0.12);
            border-left: 3px solid var(--negative);
            padding: 12px 16px;
            border-radius: 10px;
            margin: 18px 0;
        }
        .comparison-note ul {
            margin: 8px 0 0;
            padding-left: 18px;
        }
        .comparison-table-wrapper {
            overflow-x: auto;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
        }
        .comparison-table th,
        .comparison-table td {
            padding: 10px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.07);
            text-align: left;
            font-size: 0.9rem;
        }
        .comparison-table th {
            text-transform: uppercase;
            letter-spacing: 0.05rem;
            color: var(--text-secondary);
            font-size: 0.75rem;
        }
        .comparison-table .status.ok {
            color: var(--positive);
            font-weight: 600;
        }
        .comparison-table .status.warn {
            color: var(--negative);
            font-weight: 600;
        }
        .tables {
            display: flex;
            flex-direction: column;
            gap: 32px;
            margin-bottom: 36px;
        }
        .table-wrapper {
            background: var(--panel);
            border-radius: 18px;
            padding: 20px 24px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.18);
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            text-align: left;
            font-size: 0.9rem;
        }
        th {
            text-transform: uppercase;
            letter-spacing: 0.05rem;
            color: var(--text-secondary);
            font-size: 0.75rem;
        }
        tr:last-child td {
            border-bottom: none;
        }
        .chart-grid {
            display: flex;
            flex-direction: column;
            gap: 24px;
            margin-bottom: 32px;
        }
        .chart-card {
            background: var(--panel);
            border-radius: 18px;
            padding: 18px 20px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
        }
        .chart-card h2 {
            margin: 0 0 12px;
            font-size: 1.05rem;
            font-weight: 600;
        }
        @media (max-width: 900px) {
            .chart-grid {
                flex-direction: column;
            }
        }
        """
