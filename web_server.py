from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import urlencode
from typing import Sequence

import pandas as pd
from flask import Flask, request

from backtester.comparison import (
    RemoteBacktestError,
    compare_metrics,
    fetch_remote_metrics,
)
from backtester.engine import BacktestConfig, BacktestEngine
from backtester.report import ReportBuilder
from run_backtest import (
    collect_available_tickers,
    default_company_list_path,
    parse_ticker_selection,
    parse_weight_entries,
    read_company_list,
    safe_console_text,
)

DATA_DIR = Path("stocks")
DEFAULT_START = "2017-07-03"
DEFAULT_END = "2024-06-30"
REMOTE_BASE = "http://172.23.22.100:7000"
REMOTE_STRATEGY = "BuyAndHold"
REMOTE_PERIOD = "d"
REMOTE_TIMEOUT = 30.0
COMPANY_LIST_70_PATH = Path('company_list_70.csv')

REMOTE_UI_BASE = "http://172.23.22.100:8499/backtest"

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

app = Flask(__name__)


def _execute_backtest(
    tickers: Sequence[str],
    start: str,
    end: str,
    skip_remote: bool,
    weights: dict[str, float] | None = None,
) -> tuple[str, str | None]:
    config = BacktestConfig(
        tickers=list(tickers),
        start_date=start,
        end_date=end,
        initial_capital=1_000_000.0,
        risk_free_rate=0.0,
        data_dir=str(DATA_DIR),
        weights=weights,
    )

    engine = BacktestEngine(config)
    try:
        result = engine.run()
    except Exception as exc:
        raise RuntimeError(f"Backtest failed: {exc}") from exc

    comparison = None
    remote_message: str | None = None
    if not skip_remote:
        try:
            remote_metrics = fetch_remote_metrics(
                result,
                base_url=REMOTE_BASE,
                strategy=REMOTE_STRATEGY,
                period=REMOTE_PERIOD,
                timeout=REMOTE_TIMEOUT,
            )
            comparison = compare_metrics(result, remote_metrics)
        except RemoteBacktestError as exc:
            remote_message = safe_console_text(str(exc))
        except Exception as exc:  # pragma: no cover - diagnostics surfaced to UI
            remote_message = safe_console_text(f"External comparison failed: {exc}")

    builder = ReportBuilder(result, Path("reports/AIL_backtest.html"), comparison=comparison)
    html_report = builder.render_html()
    builder.output_path.write_text(html_report, encoding="utf-8")

    return html_report, remote_message

EMPTY_REPORT = """<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>AIL Backtest</title>
<style>body { font-family: Arial, sans-serif; background: #0b1622; color: #e0e6ed; margin: 0; } .container { max-width: 960px; margin: 80px auto; padding: 20px; } </style>
</head>
<body>
<div class='container'>
<h1>AIL Backtest</h1>
<p>No report generated yet. Submit the form to run a backtest.</p>
</div>
</body>
</html>"""

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



def load_company_list_70_tickers() -> list[str]:
    if not COMPANY_LIST_70_PATH.exists():
        return []
    try:
        df = pd.read_csv(COMPANY_LIST_70_PATH)
    except Exception:
        return []
    for column in df.columns:
        if column.lower() in {"ticker", "symbol", "company"}:
            tickers = [str(value).strip().upper() for value in df[column].dropna().tolist() if str(value).strip()]
            return tickers
    return []



def build_preset_portfolios(available: Sequence[str]) -> list[tuple[str, str, list[str]]]:
    available_upper = [ticker.upper() for ticker in available]
    available_set = set(available_upper)

    def limit_list(tickers: list[str]) -> list[str]:
        return [ticker for ticker in tickers if ticker in available_set]

    presets: list[tuple[str, str, list[str]]] = []

    for ticker in ["NVDA", "TSLA", "BOOT"]:
        if ticker in available_set:
            presets.append((f"single_{ticker.lower()}", f"Single - {ticker}", [ticker]))

    if len(available_upper) >= 5:
        presets.append(("small", "Small Portfolio (5 tickers)", available_upper[:5]))

    if len(available_upper) >= 55:
        presets.append(("large50", "Large Portfolio (60 tickers)", available_upper[:60]))

    if len(available_upper) >= 110:
        presets.append(("large100", "Very Large Portfolio (120 tickers)", available_upper[:120]))

    list70 = limit_list(load_company_list_70_tickers())
    if list70:
        presets.append(("company_list_70", "Company List 70", list70))

    return presets



CONTROL_CSS = """
    .control-panel {
        background: var(--panel);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 18px;
        padding: 24px 26px;
        margin-bottom: 28px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.22);
    }
    .control-panel h2 {
        margin-top: 0;
        margin-bottom: 16px;
    }
    .control-panel form {
        display: grid;
        gap: 16px;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }
    .control-panel label {
        display: block;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }
    .control-panel input[type="text"],
    .control-panel textarea,
    .control-panel input[type="date"] {
        width: 100%;
        padding: 10px 12px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(0, 0, 0, 0.25);
        color: var(--text-primary);
    }
    .control-panel textarea {
        height: 90px;
        resize: vertical;
    }
    .control-panel .checkbox {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.9rem;
    }
    .control-panel button {
        padding: 12px 20px;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        background: var(--accent);
        color: #fff;
        cursor: pointer;
        transition: background 0.3s ease;
    }
    .control-panel button:hover {
        background: #2d64b9;
    }
    .control-panel .helper {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 4px;
    }
    .mode-tabs {
        display: inline-flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 8px 0 18px;
    }
    .mode-tabs .tab {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 8px 18px;
        border-radius: 999px;
        border: 1px solid rgba(58, 128, 233, 0.35);
        background: rgba(58, 128, 233, 0.12);
        color: #e0e6ed;
        text-decoration: none;
        font-weight: 600;
        transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease;
    }
    .mode-tabs .tab:hover {
        background: rgba(58, 128, 233, 0.25);
        border-color: rgba(58, 128, 233, 0.6);
        color: #ffffff;
    }
    .mode-tabs .tab.active {
        background: var(--accent);
        border-color: var(--accent);
        color: #ffffff;
        cursor: default;
    }
    .control-actions {
        margin-bottom: 18px;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
    }
    .external-link {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 18px;
        border-radius: 12px;
        text-decoration: none;
        font-weight: 600;
        background: rgba(58, 128, 233, 0.2);
        border: 1px solid rgba(58, 128, 233, 0.45);
        color: #e0e6ed;
        transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease;
    }
    .external-link:hover {
        background: rgba(58, 128, 233, 0.35);
        border-color: rgba(58, 128, 233, 0.8);
        color: #ffffff;
    }
    .alert {
        margin-top: 12px;
        padding: 12px 16px;
        border-radius: 10px;
    }
    .alert.error {
        background: rgba(224, 63, 106, 0.18);
        border: 1px solid var(--negative);
    }
    .alert.info {
        background: rgba(58, 128, 233, 0.18);
        border: 1px solid var(--accent);
    }
"""


def _inject_controls(report_html: str, controls: str) -> str:
    if "</style>" in report_html:
        report_html = report_html.replace("</style>", CONTROL_CSS + "\n</style>", 1)
    else:
        addition = f"<head>\n<style>{CONTROL_CSS}</style>"
        report_html = report_html.replace("<head>", addition, 1)
    report_html = report_html.replace("<body>", f"<body>\n{controls}\n", 1)
    return report_html

@app.route('/', methods=['GET'])

def index() -> str:
    company_list = read_company_list(default_company_list_path())
    available = collect_available_tickers(DATA_DIR, company_list)
    presets = build_preset_portfolios(available)
    preset_map = {key: tickers for key, _, tickers in presets}

    mode = request.args.get('mode', 'equal').lower()
    if mode not in {'equal', 'weighted'}:
        mode = 'equal'

    preset_param = request.args.get('preset', 'custom').lower()
    tickers_param = request.args.get('tickers', 'all')
    weights_param = request.args.get('weights', '')
    start_param = request.args.get('start', DEFAULT_START)
    end_param = request.args.get('end', DEFAULT_END)
    window = request.args.get('window', 'custom').lower()
    skip_remote = request.args.get('skip_remote') == 'on'

    if preset_param != 'custom' and preset_param in preset_map:
        preset_tickers = preset_map[preset_param]
        tickers_param = ','.join(preset_tickers)
        if mode == 'weighted' and preset_tickers:
            equal_weight = 1.0 / len(preset_tickers)
            weights_param = '\n'.join(f"{ticker}: {equal_weight:.4f}" for ticker in preset_tickers)
    else:
        preset_param = 'custom'

    error_message: str | None = None
    remote_message: str | None = None
    weights_dict: dict[str, float] | None = None

    start_effective = start_param
    end_effective = end_param

    if not available:
        error_message = safe_console_text(f"No price files found in {DATA_DIR.resolve()}")
        report_html = EMPTY_REPORT
        tickers: list[str] = []
    else:
        try:
            tickers = parse_ticker_selection(tickers_param, available)
        except ValueError as exc:
            error_message = safe_console_text(str(exc))
            tickers = parse_ticker_selection('all', available)
            tickers_param = 'all'
            preset_param = 'custom'

        if mode == 'weighted':
            try:
                weights_dict = parse_weight_entries(weights_param, tickers, available)
                tickers = list(weights_dict.keys())
                weights_param = '\n'.join(f"{ticker}: {weights_dict[ticker]:.6f}" for ticker in tickers)
            except ValueError as exc:
                error_message = safe_console_text(str(exc))
                weights_dict = None

        try:
            start_effective, end_effective = resolve_window_dates(window, start_param, end_param)
        except ValueError as exc:
            error_message = safe_console_text(str(exc))
            window = 'custom'
            start_effective = start_param or DEFAULT_START
            end_effective = end_param or DEFAULT_END

        try:
            report_html, remote_message = _execute_backtest(
                tickers,
                start_effective,
                end_effective,
                skip_remote,
                weights_dict,
            )
        except Exception as exc:
            error_message = safe_console_text(str(exc))
            fallback_path = Path('reports/AIL_backtest.html')
            if fallback_path.exists():
                report_html = fallback_path.read_text(encoding='utf-8')
            else:
                report_html = EMPTY_REPORT
            remote_message = None

    start_display = start_effective or DEFAULT_START
    end_display = end_effective or DEFAULT_END
    checked_attr = 'checked' if skip_remote else ''

    weights_query: str | None = None
    if mode == 'weighted':
        if weights_dict:
            weights_query = ','.join(f"{ticker}:{weights_dict[ticker]:.6f}" for ticker in tickers)
        else:
            raw_tokens = re.split(r'[\s,;]+', weights_param)
            tokens = [
                entry.strip().replace(' ', '')
                for entry in raw_tokens
                if entry.strip()
            ]
            if tokens:
                weights_query = ','.join(tokens)

    remote_params: dict[str, str] = {}
    if tickers:
        remote_params['tickers'] = ','.join(tickers)
    if start_display:
        remote_params['start'] = start_display
    if end_display:
        remote_params['end'] = end_display
    if weights_query:
        remote_params['mode'] = 'weighted'
        remote_params['weights'] = weights_query
    remote_url = REMOTE_UI_BASE
    if remote_params:
        remote_url = f"{REMOTE_UI_BASE}?{urlencode(remote_params)}"

    base_params: dict[str, str] = {
        'preset': preset_param,
        'tickers': tickers_param,
        'start': start_display,
        'end': end_display,
        'window': window,
    }
    if skip_remote:
        base_params['skip_remote'] = 'on'
    if mode == 'weighted' and weights_param.strip():
        base_params['weights'] = weights_param

    def build_mode_url(target_mode: str) -> str:
        params = dict(base_params)
        params['mode'] = target_mode
        if target_mode != 'weighted':
            params.pop('weights', None)
        query = urlencode(params, doseq=True)
        return f"?{query}" if query else f"?mode={target_mode}"

    mode_equal_url = build_mode_url('equal')
    mode_weighted_url = build_mode_url('weighted')

    window_options = []
    for value in WINDOW_CHOICES:
        label = WINDOW_LABELS[value]
        selected_attr = ' selected' if value == window else ''
        window_options.append(f"<option value='{value}'{selected_attr}>{html.escape(label)}</option>")

    preset_options = ["<option value='custom'" + (" selected" if preset_param == 'custom' else "") + ">Custom</option>"]
    for value, label, tickers_list in presets:
        if not tickers_list:
            continue
        selected_attr = ' selected' if value == preset_param else ''
        preset_options.append(f"<option value='{value}'{selected_attr}>{html.escape(label)}</option>")

    controls = [
        "<section class='control-panel'>",
        "<h2>Backtest Controls</h2>",
        "<div class='mode-tabs'>",
        f"<a class='tab{' active' if mode == 'equal' else ''}' href='{html.escape(mode_equal_url)}'>Equal Weight</a>",
        f"<a class='tab{' active' if mode == 'weighted' else ''}' href='{html.escape(mode_weighted_url)}'>Custom Weights</a>",
        "</div>",
        "<div class='control-actions'>",
        f"<a id='remote-link' data-base='{html.escape(REMOTE_UI_BASE)}' class='external-link' href='{html.escape(remote_url)}' target='_blank' rel='noopener'>Open Remote Platform</a>",
        "</div>",
        "<form method='get'>",
        f"<input type='hidden' name='mode' value='{html.escape(mode)}'>",
        "<div>",
        "<label for='preset'>Preset Portfolio</label>",
        f"<select id='preset' name='preset'>{''.join(preset_options)}</select>",
        "<p class='helper'>Choose a preset or stay on Custom to edit manually.</p>",
        "</div>",
        "<div>",
        "<label for='tickers'>Tickers</label>",
        f"<textarea id='tickers' name='tickers' placeholder='e.g. NVDA,AAPL,MSFT'>{html.escape(tickers_param)}</textarea>",
        "<p class='helper'>Use comma separated symbols, '@file.txt', or 'all'.</p>",
        "</div>",
    ]

    if mode == 'weighted':
        controls.extend([
            "<div>",
            "<label for='weights'>Weights (Ticker: Weight)</label>",
            f"<textarea id='weights' name='weights' placeholder='e.g. NVDA:0.6,TSLA:0.4'>{html.escape(weights_param)}</textarea>",
            "<p class='helper'>Leave blank to assign equal weights to the listed tickers.</p>",
            "</div>",
        ])

    controls.extend([
        "<div>",
        "<label for='window'>Date Window</label>",
        f"<select id='window' name='window'>{''.join(window_options)}</select>",
        "<p class='helper'>Preset windows override the start date.</p>",
        "</div>",
        "<div>",
        "<label for='start'>Start Date</label>",
        f"<input type='date' id='start' name='start' value='{html.escape(start_display)}'>",
        "</div>",
        "<div>",
        "<label for='end'>End Date</label>",
        f"<input type='date' id='end' name='end' value='{html.escape(end_display)}'>",
        "</div>",
        "<div class='checkbox'>",
        f"<input type='checkbox' id='skip_remote' name='skip_remote' value='on' {checked_attr}>",
        "<label for='skip_remote'>Skip external comparison</label>",
        "</div>",
        "<div>",
        "<button type='submit'>Run Backtest</button>",
        "</div>",
        "</form>",
    ])

    if error_message:
        controls.append(f"<div class='alert error'>{html.escape(error_message)}</div>")
    if remote_message:
        controls.append(f"<div class='alert info'>{html.escape(remote_message)}</div>")

    controls.append("</section>")
    controls_html = '\n'.join(controls)
    script = r"""<script>
(function(){
  const link = document.getElementById('remote-link');
  if (!link) return;
  const base = link.dataset.base || link.href;
  const tickersInput = document.getElementById('tickers');
  const weightsInput = document.getElementById('weights');
  const startInput = document.getElementById('start');
  const endInput = document.getElementById('end');
  const windowSelect = document.getElementById('window');
  const presetSelect = document.getElementById('preset');
  const modeInput = document.querySelector("input[name='mode']");

  function formatDate(date){
    const tzOffset = date.getTimezoneOffset();
    const local = new Date(date.getTime() - tzOffset * 60000);
    return local.toISOString().slice(0, 10);
  }

  function applyWindow(){
    if (!windowSelect || !endInput || !startInput) return;
    const endVal = endInput.value ? new Date(endInput.value + 'T00:00:00') : new Date();
    if (!endInput.value) endInput.value = formatDate(endVal);
    let startDate = null;
    switch(windowSelect.value){
      case '1d':
        startDate = endVal;
        break;
      case '1w':
        startDate = new Date(endVal);
        startDate.setDate(startDate.getDate() - 6);
        break;
      case '1m':
        startDate = new Date(endVal);
        startDate.setMonth(startDate.getMonth() - 1);
        break;
      case '3m':
        startDate = new Date(endVal);
        startDate.setMonth(startDate.getMonth() - 3);
        break;
      case '6m':
        startDate = new Date(endVal);
        startDate.setMonth(startDate.getMonth() - 6);
        break;
      case '1y':
        startDate = new Date(endVal);
        startDate.setFullYear(startDate.getFullYear() - 1);
        break;
      case '3y':
        startDate = new Date(endVal);
        startDate.setFullYear(startDate.getFullYear() - 3);
        break;
      case '5y':
        startDate = new Date(endVal);
        startDate.setFullYear(startDate.getFullYear() - 5);
        break;
      case '10y':
        startDate = new Date(endVal);
        startDate.setFullYear(startDate.getFullYear() - 10);
        break;
      case 'max':
        startDate = new Date('1970-01-01T00:00:00Z');
        break;
      default:
        startDate = null;
    }
    if (startDate) startInput.value = formatDate(startDate);
  }

  function normalizeWeights(raw){
    if (!raw) return '';
    return raw
      .split(/[\n,]+/)
      .map(part => part.trim())
      .filter(Boolean)
      .map(entry => entry.replace(/\s+/g, ''))
      .join(',');
  }

  function updateLink(){
    const tickersRaw = tickersInput ? tickersInput.value : '';
    const tickers = tickersRaw ? tickersRaw.replace(/\s+/g, '') : '';
    const start = startInput ? startInput.value : '';
    const end = endInput ? endInput.value : '';
    const params = new URLSearchParams();
    if (tickers) params.set('tickers', tickers);
    if (start) params.set('start', start);
    if (end) params.set('end', end);
    const mode = modeInput ? modeInput.value : 'equal';
    if (mode === 'weighted') {
      params.set('mode', 'weighted');
      const weights = weightsInput ? normalizeWeights(weightsInput.value) : '';
      if (weights) params.set('weights', weights);
    }
    const query = params.toString();
    link.href = query ? `${base}?${query}` : base;
  }

  if (windowSelect) {
    windowSelect.addEventListener('change', () => {
      if (windowSelect.value !== 'custom') {
        applyWindow();
      }
      updateLink();
    });
  }

  [tickersInput, startInput, endInput, weightsInput].forEach(el => {
    if (el) el.addEventListener('input', updateLink);
  });

  if (presetSelect) {
    presetSelect.addEventListener('change', () => {
      setTimeout(() => {
        if (windowSelect && windowSelect.value !== 'custom') {
          applyWindow();
        }
        updateLink();
      }, 0);
    });
  }

  if (windowSelect && windowSelect.value !== 'custom') {
    applyWindow();
  }
  updateLink();
})();
</script>"""

    controls_html = controls_html + '\n' + script

    return _inject_controls(report_html, controls_html)

if __name__ == '__main__':
    app.run(debug=True, port=5000)






