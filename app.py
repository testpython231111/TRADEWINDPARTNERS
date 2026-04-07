"""
Aksjeanalyse PRO — Flask Web App
"""

import os, io, base64, warnings, json, gc, time, logging, hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from flask import Flask, request, jsonify, render_template
import requests as _requests

warnings.filterwarnings("ignore")

app = Flask(__name__)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

RISIKOFRI_RENTE = 0.045
BENCHMARK       = "^GSPC"

MAKRO_ASSETS = {
    "S&P 500":      {"yf": "^GSPC",    "analysis": "^GSPC",    "tv": "SP:SPX"},
    "Nasdaq":       {"yf": "^IXIC",    "analysis": "^IXIC",    "tv": "NASDAQ:IXIC"},
    "VIX":          {"yf": "^VIX",     "analysis": "^VIX",     "tv": "CBOE:VIX"},
    "US 10y Yield": {"yf": "^TNX",     "analysis": "^TNX",     "tv": "TVC:US10Y"},
    "Gold":         {"yf": "GC=F",     "analysis": "GC=F",     "tv": "OANDA:XAUUSD"},
    "Oil (Brent)":  {"yf": "BZ=F",     "analysis": "BZ=F",     "tv": "TVC:UKOIL"},
    "EUR/USD":      {"yf": "EURUSD=X", "analysis": "EURUSD=X", "tv": "OANDA:EURUSD"},
    "USD/NOK":      {"yf": "USDNOK=X", "analysis": "USDNOK=X", "tv": "FX_IDC:USDNOK"},
}
MAKRO_TICKERS = {name: meta["yf"] for name, meta in MAKRO_ASSETS.items()}

# ── Simple in-memory cache (TTL-based) ───────────────────────────────────────
_cache = {}
_cache_lock = Lock()
CACHE_TTL = 300  # 5 minutes
AI_GUARDRAILS = """
Use only the supplied data and context.
If a metric is missing, explicitly say the data is unavailable instead of guessing.
Do not invent dated catalysts, event schedules, analyst actions, insider motives, or news facts that are not in the input.
Commit to a directional view — mixed signals are normal in markets. Weigh the preponderance of evidence and take a stand. HOLD is a last resort, not a default.
""".strip()

def cache_get(key):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.time() - entry["ts"] < CACHE_TTL:
            return entry["data"]
        return None

def cache_set(key, data):
    with _cache_lock:
        _cache[key] = {"data": data, "ts": time.time()}
        # Evict old entries if cache grows large
        if len(_cache) > 200:
            cutoff = time.time() - CACHE_TTL
            for k in [k for k,v in _cache.items() if v["ts"] < cutoff]:
                del _cache[k]

# ── Hjelpefunksjoner ──────────────────────────────────────────────────────────

# Flask 3.x compatible NaN sanitizer — replaces NaN/Inf with None before jsonify
def sanitize(obj):
    if isinstance(obj, float):
        return None if (obj != obj or obj == float('inf') or obj == float('-inf')) else obj
    if isinstance(obj, dict):  return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [sanitize(v) for v in obj]
    return obj

def safe_jsonify(data):
    return jsonify(sanitize(data))

def get_request_data():
    return request.get_json(silent=True) or {}

def safe(v, pst=False, dec=2):
    try:
        if v is None or (isinstance(v, float) and np.isnan(float(v))): return "N/A"
        return f"{float(v)*100:.{dec}f}%" if pst else f"{float(v):.{dec}f}"
    except: return "N/A"

def safe_float(v):
    try:
        if v is None:
            return None
        out = float(v)
        if np.isnan(out) or np.isinf(out):
            return None
        return out
    except:
        return None

def safe_int(v):
    try:
        if v is None:
            return None
        out = int(v)
        return out
    except Exception:
        return None

def record_warning(warnings_list, label, exc):
    logger.warning(f"[{label}] {exc}")
    if warnings_list is not None and label not in warnings_list:
        warnings_list.append(label)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def stor_tall(n):
    try:
        n = float(n)
        for g, e in [(1e12,"T"),(1e9,"B"),(1e6,"M")]:
            if abs(n) >= g: return f"{n/g:.2f}{e}"
        return f"{n:.2f}"
    except: return "N/A"

def normalize_macro_series(symbol, series):
    vals = pd.Series(series).dropna().astype(float)
    if symbol == "^TNX":
        vals = vals / 10.0
    return vals

def format_macro_value(name, value):
    val = safe_float(value)
    if val is None:
        return "N/A"
    if "Yield" in name:
        return f"{val:.2f}%"
    if abs(val) >= 1000:
        return f"{val:,.2f}"
    if abs(val) >= 1:
        return f"{val:.2f}"
    return f"{val:.4f}"

def first_valid_number(*values):
    for value in values:
        out = safe_float(value)
        if out is not None:
            return out
    return None

def get_statement_frame(ticker_obj, attrs):
    for attr in attrs:
        try:
            stmt = getattr(ticker_obj, attr)
        except Exception:
            stmt = None
        if stmt is not None and not stmt.empty:
            return stmt
    return None

def extract_statement_series(statement, labels):
    try:
        if statement is None or statement.empty:
            return None
        for label in labels:
            if label in statement.index:
                series = pd.to_numeric(statement.loc[label], errors="coerce").dropna()
                if not series.empty:
                    try:
                        series = series.sort_index(ascending=False)
                    except Exception:
                        pass
                    return series
    except Exception:
        return None
    return None

def extract_statement_value(statement, labels):
    series = extract_statement_series(statement, labels)
    if series is None or series.empty:
        return None
    return safe_float(series.iloc[0])

def extract_ttm_statement_value(statement, labels):
    series = extract_statement_series(statement, labels)
    if series is None or series.empty:
        return None
    return safe_float(series.iloc[: min(len(series), 4)].sum())

def extract_yoy_growth(statement, labels):
    series = extract_statement_series(statement, labels)
    if series is None or len(series) < 8:
        return None
    current = safe_float(series.iloc[:4].sum())
    previous = safe_float(series.iloc[4:8].sum())
    if current is None or previous in (None, 0):
        return None
    return safe_float((current / previous) - 1)

def derive_fcf_from_statement(statement, ttm=False):
    extractor = extract_ttm_statement_value if ttm else extract_statement_value
    direct = extractor(statement, ["Free Cash Flow", "FreeCashFlow"])
    if direct is not None:
        return direct
    operating_cf = extractor(
        statement,
        ["Operating Cash Flow", "Total Cash From Operating Activities"],
    )
    capex = extractor(
        statement,
        ["Capital Expenditure", "Capital Expenditures"],
    )
    if operating_cf is not None and capex is not None:
        return safe_float(operating_cf - abs(capex))
    return None

def get_download_close_series(frame, symbol):
    try:
        if frame is None or frame.empty:
            return pd.Series(dtype=float)
        if isinstance(frame.columns, pd.MultiIndex):
            level0 = frame.columns.get_level_values(0)
            if "Close" in level0:
                close_frame = frame["Close"]
                if isinstance(close_frame, pd.Series):
                    return close_frame.dropna().astype(float)
                if symbol in close_frame.columns:
                    return close_frame[symbol].dropna().astype(float)
            if symbol in level0:
                symbol_frame = frame[symbol]
                if isinstance(symbol_frame, pd.Series):
                    return symbol_frame.dropna().astype(float)
                if "Close" in symbol_frame.columns:
                    return symbol_frame["Close"].dropna().astype(float)
        if "Close" in frame.columns:
            return frame["Close"].dropna().astype(float)
    except Exception:
        return pd.Series(dtype=float)
    return pd.Series(dtype=float)

def get_risk_free_rate():
    cached = cache_get("risk_free_rate")
    if cached is not None:
        return cached
    try:
        ticker = yf.Ticker("^TNX")
        try:
            fast_info = ticker.fast_info
        except Exception:
            fast_info = None
        raw_value = first_valid_number(
            getattr(fast_info, "last_price", None),
            getattr(fast_info, "previous_close", None),
        )
        if raw_value is None:
            hist = yf.download("^TNX", period="5d", interval="1d", progress=False, auto_adjust=False)
            series = normalize_macro_series("^TNX", get_download_close_series(hist, "^TNX"))
            raw_value = first_valid_number(series.iloc[-1] if not series.empty else None)
        else:
            raw_value = safe_float(raw_value / 10.0)
        if raw_value is not None:
            rate = clamp(raw_value / 100.0, 0.02, 0.08)
            cache_set("risk_free_rate", rate)
            return rate
    except Exception:
        pass
    return RISIKOFRI_RENTE

def build_company_snapshot(aksje, info=None, fast_info=None):
    info = info if info is not None else aksje.info
    if fast_info is None:
        try:
            fast_info = aksje.fast_info
        except Exception:
            fast_info = None

    quarterly_income = get_statement_frame(aksje, ("quarterly_income_stmt", "quarterly_financials"))
    annual_income = get_statement_frame(aksje, ("income_stmt", "financials"))
    quarterly_cash = get_statement_frame(aksje, ("quarterly_cash_flow", "quarterly_cashflow"))
    annual_cash = get_statement_frame(aksje, ("cash_flow", "cashflow"))
    quarterly_balance = get_statement_frame(aksje, ("quarterly_balance_sheet", "quarterly_balancesheet"))
    annual_balance = get_statement_frame(aksje, ("balance_sheet", "balancesheet"))

    revenue_labels = ["Total Revenue", "Operating Revenue", "Revenue"]
    gross_profit_labels = ["Gross Profit"]
    operating_income_labels = ["Operating Income", "OperatingIncome"]
    net_income_labels = ["Net Income", "NetIncome"]
    equity_labels = ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"]
    cash_labels = [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Short Term Investments",
    ]
    debt_labels = ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"]

    revenue = first_valid_number(
        info.get("totalRevenue"),
        extract_ttm_statement_value(quarterly_income, revenue_labels),
        extract_statement_value(annual_income, revenue_labels),
    )
    revenue_growth = first_valid_number(
        info.get("revenueGrowth"),
        extract_yoy_growth(quarterly_income, revenue_labels),
    )
    earnings_growth = first_valid_number(
        info.get("earningsGrowth"),
        info.get("earningsQuarterlyGrowth"),
        extract_yoy_growth(quarterly_income, net_income_labels),
    )

    ttm_fcf = derive_fcf_from_statement(quarterly_cash, ttm=True)
    reported_fcf = safe_float(info.get("freeCashflow"))
    annual_fcf = derive_fcf_from_statement(annual_cash, ttm=False)
    if ttm_fcf is not None:
        free_cashflow = ttm_fcf
        fcf_source = "TTM free cash flow from quarterly cash flow statement"
    elif reported_fcf is not None:
        free_cashflow = reported_fcf
        fcf_source = "Reported free cash flow"
    elif annual_fcf is not None:
        free_cashflow = annual_fcf
        fcf_source = "Latest annual free cash flow from cash flow statement"
    else:
        free_cashflow = None
        fcf_source = "Unavailable"

    current_price = first_valid_number(
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
        getattr(fast_info, "last_price", None),
    )
    shares_out = first_valid_number(
        info.get("sharesOutstanding"),
        getattr(fast_info, "shares", None),
    )
    market_cap = first_valid_number(
        info.get("marketCap"),
        current_price * shares_out if current_price and shares_out else None,
    )
    year_high = first_valid_number(
        info.get("fiftyTwoWeekHigh"),
        getattr(fast_info, "year_high", None),
    )
    year_low = first_valid_number(
        info.get("fiftyTwoWeekLow"),
        getattr(fast_info, "year_low", None),
    )
    target_mean = first_valid_number(info.get("targetMeanPrice"))
    target_high = first_valid_number(info.get("targetHighPrice"))
    target_low = first_valid_number(info.get("targetLowPrice"))

    total_debt = max(
        first_valid_number(
            info.get("totalDebt"),
            extract_statement_value(quarterly_balance, debt_labels),
            extract_statement_value(annual_balance, debt_labels),
        ) or 0,
        0,
    )
    cash = max(
        first_valid_number(
            info.get("totalCash"),
            extract_statement_value(quarterly_balance, cash_labels),
            extract_statement_value(annual_balance, cash_labels),
        ) or 0,
        0,
    )
    stock_equity = first_valid_number(
        info.get("totalStockholderEquity"),
        extract_statement_value(quarterly_balance, equity_labels),
        extract_statement_value(annual_balance, equity_labels),
    )
    debt_to_equity = first_valid_number(
        info.get("debtToEquity"),
        (total_debt / stock_equity) * 100 if total_debt and stock_equity else None,
    )

    gross_profit = first_valid_number(
        extract_ttm_statement_value(quarterly_income, gross_profit_labels),
        extract_statement_value(annual_income, gross_profit_labels),
    )
    operating_income = first_valid_number(
        extract_ttm_statement_value(quarterly_income, operating_income_labels),
        extract_statement_value(annual_income, operating_income_labels),
    )
    net_income = first_valid_number(
        extract_ttm_statement_value(quarterly_income, net_income_labels),
        extract_statement_value(annual_income, net_income_labels),
    )
    gross_margin = first_valid_number(
        info.get("grossMargins"),
        (gross_profit / revenue) if gross_profit is not None and revenue else None,
    )
    operating_margin = first_valid_number(
        info.get("operatingMargins"),
        (operating_income / revenue) if operating_income is not None and revenue else None,
    )
    net_margin = first_valid_number(
        info.get("profitMargins"),
        (net_income / revenue) if net_income is not None and revenue else None,
    )
    ebitda = first_valid_number(
        info.get("ebitda"),
        extract_ttm_statement_value(quarterly_income, ["EBITDA"]),
        extract_statement_value(annual_income, ["EBITDA"]),
    )

    return {
        "current_price": current_price,
        "shares_out": shares_out,
        "market_cap": market_cap,
        "free_cashflow": free_cashflow,
        "fcf_source": fcf_source,
        "revenue": revenue,
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "year_high": year_high,
        "year_low": year_low,
        "target_mean": target_mean,
        "target_high": target_high,
        "target_low": target_low,
        "total_debt": total_debt,
        "cash": cash,
        "debt_to_equity": debt_to_equity,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "ebitda": ebitda,
    }

def spør_ai(prompt: str, api_key: str, maks=1200) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key: return "No API key configured."
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": maks,
                "temperature": 0.5,
                "thinkingConfig": {"thinkingBudget": 0}
            }
        }
        r = _requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        logger.error(f"[AI] API error: {e}")
        return f"AI error: {e}"

# ── Teknisk analyse ───────────────────────────────────────────────────────────

def beregn_tekniske(df):
    c = df["Close"]
    df["SMA20"]  = c.rolling(20).mean()
    df["SMA50"]  = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["EMA12"]  = c.ewm(span=12, adjust=False).mean()
    df["EMA26"]  = c.ewm(span=26, adjust=False).mean()
    df["MACD"]   = df["EMA12"] - df["EMA26"]
    df["MACD_S"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_H"] = df["MACD"] - df["MACD_S"]
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    df["BB_Mid"] = c.rolling(20).mean()
    std          = c.rolling(20).std()
    df["BB_Up"]  = df["BB_Mid"] + 2*std
    df["BB_Lo"]  = df["BB_Mid"] - 2*std
    df["BB_pct"] = (c - df["BB_Lo"]) / (df["BB_Up"] - df["BB_Lo"])
    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-c.shift()).abs(),
                    (df["Low"]-c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"]       = tr.rolling(14).mean()
    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
    return df

def hent_signaler(df):
    s = df.iloc[-1]
    p = float(s["Close"])
    return {
        "pris":       round(p, 2),
        "rsi":        round(float(s["RSI"]),    1) if not np.isnan(s["RSI"])    else 50,
        "macd_bull":  bool(s["MACD"] > s["MACD_S"]),
        "over_sma20": bool(p > s["SMA20"])  if not np.isnan(s["SMA20"])  else False,
        "over_sma50": bool(p > s["SMA50"])  if not np.isnan(s["SMA50"])  else False,
        "over_sma200":bool(p > s["SMA200"]) if not np.isnan(s["SMA200"]) else False,
        "bb_pct":     round(float(s["BB_pct"]),3) if not np.isnan(s["BB_pct"]) else 0.5,
        "sma20":      round(float(s["SMA20"]),  2) if not np.isnan(s["SMA20"])  else p,
        "sma50":      round(float(s["SMA50"]),  2) if not np.isnan(s["SMA50"])  else p,
        "sma200":     round(float(s["SMA200"]), 2) if not np.isnan(s["SMA200"]) else p,
        "bb_upper":   round(float(s["BB_Up"]),  2) if not np.isnan(s["BB_Up"])  else p,
        "bb_lower":   round(float(s["BB_Lo"]),  2) if not np.isnan(s["BB_Lo"])  else p,
        "atr":        round(float(s["ATR"]),    2) if not np.isnan(s["ATR"])    else 0,
    }

# ── Risiko ────────────────────────────────────────────────────────────────────

def beregn_risiko(df, bm_df):
    ret  = df["Close"].pct_change().dropna()
    bret = bm_df["Close"].pct_change().dropna()
    felles = ret.index.intersection(bret.index)
    ret, bret = ret.loc[felles], bret.loc[felles]
    if len(ret) < 10: return {}
    hd = 252
    rf_annual = get_risk_free_rate()
    total  = (df["Close"].iloc[-1]/df["Close"].iloc[0]-1)*100
    dager  = (df.index[-1]-df.index[0]).days
    cagr   = ((df["Close"].iloc[-1]/df["Close"].iloc[0])**(365/max(dager,1))-1)*100
    dagvol = ret.std()
    årvol  = dagvol*np.sqrt(hd)*100
    rf     = rf_annual/hd
    sharpe = (ret.mean()-rf)/dagvol*np.sqrt(hd) if dagvol and not np.isnan(dagvol) else float("nan")
    neg    = ret[ret < rf]
    dside  = neg.std()*np.sqrt(hd) if len(neg)>1 else float("nan")
    sortino = (ret.mean()*hd-rf_annual)/dside if dside and dside!=0 else float("nan")
    rm     = df["Close"].cummax()
    dd     = (df["Close"]-rm)/rm
    max_dd = dd.min()*100
    kov    = np.cov(ret, bret)
    beta   = kov[0,1]/kov[1,1] if kov[1,1]!=0 else float("nan")
    alpha  = (ret.mean()-(rf+beta*(bret.mean()-rf)))*hd*100
    var95  = np.percentile(ret,5)*100
    var99  = np.percentile(ret,1)*100
    calmar = cagr/abs(max_dd) if max_dd!=0 else float("nan")
    return dict(total=round(total,2), cagr=round(cagr,2), vol=round(årvol,2),
                sharpe=round(sharpe,3) if not np.isnan(sharpe) else None, sortino=round(sortino,3) if not np.isnan(sortino) else None,
                calmar=round(calmar,3) if not np.isnan(calmar) else None,
                max_dd=round(max_dd,2), beta=round(beta,3) if not np.isnan(beta) else None,
                alpha=round(alpha,2), var95=round(var95,2), var99=round(var99,2), risk_free=round(rf_annual*100, 2),
                dd_series=dd.tolist(), dd_dates=[str(d)[:10] for d in dd.index])

# ── Graf → base64 ─────────────────────────────────────────────────────────────

def lag_graf(df: pd.DataFrame, ticker: str) -> str:
    style = {
        "axes.facecolor":  "#06090F",
        "figure.facecolor":"#06090F",
        "axes.edgecolor":  "#1A2D45",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color":     "#7A92B0",
        "ytick.color":     "#7A92B0",
        "text.color":      "#7A92B0",
        "grid.color":      "#1A2D45",
        "grid.linestyle":  "--",
        "grid.alpha":      0.3,
    }
    with matplotlib.rc_context(style):
        fig = plt.figure(figsize=(16, 12), facecolor="#06090F")
        gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.3)
        C   = {"kurs":"#4A9ED6","sma20":"#F0B429","sma50":"#64B5F6",
               "sma200":"#E05470","bull":"#4A9ED6","bear":"#E05470","text":"#7A92B0"}

        ax1 = fig.add_subplot(gs[0,:])
        ax1.set_facecolor("#06090F")
        ax1.plot(df.index, df["Close"],  color=C["kurs"],  lw=1.5, label="Kurs")
        ax1.plot(df.index, df["SMA20"],  color=C["sma20"], lw=0.8, ls="--", label="SMA20")
        ax1.plot(df.index, df["SMA50"],  color=C["sma50"], lw=0.8, ls="--", label="SMA50")
        ax1.plot(df.index, df["SMA200"], color=C["sma200"],lw=0.8, ls="--", label="SMA200")
        ax1.fill_between(df.index, df["BB_Up"], df["BB_Lo"],
                         alpha=0.05, color=C["kurs"])
        ax1.set_title(f"{ticker} — Kurs & Indikatorer", color=C["text"], fontsize=10)
        ax1.legend(fontsize=7, labelcolor=C["text"])
        ax1.tick_params(colors=C["text"]); ax1.spines[:].set_color("#1A2D45")

        ax2 = fig.add_subplot(gs[1,:])
        ax2.set_facecolor("#06090F")
        fc = [C["bull"] if df["Close"].iloc[i]>=df["Close"].iloc[i-1] else C["bear"]
              for i in range(1,len(df))]
        ax2.bar(df.index[1:], df["Volume"].iloc[1:], color=fc, alpha=0.6, width=1)
        ax2.plot(df.index, df["Vol_SMA20"], color=C["sma20"], lw=0.8, ls="--")
        ax2.set_title("Volume", color=C["text"], fontsize=10)
        ax2.tick_params(colors=C["text"]); ax2.spines[:].set_color("#1A2D45")

        ax3 = fig.add_subplot(gs[2,0])
        ax3.set_facecolor("#06090F")
        ax3.plot(df.index, df["RSI"], color="#f4a261", lw=1.2)
        ax3.axhline(70, color=C["bear"], ls="--", lw=0.7, alpha=0.6)
        ax3.axhline(30, color=C["bull"], ls="--", lw=0.7, alpha=0.6)
        ax3.fill_between(df.index, df["RSI"], 70, where=df["RSI"]>=70, alpha=0.1, color=C["bear"])
        ax3.fill_between(df.index, df["RSI"], 30, where=df["RSI"]<=30, alpha=0.1, color=C["bull"])
        ax3.set_ylim(0,100); ax3.set_title("RSI (14d)", color=C["text"], fontsize=10)
        ax3.tick_params(colors=C["text"]); ax3.spines[:].set_color("#1A2D45")

        ax4 = fig.add_subplot(gs[2,1])
        ax4.set_facecolor("#06090F")
        ax4.plot(df.index, df["MACD"],   color=C["sma50"], lw=1.2, label="MACD")
        ax4.plot(df.index, df["MACD_S"], color=C["sma20"], lw=1.0, ls="--", label="Signal")
        hc = [C["bull"] if v>=0 else C["bear"] for v in df["MACD_H"]]
        ax4.bar(df.index, df["MACD_H"], color=hc, alpha=0.5, width=1)
        ax4.set_title("MACD", color=C["text"], fontsize=10)
        ax4.legend(fontsize=7, labelcolor=C["text"])
        ax4.tick_params(colors=C["text"]); ax4.spines[:].set_color("#1A2D45")

        ax5 = fig.add_subplot(gs[3,0])
        ax5.set_facecolor("#06090F")
        rm = df["Close"].cummax()
        dd = (df["Close"]-rm)/rm*100
        ax5.fill_between(df.index, dd, 0, color=C["bear"], alpha=0.5)
        ax5.plot(df.index, dd, color=C["bear"], lw=0.8)
        ax5.set_title("Drawdown (%)", color=C["text"], fontsize=10)
        ax5.tick_params(colors=C["text"]); ax5.spines[:].set_color("#1A2D45")

        ax6 = fig.add_subplot(gs[3,1])
        ax6.set_facecolor("#06090F")
        dagret = df["Close"].pct_change().dropna()*100
        ax6.hist(dagret, bins=50, color=C["sma50"], alpha=0.6, edgecolor="none")
        ax6.axvline(dagret.mean(), color=C["sma20"], lw=1.2, ls="--")
        ax6.axvline(np.percentile(dagret,5), color=C["bear"], lw=1.0, ls=":")
        ax6.set_title("Return Distribution", color=C["text"], fontsize=10)
        ax6.tick_params(colors=C["text"]); ax6.spines[:].set_color("#1A2D45")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#06090F")
        plt.close(fig)
        gc.collect()
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

# ── Flask ruter ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    data   = get_request_data()
    ticker = data.get("ticker","").strip().upper()
    periode= data.get("periode","1y")

    if not ticker:
        return safe_jsonify({"error": "No ticker provided"}), 400

    # Check cache
    cache_key = f"analyse:{ticker}:{periode}"
    cached = cache_get(cache_key)
    if cached:
        return safe_jsonify(cached)

    try:
        # Fetch stock data + benchmark in parallel
        def fetch_stock():
            aksje = yf.Ticker(ticker)
            try:
                fast_info = aksje.fast_info
            except Exception:
                fast_info = None
            df = yf.download(ticker, period=periode, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return aksje, fast_info, df

        def fetch_benchmark():
            try:
                bm = yf.download(BENCHMARK, period=periode, progress=False, auto_adjust=True)
                if isinstance(bm.columns, pd.MultiIndex): bm.columns = bm.columns.get_level_values(0)
                return bm
            except: return None

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_stock = ex.submit(fetch_stock)
            f_bm    = ex.submit(fetch_benchmark)
            aksje, fast_info, df = f_stock.result()
            bm_result = f_bm.result()

        if df.empty:
            return safe_jsonify({"error": f"No data for {ticker}. Check the ticker symbol."}), 404

        info  = aksje.info
        bm_df = bm_result if bm_result is not None and not bm_result.empty else df.copy()

    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500

    df  = beregn_tekniske(df)
    sig = hent_signaler(df)
    ri  = beregn_risiko(df, bm_df)
    latest_close = safe_float(df["Close"].dropna().iloc[-1]) if not df["Close"].dropna().empty else None
    snapshot = build_company_snapshot(aksje, info, fast_info)
    current_price   = first_valid_number(snapshot["current_price"], latest_close)
    shares_out      = snapshot["shares_out"]
    market_cap      = first_valid_number(snapshot["market_cap"], current_price * shares_out if current_price and shares_out else None)
    free_cashflow   = snapshot["free_cashflow"]
    revenue_growth  = snapshot["revenue_growth"]
    earnings_growth = snapshot["earnings_growth"]
    year_high       = snapshot["year_high"]
    year_low        = snapshot["year_low"]
    target_mean     = snapshot["target_mean"]
    target_high     = snapshot["target_high"]
    target_low      = snapshot["target_low"]
    fcf_yield       = (free_cashflow / market_cap * 100) if free_cashflow and market_cap else None

    # Kurshistorikk for chart
    historikk = {
        "datoer": [str(d)[:10] for d in df.index],
        "priser": [round(float(p),2) for p in df["Close"]],
    }

    fundamental = {
        "navn":       info.get("longName", ticker),
        "sektor":     info.get("sector","N/A"),
        "bransje":    info.get("industry","N/A"),
        "børs":       info.get("exchange","N/A"),
        "valuta":     info.get("currency","N/A"),
        "pris":       safe(current_price),
        "pe":         safe(info.get("trailingPE")),
        "forward_pe": safe(info.get("forwardPE")),
        "pb":         safe(info.get("priceToBook")),
        "ps":         safe(info.get("priceToSalesTrailing12Months")),
        "eps":        safe(info.get("trailingEps")),
        "mktcap":     stor_tall(market_cap),
        "omsetning":  stor_tall(snapshot["revenue"]),
        "ebitda":     stor_tall(snapshot["ebitda"]),
        "frikontant": stor_tall(free_cashflow),
        "fcf_yield":  safe(fcf_yield, pst=True),
        "gjeld_ek":   safe(snapshot["debt_to_equity"]),
        "utbytte":    safe(info.get("dividendYield"), pst=True),
        "roe":        safe(info.get("returnOnEquity"), pst=True),
        "roa":        safe(info.get("returnOnAssets"), pst=True),
        "bruttomargin":  safe(snapshot["gross_margin"], pst=True),
        "driftsmargin":  safe(snapshot["operating_margin"], pst=True),
        "nettmargin":    safe(snapshot["net_margin"], pst=True),
        "revenue_growth":  safe(revenue_growth, pst=True),
        "earnings_growth": safe(earnings_growth, pst=True),
        "52u_høy":    safe(year_high),
        "52u_lav":    safe(year_low),
        "mål":        safe(target_mean),
        "mål_høy":    safe(target_high),
        "mål_lav":    safe(target_low),
        "analyst_count": info.get("numberOfAnalystOpinions") or 0,
        "konsensus":  info.get("recommendationKey","N/A").upper(),
        # Raw numeric fields for Margin of Safety calculations (None if unavailable)
        "_pris_raw":          current_price,
        "_pe_raw":            info.get("trailingPE"),
        "_eps_raw":           info.get("trailingEps"),
        "_pb_raw":            info.get("priceToBook"),
        "_fcf_raw":           free_cashflow,
        "_revenue_raw":       snapshot["revenue"],
        "_fcf_source":        snapshot["fcf_source"],
        "_earnings_growth":   earnings_growth,
        "_revenue_growth":    revenue_growth,
        "_52w_high_raw":      year_high,
        "_52w_low_raw":       year_low,
        "_target_mean_raw":   target_mean,
        "_evEbitda_raw":      safe_float(info.get("enterpriseToEbitda")),
        "_peg_raw":           info.get("pegRatio"),
    }

    result = {
        "ticker":       ticker,
        "fundamental":  fundamental,
        "signaler":     sig,
        "risiko":       ri,
        "graf":         "",  # loaded separately via /api/chart
        "historikk":    historikk,
        "ai":           "",
    }
    cache_set(cache_key, result)
    return safe_jsonify(result)


@app.route("/api/chart", methods=["POST"])
def api_chart():
    data    = get_request_data()
    ticker  = data.get("ticker","").strip().upper()
    periode = data.get("periode","1y")
    if not ticker:
        return safe_jsonify({"error": "No ticker"}), 400
    cache_key = f"chart:{ticker}:{periode}"
    cached = cache_get(cache_key)
    if cached:
        return safe_jsonify({"graf": cached})
    try:
        df = yf.download(ticker, period=periode, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty:
            return safe_jsonify({"error": "No data"}), 404
        df = beregn_tekniske(df)
        graf = lag_graf(df, ticker)
        cache_set(cache_key, graf)
        return safe_jsonify({"graf": graf})
    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500

@app.route("/api/dcf", methods=["POST"])
def api_dcf():
    """
    Professional DCF valuation:
      • WACC built from CAPM (cost of equity) + after-tax cost of debt
      • 2-stage DCF: 5-year explicit FCF projection + Gordon Growth terminal value
      • Bull / Base / Bear scenarios
      • Sensitivity matrix: intrinsic value across growth × WACC grid
    """
    data   = get_request_data()
    ticker = data.get("ticker","").strip().upper()
    if not ticker:
        return safe_jsonify({"error": "No ticker"}), 400

    cache_key = f"dcf:{ticker}"
    cached = cache_get(cache_key)
    if cached:
        return safe_jsonify(cached)

    try:
        aksje = yf.Ticker(ticker)
        warnings_list = []
        info  = aksje.info
        try:
            fast_info = aksje.fast_info
        except Exception:
            fast_info = None

        snapshot = build_company_snapshot(aksje, info, fast_info)
        cur_price = snapshot["current_price"]
        shares_out = snapshot["shares_out"]
        market_cap = snapshot["market_cap"]
        total_debt = snapshot["total_debt"]
        cash = snapshot["cash"]
        fcf = snapshot["free_cashflow"]
        revenue = snapshot["revenue"]
        ebitda_val = snapshot["ebitda"]
        op_margin = snapshot["operating_margin"]
        beta_raw = first_valid_number(info.get("beta"), 1.0) or 1.0
        beta = clamp(beta_raw, 0.6, 2.2)
        tax_rate_raw = first_valid_number(info.get("effectiveTaxRate"), 0.21) or 0.21
        tax_rate = clamp(tax_rate_raw, 0.05, 0.30)
        interest_exp = abs(first_valid_number(info.get("interestExpense"), 0) or 0)
        sector = info.get("sector", "")
        currency = info.get("currency", "USD")
        revenue_growth = snapshot["revenue_growth"]
        earnings_growth = snapshot["earnings_growth"]
        fcf_source = snapshot["fcf_source"]

        if fcf is None or fcf <= 0 or shares_out is None or shares_out <= 0 or cur_price is None:
            return safe_jsonify({
                "error": "Insufficient financial data for a reliable DCF. A positive free cash flow, share count, and current price are required."
            }), 422

        RF          = get_risk_free_rate()
        ERP         = 0.055
        cost_equity = RF + beta * ERP

        enterprise_value = first_valid_number(
            info.get("enterpriseValue"),
            market_cap + total_debt - cash if market_cap is not None else None,
        )
        capital_base  = (market_cap or 0) + total_debt
        debt_weight   = (total_debt / capital_base) if capital_base > 0 else 0.0
        debt_weight   = clamp(debt_weight, 0.0, 0.60)
        equity_weight = 1.0 - debt_weight
        cost_debt_pre = (interest_exp / total_debt) if total_debt and interest_exp else 0.045
        cost_debt_pre = clamp(cost_debt_pre, 0.02, 0.12)
        cost_debt     = cost_debt_pre * (1 - tax_rate)
        wacc          = clamp(equity_weight * cost_equity + debt_weight * cost_debt, 0.055, 0.16)

        SECTOR_GROWTH = {
            "Technology": 0.10, "Healthcare": 0.08, "Communication Services": 0.09,
            "Consumer Discretionary": 0.07, "Financials": 0.06, "Industrials": 0.06,
            "Consumer Staples": 0.04, "Energy": 0.04, "Materials": 0.04,
            "Real Estate": 0.035, "Utilities": 0.03,
        }
        sector_g = SECTOR_GROWTH.get(sector, 0.055)
        growth_inputs = [sector_g]
        if revenue_growth is not None:
            growth_inputs.append(clamp(revenue_growth, -0.10, 0.20))
        if earnings_growth is not None:
            growth_inputs.append(clamp(earnings_growth, -0.15, 0.25))

        base_g = sum(growth_inputs) / len(growth_inputs)
        if op_margin is not None:
            if op_margin > 0.20:
                base_g += 0.005
            elif op_margin < 0.05:
                base_g -= 0.005
        if market_cap and total_debt and market_cap > 0 and total_debt / market_cap > 1.0:
            base_g -= 0.01
        base_g = clamp(base_g, -0.02, 0.16)

        if sector in ("Utilities", "Real Estate", "Consumer Staples"):
            terminal_base = 0.020
        elif sector in ("Technology", "Communication Services", "Healthcare"):
            terminal_base = 0.027
        else:
            terminal_base = 0.025

        scenarios = {
            "Bear": {
                "g5": clamp(base_g - 0.04, -0.08, 0.12),
                "terminal": clamp(terminal_base - 0.010, 0.010, 0.025),
                "wacc_adj": 0.0125,
                "label": "BEAR",
            },
            "Base": {
                "g5": base_g,
                "terminal": terminal_base,
                "wacc_adj": 0.0,
                "label": "BASE",
            },
            "Bull": {
                "g5": clamp(base_g + 0.04, 0.0, 0.20),
                "terminal": clamp(terminal_base + 0.005, 0.020, 0.030),
                "wacc_adj": -0.0075,
                "label": "BULL",
            },
        }

        def run_dcf(fcf_base, g5, terminal_g, w):
            w = clamp(w, 0.05, 0.18)
            g5 = clamp(g5, -0.20, 0.25)
            terminal_g = clamp(min(terminal_g, w - 0.01), 0.005, 0.03)

            # Stage 1: explicit 5-year FCFs
            pv_fcfs = 0.0
            cf = float(fcf_base)
            for yr in range(1, 6):
                cf *= (1 + g5)
                pv_fcfs += cf / (1 + w) ** yr

            # Stage 2: terminal value (Gordon Growth) at end of year 5
            terminal_fcf = cf * (1 + terminal_g)
            terminal_val = terminal_fcf / (w - terminal_g)
            pv_terminal  = terminal_val / (1 + w) ** 5

            enterprise_dcf = pv_fcfs + pv_terminal
            equity_val     = max(enterprise_dcf + float(cash) - float(total_debt), 0)
            return round(equity_val / float(shares_out), 2), round(pv_fcfs, 0), round(pv_terminal, 0)

        scenario_results = {}
        for name, sc in scenarios.items():
            w_adj    = wacc + sc["wacc_adj"]
            iv, pv1, pv2 = run_dcf(fcf, sc["g5"], sc["terminal"], w_adj)
            upside   = round((iv - float(cur_price)) / float(cur_price) * 100, 1) if cur_price else None
            scenario_results[name] = {
                "label":       sc["label"],
                "g5":          round(sc["g5"] * 100, 1),
                "terminal_g":  round(sc["terminal"] * 100, 2),
                "wacc":        round(w_adj * 100, 2),
                "intrinsic":   iv,
                "upside":      upside,
                "pv_fcfs":     pv1,
                "pv_terminal": pv2,
            }

        g_range    = [clamp(base_g + d, -0.10, 0.22) for d in (-0.06, -0.03, -0.01, 0, 0.02, 0.04, 0.06)]
        wacc_range = [clamp(wacc + d, 0.055, 0.18) for d in (-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02)]
        base_terminal = scenarios["Base"]["terminal"]

        matrix = []
        for g in g_range:
            row = []
            for w in wacc_range:
                iv, _, _ = run_dcf(fcf, g, base_terminal, w)
                row.append(iv)
            matrix.append(row)

        # ── Explicit 5-year FCF table (Base scenario) ───────────────────────
        base_w = wacc + scenarios["Base"]["wacc_adj"]
        base_g5 = scenarios["Base"]["g5"]
        fcf_table = []
        cf = float(fcf)
        for yr in range(1, 6):
            cf *= (1 + base_g5)
            pv = cf / (1 + base_w) ** yr
            fcf_table.append({
                "year":    yr,
                "fcf":     round(cf / 1e9, 3),
                "pv":      round(pv / 1e9, 3),
            })

        mos_signals = []

        dcf_base_iv   = scenario_results["Base"]["intrinsic"]
        dcf_base_up   = scenario_results["Base"]["upside"]
        if dcf_base_iv and dcf_base_up is not None:
            if dcf_base_up >= 30:
                dcf_verdict = ("STRONG BUY", "green", f"DCF intrinsic {dcf_base_up:+.1f}% above current price - significant margin of safety")
            elif dcf_base_up >= 10:
                dcf_verdict = ("UNDERVALUED", "green", f"DCF intrinsic {dcf_base_up:+.1f}% above current price - moderate margin of safety")
            elif dcf_base_up >= -10:
                dcf_verdict = ("FAIRLY VALUED", "yellow", f"DCF intrinsic within +/-10% of current price - limited margin of safety")
            elif dcf_base_up >= -25:
                dcf_verdict = ("OVERVALUED", "red", f"DCF intrinsic {dcf_base_up:.1f}% below current price - no margin of safety")
            else:
                dcf_verdict = ("SIGNIFICANTLY OVERVALUED", "red", f"DCF intrinsic {dcf_base_up:.1f}% below current price - avoid")
            mos_signals.append({
                "method":  "DCF (Base Case)",
                "iv":      round(dcf_base_iv, 2),
                "upside":  dcf_base_up,
                "verdict": dcf_verdict[0],
                "color":   dcf_verdict[1],
                "note":    dcf_verdict[2],
            })

        eps     = first_valid_number(info.get("trailingEps"))
        bvps    = first_valid_number(info.get("bookValue"))
        graham  = None
        if eps and bvps and eps > 0 and bvps > 0:
            graham = round((22.5 * float(eps) * float(bvps)) ** 0.5, 2)
            g_up   = round((graham - float(cur_price)) / float(cur_price) * 100, 1)
            if g_up >= 20:
                gv = ("UNDERVALUED", "green", f"Graham Number {graham:.2f} - {g_up:+.1f}% above price")
            elif g_up >= -5:
                gv = ("FAIRLY VALUED", "yellow", f"Graham Number {graham:.2f} - {g_up:+.1f}% vs price")
            else:
                gv = ("OVERVALUED", "red", f"Graham Number {graham:.2f} - {g_up:.1f}% below price")
            mos_signals.append({
                "method":  "Graham Number",
                "iv":      graham,
                "upside":  g_up,
                "verdict": gv[0],
                "color":   gv[1],
                "note":    gv[2],
            })

        pe_ratio = first_valid_number(info.get("trailingPE"))
        growth_for_peg = first_valid_number(earnings_growth, revenue_growth)
        peg = first_valid_number(info.get("pegRatio"))
        if peg is None and pe_ratio and growth_for_peg and growth_for_peg > 0:
            peg = round(float(pe_ratio) / (float(growth_for_peg) * 100), 2)
        elif peg is not None:
            peg = round(float(peg), 2)
        if peg is not None and peg > 0:
            if peg < 0.5:
                pgv = ("VERY CHEAP", "green", f"PEG {peg} - deeply undervalued relative to growth")
            elif peg < 1.0:
                pgv = ("UNDERVALUED", "green", f"PEG {peg} - trading below growth rate")
            elif peg < 1.5:
                pgv = ("FAIRLY VALUED", "yellow", f"PEG {peg} - fairly priced for its growth rate")
            elif peg < 2.5:
                pgv = ("OVERVALUED", "red", f"PEG {peg} - paying a premium over growth rate")
            else:
                pgv = ("EXPENSIVE", "red", f"PEG {peg} - significantly expensive vs. growth")
            mos_signals.append({
                "method":  "PEG Ratio",
                "iv":      None,
                "upside":  None,
                "peg":     peg,
                "verdict": pgv[0],
                "color":   pgv[1],
                "note":    pgv[2],
            })

        SECTOR_EV_EBITDA = {
            "Technology": 25, "Healthcare": 18, "Communication Services": 15,
            "Consumer Discretionary": 14, "Financials": 12, "Industrials": 13,
            "Consumer Staples": 14, "Energy": 8, "Materials": 10,
            "Real Estate": 20, "Utilities": 12,
        }
        ev_ebitda = first_valid_number(info.get("enterpriseToEbitda"))
        if ev_ebitda is None and enterprise_value and ebitda_val and float(ebitda_val) > 0:
            ev_ebitda = float(enterprise_value) / float(ebitda_val)
        if ev_ebitda is not None and ev_ebitda > 0:
            ev_ebitda     = round(float(ev_ebitda), 1)
            sector_median = SECTOR_EV_EBITDA.get(sector, 14)
            premium       = round((ev_ebitda - sector_median) / sector_median * 100, 1)
            if premium <= -20:
                evv = ("UNDERVALUED", "green", f"EV/EBITDA {ev_ebitda}x vs. sector median {sector_median}x - {abs(premium):.0f}% discount")
            elif premium <= 5:
                evv = ("IN LINE", "yellow", f"EV/EBITDA {ev_ebitda}x vs. sector median {sector_median}x - roughly in line")
            elif premium <= 30:
                evv = ("PREMIUM", "yellow", f"EV/EBITDA {ev_ebitda}x vs. sector median {sector_median}x - {premium:.0f}% premium")
            else:
                evv = ("EXPENSIVE", "red", f"EV/EBITDA {ev_ebitda}x vs. sector median {sector_median}x - {premium:.0f}% premium")
            mos_signals.append({
                "method":   "EV/EBITDA",
                "iv":       None,
                "upside":   None,
                "multiple": ev_ebitda,
                "sector_median": sector_median,
                "premium":  premium,
                "verdict":  evv[0],
                "color":    evv[1],
                "note":     evv[2],
            })

        target_mean   = first_valid_number(info.get("targetMeanPrice"))
        analyst_up    = None
        if target_mean and cur_price:
            analyst_up = round((float(target_mean) - float(cur_price)) / float(cur_price) * 100, 1)
            n_analysts  = info.get("numberOfAnalystOpinions", 0) or 0
            if analyst_up >= 20:
                av = ("STRONG BUY", "green", f"Wall Street target {analyst_up:+.1f}% above price ({n_analysts} analysts)")
            elif analyst_up >= 5:
                av = ("BUY", "green", f"Wall Street target {analyst_up:+.1f}% above price ({n_analysts} analysts)")
            elif analyst_up >= -5:
                av = ("HOLD", "yellow", f"Wall Street target in line with price ({n_analysts} analysts)")
            else:
                av = ("SELL", "red", f"Wall Street target {analyst_up:.1f}% below price ({n_analysts} analysts)")
            mos_signals.append({
                "method":  "Analyst Consensus",
                "iv":      round(float(target_mean), 2),
                "upside":  analyst_up,
                "verdict": av[0],
                "color":   av[1],
                "note":    av[2],
            })

        COLOR_SCORE = {"green": 2, "yellow": 1, "red": 0}
        if mos_signals:
            avg = sum(COLOR_SCORE.get(s["color"], 1) for s in mos_signals) / len(mos_signals)
            if avg >= 1.7:
                mos_overall = {"verdict": "STRONG VALUE", "color": "green",
                               "desc": "Multiple valuation methods agree the stock is undervalued with a meaningful margin of safety."}
            elif avg >= 1.2:
                mos_overall = {"verdict": "MODERATE VALUE", "color": "green",
                               "desc": "More signals point to undervaluation than overvaluation - some margin of safety is present."}
            elif avg >= 0.8:
                mos_overall = {"verdict": "FAIRLY VALUED", "color": "yellow",
                               "desc": "Valuation signals are mixed - limited margin of safety at the current price."}
            elif avg >= 0.4:
                mos_overall = {"verdict": "OVERVALUED", "color": "red",
                               "desc": "Most signals indicate the stock trades above intrinsic value."}
            else:
                mos_overall = {"verdict": "SIGNIFICANTLY OVERVALUED", "color": "red",
                               "desc": "Nearly all valuation methods suggest the stock is materially overpriced."}
        else:
            mos_overall = {"verdict": "INSUFFICIENT DATA", "color": "yellow",
                           "desc": "Not enough data to calculate a meaningful margin of safety score."}

        fcf_margin = (float(fcf) / float(revenue) * 100) if revenue else None
        fcf_yield  = (float(fcf) / float(market_cap) * 100) if market_cap else None

        result = {
            "ticker":        ticker,
            "currency":      currency,
            "currentPrice":  round(float(cur_price), 2),
            "fcfBase":       round(float(fcf) / 1e9, 3),
            "sharesOut":     round(float(shares_out) / 1e9, 3),
            "netDebt":       round((total_debt - cash) / 1e9, 2),
            "fcfMargin":     round(fcf_margin, 2) if fcf_margin is not None else None,
            "fcfYield":      round(fcf_yield, 2) if fcf_yield is not None else None,
            "wacc": {
                "total":        round(wacc * 100, 2),
                "costEquity":   round(cost_equity * 100, 2),
                "costDebt":     round(cost_debt * 100, 2),
                "costDebtPre":  round(cost_debt_pre * 100, 2),
                "equityWeight": round(equity_weight * 100, 1),
                "debtWeight":   round(debt_weight * 100, 1),
                "beta":         round(float(beta), 2),
                "rf":           round(RF * 100, 2),
                "erp":          round(ERP * 100, 2),
                "taxRate":      round(float(tax_rate) * 100, 1),
            },
            "scenarios":     scenario_results,
            "sensitivity": {
                "matrix":      matrix,
                "g_labels":    [f"{round(g*100,1)}%" for g in g_range],
                "wacc_labels": [f"{round(w*100,2)}%" for w in wacc_range],
                "base_g_idx":  3,
                "base_w_idx":  3,
            },
            "fcfTable":      fcf_table,
            "baseGrowth":    round(base_g * 100, 1),
            "sector":        sector,
            "assumptions": {
                "fcfSource":        fcf_source,
                "sectorBaseGrowth": round(sector_g * 100, 1),
                "revenueGrowth":    round(revenue_growth * 100, 1) if revenue_growth is not None else None,
                "earningsGrowth":   round(earnings_growth * 100, 1) if earnings_growth is not None else None,
                "terminalBase":     round(terminal_base * 100, 2),
            },
            "mos":           mos_signals,
            "mosOverall":    mos_overall,
            "graham":        graham,
            "peg":           peg,
            "evEbitda":      ev_ebitda,
            "analystUpside": analyst_up,
        }
        cache_set(cache_key, result)
        return safe_jsonify(result)

    except Exception as e:
        logger.error(f"[api_dcf] {ticker}: {e}")
        return safe_jsonify({"error": str(e)}), 500


@app.route("/api/ai_analyse", methods=["POST"])
def api_ai_analyse():
    data       = get_request_data()
    ticker     = data.get("ticker","").strip().upper()
    periode    = data.get("periode","1y")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    if not ticker:
        return safe_jsonify({"error": "No ticker"}), 400
    if not gemini_key:
        return safe_jsonify({"error": "No API key configured"}), 400

    try:
        cache_key_ai = f"ai:{ticker}:{periode}"
        cached_ai = cache_get(cache_key_ai)
        if cached_ai:
            return safe_jsonify({"ai": cached_ai})

        cache_key = f"analyse:{ticker}:{periode}"
        cached = cache_get(cache_key)

        if cached:
            fundamental = cached.get("fundamental", {})
            sig         = cached.get("signaler", {})
            ri          = cached.get("risiko", {})
            aksje = yf.Ticker(ticker)
            info_full = aksje.info
            fast_info_full = None
            try:
                fast_info_full = aksje.fast_info
            except Exception:
                fast_info_full = None
            snapshot = build_company_snapshot(aksje, info_full, fast_info_full)
        else:
            aksje = yf.Ticker(ticker)
            def fetch_stock():
                df = yf.download(ticker, period=periode, progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
            def fetch_bm():
                try:
                    bm = yf.download(BENCHMARK, period=periode, progress=False, auto_adjust=True)
                    if isinstance(bm.columns, pd.MultiIndex):
                        bm.columns = bm.columns.get_level_values(0)
                    return bm
                except Exception as e:
                    logger.warning(f"[ai_analyse] benchmark fetch failed: {e}")
                    return None
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_df = ex.submit(fetch_stock)
                f_bm = ex.submit(fetch_bm)
                df   = f_df.result()
                bm   = f_bm.result()
            if df.empty:
                return safe_jsonify({"error": f"No data for {ticker}"}), 404
            info_full = aksje.info
            try:
                fast_info_full = aksje.fast_info
            except Exception:
                fast_info_full = None
            df = beregn_tekniske(df)
            sig = hent_signaler(df)
            bm_df = bm if (bm is not None and not bm.empty) else df.copy()
            ri = beregn_risiko(df, bm_df)

            latest_close = safe_float(df["Close"].dropna().iloc[-1]) if not df["Close"].dropna().empty else None
            snapshot = build_company_snapshot(aksje, info_full, fast_info_full)
            current_price = first_valid_number(snapshot["current_price"], latest_close)
            shares_out = snapshot["shares_out"]
            market_cap = first_valid_number(
                snapshot["market_cap"],
                current_price * shares_out if current_price and shares_out else None,
            )
            free_cashflow = snapshot["free_cashflow"]
            revenue_growth = snapshot["revenue_growth"]
            earnings_growth = snapshot["earnings_growth"]
            year_high = snapshot["year_high"]
            year_low = snapshot["year_low"]
            target_mean = snapshot["target_mean"]
            target_high = snapshot["target_high"]
            target_low = snapshot["target_low"]
            fcf_yield = (free_cashflow / market_cap * 100) if free_cashflow and market_cap else None

            fundamental = {
                "navn":             info_full.get("longName", ticker),
                "pe":               safe(info_full.get("trailingPE")),
                "forward_pe":       safe(info_full.get("forwardPE")),
                "mktcap":           stor_tall(market_cap),
                "roe":              safe(info_full.get("returnOnEquity"), pst=True),
                "driftsmargin":     safe(snapshot["operating_margin"], pst=True),
                "bruttomargin":     safe(snapshot["gross_margin"], pst=True),
                "nettmargin":       safe(snapshot["net_margin"], pst=True),
                "fcf_yield":        safe(fcf_yield, pst=True),
                "frikontant":       stor_tall(free_cashflow),
                "revenue_growth":   safe(revenue_growth, pst=True),
                "earnings_growth":  safe(earnings_growth, pst=True),
                "52u_høy":          safe(year_high),
                "52u_lav":          safe(year_low),
                "mål":              safe(target_mean),
                "mål_høy":          safe(target_high),
                "mål_lav":          safe(target_low),
                "analyst_count":    info_full.get("numberOfAnalystOpinions") or 0,
                "konsensus":        info_full.get("recommendationKey","N/A").upper(),
                "_pris_raw":        current_price,
                "_52w_high_raw":    year_high,
                "_52w_low_raw":     year_low,
                "_target_mean_raw": target_mean,
                "_fcf_source":      snapshot["fcf_source"],
                "_debt_to_equity_raw": snapshot["debt_to_equity"],
            }

        info = info_full
        fundamental.setdefault("_pris_raw", snapshot["current_price"])
        fundamental.setdefault("_52w_high_raw", snapshot["year_high"])
        fundamental.setdefault("_52w_low_raw", snapshot["year_low"])
        fundamental.setdefault("_target_mean_raw", snapshot["target_mean"])
        fundamental.setdefault("_fcf_source", snapshot["fcf_source"])
        fundamental.setdefault("_debt_to_equity_raw", snapshot["debt_to_equity"])
        price_now = first_valid_number(info.get("currentPrice"), info.get("regularMarketPrice"), fundamental.get("_pris_raw"))
        year_high = first_valid_number(info.get("fiftyTwoWeekHigh"), fundamental.get("_52w_high_raw"))
        year_low = first_valid_number(info.get("fiftyTwoWeekLow"), fundamental.get("_52w_low_raw"))
        range_position = None
        if price_now is not None and year_high is not None and year_low is not None and year_high > year_low:
            range_position = round((price_now - year_low) / (year_high - year_low) * 100, 1)

        # ── Analyst consensus + upgrades/downgrades ───────────────────────────
        analyst_ctx = ""
        upgrades_ctx = ""
        try:
            target_mean = first_valid_number(info.get("targetMeanPrice"), fundamental.get("_target_mean_raw"))
            target_mean = first_valid_number(target_mean, snapshot.get("target_mean"))
            target_high = first_valid_number(info.get("targetHighPrice"), snapshot.get("target_high"))
            target_low  = first_valid_number(info.get("targetLowPrice"), snapshot.get("target_low"))
            upside      = round((target_mean - price_now) / price_now * 100, 1) if target_mean and price_now else None
            n_analysts  = info.get("numberOfAnalystOpinions", 0) or 0
            strong_buy  = info.get("numberOfStrongBuyOpinions", 0) or info.get("strongBuy", 0)
            buy         = info.get("numberOfBuyOpinions", 0) or info.get("buy", 0)
            hold        = info.get("numberOfHoldOpinions", 0) or info.get("hold", 0)
            sell        = info.get("numberOfSellOpinions", 0) or info.get("sell", 0)
            strong_sell = info.get("numberOfStrongSellOpinions", 0) or info.get("strongSell", 0)
            analyst_ctx = (
                f"Analyst consensus: {n_analysts} analysts. Distribution: Strong Buy={strong_buy}, Buy={buy}, "
                f"Hold={hold}, Sell={sell}, Strong Sell={strong_sell}. "
                f"Targets: low={safe(target_low)}, mean={safe(target_mean)}, high={safe(target_high)}. "
                f"Mean-target upside={upside}%."
            )
            # Upgrades/downgrades
            ACTION_MAP_AI = {
                "main": "MAINTAIN", "reit": "REITERATE", "init": "INITIATED",
                "up": "UPGRADE", "down": "DOWNGRADE", "upgrade": "UPGRADE",
                "downgrade": "DOWNGRADE", "resume": "RESUMED", "suspend": "SUSPENDED",
                "coverage": "INITIATED", "new": "INITIATED",
            }
            GRADE_RANK = {
                "strong buy": 5, "outperform": 4, "overweight": 4, "buy": 4,
                "market perform": 3, "neutral": 3, "equal-weight": 3, "hold": 3, "sector perform": 3,
                "underperform": 2, "underweight": 2, "sell": 1, "strong sell": 1,
            }
            ug = aksje.upgrades_downgrades
            if ug is not None and not ug.empty:
                ug = ug.sort_index(ascending=False).head(100)
                recent_actions = []
                for idx, row in ug.iterrows():
                    raw_action = str(row.get("Action", "")).lower().strip()
                    from_grade = str(row.get("From Grade", "")).strip()
                    to_grade   = str(row.get("To Grade", "")).strip()
                    if from_grade.lower() in ("nan", "none", ""): from_grade = ""
                    if to_grade.lower()   in ("nan", "none", ""): to_grade   = ""
                    if raw_action in ("main", "reit") and from_grade and to_grade and from_grade.lower() != to_grade.lower():
                        f = GRADE_RANK.get(from_grade.lower(), 3)
                        t = GRADE_RANK.get(to_grade.lower(), 3)
                        if t > f:   raw_action = "upgrade"
                        elif t < f: raw_action = "downgrade"
                    label = ACTION_MAP_AI.get(raw_action, raw_action.upper())
                    if label in ("MAINTAIN", "REITERATE") and not (from_grade or to_grade):
                        continue
                    rating_str = f"{from_grade} → {to_grade}" if (from_grade and to_grade and from_grade.lower() != to_grade.lower()) else (to_grade or from_grade)
                    firm = str(row.get("Firm", ""))[:25]
                    date_str = str(idx)[:10]
                    recent_actions.append(f"{date_str} {firm}: {label}" + (f" ({rating_str})" if rating_str else ""))
                    if len(recent_actions) >= 8:
                        break
                if recent_actions:
                    upgrades_ctx = "Recent analyst actions: " + "; ".join(recent_actions) + "."
        except Exception as e:
            logger.warning(f"[ai_analyse] analyst context: {e}")

        # ── Insider activity ──────────────────────────────────────────────────
        insider_ctx = ""
        try:
            ins = aksje.insider_transactions
            if ins is not None and not ins.empty:
                ins = ins.head(10)
                buys = sum(
                    1 for _, r in ins.iterrows()
                    if "buy" in str(r.get("Transaction", "")).lower() or "purchase" in str(r.get("Transaction", "")).lower()
                )
                sells = len(ins) - buys
                recent = [
                    f"{str(r.get('Insider',''))[:20]} ({str(r.get('Position',''))[:15]}): {str(r.get('Transaction',''))}"
                    for _, r in ins.head(5).iterrows()
                ]
                insider_ctx = f"Insider activity: {buys} buys and {sells} sells across the latest 10 records. Recent: {'; '.join(recent)}."
        except Exception as e:
            logger.warning(f"[ai_analyse] insider context: {e}")

        # ── Earnings: next date + full EPS history ────────────────────────────
        earnings_ctx = ""
        next_earnings_ctx = ""
        try:
            cal = aksje.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed:
                        next_date_val = str(ed[0])[:10] if hasattr(ed, "__len__") else str(ed)[:10]
                        next_earnings_ctx = f"Next earnings date: {next_date_val}."
                elif hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                    next_earnings_ctx = f"Next earnings date: {str(cal['Earnings Date'].iloc[0])[:10]}."
        except Exception as e:
            logger.warning(f"[ai_analyse] next earnings date: {e}")
        try:
            ei = aksje.earnings_history
            if ei is not None and not ei.empty:
                beats = misses = 0
                eps_rows = []
                for idx, row in ei.head(8).iterrows():
                    actual = row.get("epsActual") or row.get("Reported EPS")
                    est    = row.get("epsEstimate") or row.get("EPS Estimate")
                    if actual is not None and est is not None and actual == actual and est == est:
                        a, e_ = float(actual), float(est)
                        beat = a >= e_
                        beats += int(beat)
                        misses += int(not beat)
                        eps_rows.append(f"{str(idx)[:10]}: actual={round(a,2)}, est={round(e_,2)}, {'BEAT' if beat else 'MISS'}")
                summary = f"{beats} beats and {misses} misses over latest {len(eps_rows)} quarters."
                if eps_rows:
                    summary += " Detail: " + "; ".join(eps_rows[:6]) + "."
                earnings_ctx = summary
        except Exception as e:
            logger.warning(f"[ai_analyse] earnings context: {e}")

        # ── Short interest ────────────────────────────────────────────────────
        short_ctx = ""
        try:
            short_pct = info.get("shortPercentOfFloat")
            short_ratio = info.get("shortRatio")
            shares_short = info.get("sharesShort")
            shares_short_prev = info.get("sharesShortPriorMonth")
            if short_pct is not None:
                short_pct_str = f"{round(float(short_pct) * 100, 1)}%"
                change_str = ""
                if shares_short and shares_short_prev and shares_short_prev > 0:
                    chg = (shares_short - shares_short_prev) / shares_short_prev * 100
                    change_str = f", month-over-month change {chg:+.1f}%"
                short_ctx = f"Short interest: {short_pct_str} of float shorted, days-to-cover={short_ratio}{change_str}."
        except Exception as e:
            logger.warning(f"[ai_analyse] short context: {e}")

        # ── Options flow: enriched with sentiment label + top strikes ─────────
        options_ctx = ""
        try:
            exps = aksje.options
            if exps:
                all_call_vol = all_put_vol = all_call_oi = all_put_oi = 0
                for exp in exps[:4]:
                    try:
                        chain = aksje.option_chain(exp)
                        all_call_vol += chain.calls["volume"].fillna(0).sum()
                        all_put_vol  += chain.puts["volume"].fillna(0).sum()
                        all_call_oi  += chain.calls["openInterest"].fillna(0).sum()
                        all_put_oi   += chain.puts["openInterest"].fillna(0).sum()
                    except Exception as e:
                        logger.warning(f"[ai_analyse] options chain {exp}: {e}")
                pc_vol = round(all_put_vol / max(all_call_vol, 1), 2)
                pc_oi  = round(all_put_oi  / max(all_call_oi, 1), 2)
                if pc_vol > 1.5:   flow_sentiment = "VERY BEARISH"
                elif pc_vol > 1.1: flow_sentiment = "BEARISH"
                elif pc_vol < 0.5: flow_sentiment = "VERY BULLISH"
                elif pc_vol < 0.8: flow_sentiment = "BULLISH"
                else:              flow_sentiment = "NEUTRAL"
                atm_iv_str = ""
                top_calls_str = ""
                top_puts_str = ""
                try:
                    chain0 = aksje.option_chain(exps[0])
                    calls = chain0.calls.dropna(subset=["impliedVolatility"])
                    if price_now and not calls.empty:
                        atm = calls.iloc[(calls["strike"] - price_now).abs().argsort()[:1]]
                        atm_iv = round(float(atm["impliedVolatility"].values[0]) * 100, 1)
                        atm_iv_str = f", ATM IV={atm_iv}%"
                    top_c = chain0.calls.dropna(subset=["openInterest"]).sort_values("openInterest", ascending=False).head(3)
                    top_p = chain0.puts.dropna(subset=["openInterest"]).sort_values("openInterest", ascending=False).head(3)
                    top_calls_str = ", ".join([f"${r['strike']:.0f}(OI={int(r['openInterest'])})" for _, r in top_c.iterrows()])
                    top_puts_str  = ", ".join([f"${r['strike']:.0f}(OI={int(r['openInterest'])})" for _, r in top_p.iterrows()])
                except Exception as e:
                    logger.warning(f"[ai_analyse] options detail: {e}")
                options_ctx = (
                    f"Options: sentiment={flow_sentiment}, put/call volume={pc_vol}, put/call OI={pc_oi}{atm_iv_str}."
                )
                if top_calls_str: options_ctx += f" Top call strikes (by OI): {top_calls_str}."
                if top_puts_str:  options_ctx += f" Top put strikes (by OI): {top_puts_str}."
        except Exception as e:
            logger.warning(f"[ai_analyse] options context: {e}")

        # ── DCF: use cache or compute base scenario inline ────────────────────
        dcf_ctx = ""
        cached_dcf = cache_get(f"dcf:{ticker}")
        if not cached_dcf:
            try:
                _dcf_snap = build_company_snapshot(aksje, info, None)
                _fcf = _dcf_snap["free_cashflow"]
                _shares = _dcf_snap["shares_out"]
                _price  = _dcf_snap["current_price"]
                if _fcf and _fcf > 0 and _shares and _price:
                    _wacc = 0.09
                    _g5   = 0.07
                    _tg   = 0.025
                    _pv = sum(_fcf * (1 + _g5) ** i / (1 + _wacc) ** i for i in range(1, 6))
                    _tv = (_fcf * (1 + _g5) ** 5 * (1 + _tg)) / (_wacc - _tg) / (1 + _wacc) ** 5
                    _iv = round((_pv + _tv - (_dcf_snap["total_debt"] - _dcf_snap["cash"])) / _shares, 2)
                    _upside = round((_iv - _price) / _price * 100, 1) if _price else None
                    dcf_ctx = f"DCF quick estimate (base, default WACC={round(_wacc*100,1)}%, growth={round(_g5*100,1)}%): intrinsic value≈{_iv}, upside≈{_upside}%. Note: run full DCF tab for bear/bull scenarios and sensitivity."
            except Exception as e:
                logger.warning(f"[ai_analyse] inline DCF: {e}")
        else:
            try:
                scenarios  = cached_dcf.get("scenarios", {})
                base_case  = scenarios.get("Base", {})
                bear_case  = scenarios.get("Bear", {})
                bull_case  = scenarios.get("Bull", {})
                mos_overall = cached_dcf.get("mosOverall", {})
                dcf_ctx = (
                    f"DCF scenarios — Bear: intrinsic={bear_case.get('intrinsic')}, upside={bear_case.get('upside')}%; "
                    f"Base: intrinsic={base_case.get('intrinsic')}, upside={base_case.get('upside')}%; "
                    f"Bull: intrinsic={bull_case.get('intrinsic')}, upside={bull_case.get('upside')}%. "
                    f"WACC={base_case.get('wacc')}%, 5yr growth={base_case.get('g5')}%, terminal={base_case.get('terminal_g')}%. "
                    f"Margin of safety verdict: {mos_overall.get('verdict','N/A')} — {mos_overall.get('desc','')}"
                )
            except Exception:
                dcf_ctx = ""

        # ── News headlines ────────────────────────────────────────────────────
        news_ctx = ""
        try:
            nyheter = aksje.news or []
            titles = []
            for n in nyheter[:8]:
                content = n.get("content", {})
                title = content.get("title") or n.get("title", "")
                if title:
                    titles.append(title)
            if titles:
                news_ctx = "Recent headlines: " + " | ".join(titles[:6]) + "."
        except Exception as e:
            logger.warning(f"[ai_analyse] news context: {e}")

        # ── Macro backdrop (from shared cache) ────────────────────────────────
        macro_ctx = ""
        try:
            macro_data = cache_get("makro")
            if macro_data:
                MACRO_SHOW = {"S&P 500", "Nasdaq", "VIX", "US 10y Yield", "Oil (Brent)", "USD/NOK"}
                items = []
                for item in macro_data:
                    name = item.get("navn", "")
                    if name in MACRO_SHOW:
                        val = item.get("verdi", "N/A")
                        chg = item.get("endring", 0) or 0
                        items.append(f"{name}={val} ({chg:+.2f}%)")
                if items:
                    macro_ctx = "Macro backdrop: " + ", ".join(items) + "."
        except Exception as e:
            logger.warning(f"[ai_analyse] macro context: {e}")

        # ── Build context strings ─────────────────────────────────────────────
        valuation_ctx = (
            f"Price={safe(price_now)}, P/E={fundamental.get('pe','N/A')}, Forward P/E={fundamental.get('forward_pe','N/A')}, "
            f"P/B={safe(info.get('priceToBook'))}, P/S={safe(info.get('priceToSalesTrailing12Months'))}, "
            f"PEG={safe(info.get('pegRatio'))}, EV/EBITDA={safe(info.get('enterpriseToEbitda'))}, "
            f"Dividend Yield={safe(info.get('dividendYield'), pst=True)}, "
            f"Market Cap={fundamental.get('mktcap','N/A')}, FCF={fundamental.get('frikontant','N/A')}, "
            f"FCF Yield={fundamental.get('fcf_yield','N/A')}, FCF Source={fundamental.get('_fcf_source', snapshot.get('fcf_source','N/A'))}, "
            f"Target Mean={safe(target_mean)}, Target Range={safe(target_low)} to {safe(target_high)}."
        )
        quality_ctx = (
            f"ROE={fundamental.get('roe','N/A')}, Gross Margin={fundamental.get('bruttomargin','N/A')}, "
            f"Operating Margin={fundamental.get('driftsmargin','N/A')}, Net Margin={fundamental.get('nettmargin','N/A')}, "
            f"Revenue Growth={fundamental.get('revenue_growth','N/A')}, Earnings Growth={fundamental.get('earnings_growth','N/A')}, "
            f"Debt/Equity={safe(fundamental.get('_debt_to_equity_raw', snapshot.get('debt_to_equity')))}."
        )
        bb_pct_display = round(float(sig.get("bb_pct", 0.5)) * 100, 1) if sig.get("bb_pct") is not None else "N/A"
        technical_ctx = (
            f"RSI={sig.get('rsi','N/A')}, MACD={'Bullish' if sig.get('macd_bull') else 'Bearish'}, "
            f"Price vs SMA20={'Above' if sig.get('over_sma20') else 'Below'}, "
            f"Price vs SMA50={'Above' if sig.get('over_sma50') else 'Below'}, "
            f"Price vs SMA200={'Above' if sig.get('over_sma200') else 'Below'}, "
            f"Bollinger Band position={bb_pct_display}% (0%=lower band, 100%=upper band), "
            f"ATR={sig.get('atr','N/A')}, "
            f"52-week range position={range_position if range_position is not None else 'N/A'}%."
        )
        risk_ctx = (
            f"Sharpe={ri.get('sharpe','N/A')}, Sortino={ri.get('sortino','N/A')}, Calmar={ri.get('calmar','N/A')}, "
            f"CAGR={ri.get('cagr','N/A')}%, Annualised Vol={ri.get('vol','N/A')}%, "
            f"Max Drawdown={ri.get('max_dd','N/A')}%, VaR 95%={ri.get('var95','N/A')}%, VaR 99%={ri.get('var99','N/A')}%, "
            f"Beta={ri.get('beta','N/A')}, Alpha={ri.get('alpha','N/A')}%, Risk-free rate={ri.get('risk_free','N/A')}%."
        )

        prompt = f"""
You are a senior equity analyst writing a disciplined, evidence-first stock note for {fundamental.get('navn', ticker)} ({ticker}).

GROUND RULES:
{AI_GUARDRAILS}

{f"MACRO BACKDROP: {macro_ctx}" if macro_ctx else ""}

VALUATION SNAPSHOT:
{valuation_ctx}

BUSINESS QUALITY SNAPSHOT:
{quality_ctx}

TECHNICAL SNAPSHOT:
{technical_ctx}

RISK SNAPSHOT:
{risk_ctx}

{f"DCF VALUATION: {dcf_ctx}" if dcf_ctx else ""}
{f"ANALYST CONSENSUS: {analyst_ctx}" if analyst_ctx else ""}
{f"RECENT ANALYST ACTIONS: {upgrades_ctx}" if upgrades_ctx else ""}
{f"INSIDER ACTIVITY: {insider_ctx}" if insider_ctx else ""}
{f"UPCOMING CATALYST: {next_earnings_ctx}" if next_earnings_ctx else ""}
{f"EARNINGS TRACK RECORD: {earnings_ctx}" if earnings_ctx else ""}
{f"SHORT INTEREST: {short_ctx}" if short_ctx else ""}
{f"OPTIONS POSITIONING: {options_ctx}" if options_ctx else ""}
{f"RECENT NEWS: {news_ctx}" if news_ctx else ""}

Write the response in this exact structure:
1. **Investment Setup** — 2-3 sentences summarizing valuation, quality, trend, and whether the evidence is aligned or mixed
2. **Bull Case** — 3 bullet points, each tied to a specific metric or data point above
3. **Bear Case** — 3 bullet points, each tied to a specific metric, risk measure, or data gap above
4. **Positioning Check** — 1-2 sentences on analysts, recent upgrades/downgrades, insiders, short interest, and options; if sparse, say the positioning evidence is limited
5. **What Would Change The View** — 1-2 sentences on the next metric, trend, or upcoming event that would materially improve or weaken the thesis
6. **Stance** — one line: BUY / HOLD / SELL plus confidence: Low, Medium, or High

Confidence rules:
- High only if valuation, quality, trend, and positioning are mostly aligned with limited missing data.
- Medium if the evidence is mixed but still actionable.
- Low if the evidence is conflicting or key data is missing.

Additional rules:
{AI_GUARDRAILS}
- Stay anchored to the supplied data only.
- If a metric is unavailable, state that it is unavailable instead of inferring it.
- Do not mention catalysts, schedules, or price levels unless they are explicitly present in the supplied context.
- Mixed signals are normal in markets — identify the dominant signal and commit to a directional call. Avoid HOLD unless the evidence is genuinely deadlocked.
- BUY when the weight of technicals, valuation, and quality metrics favor upside. SELL when multiple risk factors and valuation signals are negative. HOLD only as a last resort when evidence is truly 50/50.
- Keep the tone sharp, direct, and professional. Max 380 words.
"""
        ai_tekst = spør_ai(prompt, gemini_key, 1500)
        cache_set(cache_key_ai, ai_tekst)
        return safe_jsonify({"ai": ai_tekst})

    except Exception as e:
        logger.error(f"[api_ai_analyse] {ticker}: {e}")
        return safe_jsonify({"error": str(e)}), 500

@app.route("/api/short_interest", methods=["POST"])
def api_short_interest():
    ticker = get_request_data().get("ticker","").strip().upper()
    if not ticker:
        return safe_jsonify({"error": "No ticker"}), 400
    try:
        aksje = yf.Ticker(ticker)
        info  = aksje.info
        warnings_list = []

        short_pct         = info.get("shortPercentOfFloat")
        short_ratio       = info.get("shortRatio")
        shares_short      = info.get("sharesShort")
        shares_short_prev = info.get("sharesShortPriorMonth")
        date_short        = info.get("dateShortInterest")
        float_shares      = info.get("floatShares")
        shares_out        = info.get("sharesOutstanding")

        # Month-over-month change
        mom_change = None
        if shares_short and shares_short_prev and shares_short_prev > 0:
            mom_change = round((shares_short - shares_short_prev) / shares_short_prev * 100, 1)

        # Interpret short level
        if short_pct is not None:
            pct = float(short_pct) * 100
            if pct > 20:   sentiment = ("HIGH", "var(--red)")
            elif pct > 10: sentiment = ("ELEVATED", "var(--yellow)")
            elif pct > 5:  sentiment = ("MODERATE", "var(--text3)")
            else:          sentiment = ("LOW", "var(--green)")
        else:
            sentiment = ("N/A", "var(--text2)")

        return safe_jsonify({
            "shortPct":        round(float(short_pct)*100, 2) if short_pct else None,
            "shortRatio":      round(float(short_ratio), 1) if short_ratio else None,
            "sharesShort":     stor_tall(shares_short) if shares_short else "N/A",
            "sharesShortPrev": stor_tall(shares_short_prev) if shares_short_prev else "N/A",
            "momChange":       mom_change,
            "floatShares":     stor_tall(float_shares) if float_shares else "N/A",
            "sharesOut":       stor_tall(shares_out) if shares_out else "N/A",
            "dateShort":       str(pd.Timestamp(date_short, unit='s'))[:10] if date_short else "N/A",
            "sentiment":       sentiment[0],
            "sentimentColor":  sentiment[1],
        })
    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500


@app.route("/api/options_flow", methods=["POST"])
def api_options_flow():
    ticker = get_request_data().get("ticker","").strip().upper()
    if not ticker:
        return safe_jsonify({"error": "No ticker"}), 400
    try:
        aksje     = yf.Ticker(ticker)
        info      = aksje.info
        exps      = aksje.options
        cur_price = info.get("currentPrice") or info.get("regularMarketPrice")
        warnings_list = []

        if not exps:
            return safe_jsonify({"error": "No options data available for this ticker"}), 404

        # Aggregate across first 4 expirations for flow summary
        total_call_vol, total_put_vol = 0, 0
        total_call_oi,  total_put_oi  = 0, 0
        expirations_data = []

        for exp in exps[:3]:
            try:
                chain = aksje.option_chain(exp)
                cv = chain.calls['volume'].fillna(0).sum()
                pv = chain.puts['volume'].fillna(0).sum()
                ci = chain.calls['openInterest'].fillna(0).sum()
                pi = chain.puts['openInterest'].fillna(0).sum()
                total_call_vol += cv
                total_put_vol  += pv
                total_call_oi  += ci
                total_put_oi   += pi
                expirations_data.append({
                    "exp":      exp,
                    "callVol":  int(cv),
                    "putVol":   int(pv),
                    "callOI":   int(ci),
                    "putOI":    int(pi),
                    "pcVol":    round(pv / max(cv, 1), 2),
                })
            except Exception as e:
                record_warning(warnings_list, f"Options data incomplete for expiry {exp}", e)

        pc_vol = round(total_put_vol / max(total_call_vol, 1), 2)
        pc_oi  = round(total_put_oi  / max(total_call_oi,  1), 2)

        # Sentiment interpretation
        if pc_vol > 1.5:   flow_sentiment = ("VERY BEARISH", "var(--red)")
        elif pc_vol > 1.1: flow_sentiment = ("BEARISH",      "var(--red)")
        elif pc_vol < 0.5: flow_sentiment = ("VERY BULLISH", "var(--green)")
        elif pc_vol < 0.8: flow_sentiment = ("BULLISH",      "var(--green)")
        else:              flow_sentiment = ("NEUTRAL",       "var(--yellow)")

        # ATM implied volatility
        atm_iv = None
        try:
            chain0 = aksje.option_chain(exps[0])
            calls  = chain0.calls.dropna(subset=['impliedVolatility'])
            if cur_price and not calls.empty:
                atm  = calls.iloc[(calls['strike'] - cur_price).abs().argsort()[:1]]
                atm_iv = round(float(atm['impliedVolatility'].values[0]) * 100, 1)
        except Exception as e:
            record_warning(warnings_list, "ATM implied volatility unavailable", e)

        # Top 5 calls and puts by open interest (all strikes, nearest expiry)
        top_calls, top_puts = [], []
        try:
            chain0 = aksje.option_chain(exps[0])
            for _, r in chain0.calls.sort_values('openInterest', ascending=False).head(5).iterrows():
                top_calls.append({
                    "strike": round(float(r['strike']), 2),
                    "oi":     int(r['openInterest']) if r['openInterest'] == r['openInterest'] else 0,
                    "vol":    int(r['volume']) if r['volume'] == r['volume'] else 0,
                    "iv":     round(float(r['impliedVolatility'])*100, 1) if r['impliedVolatility'] == r['impliedVolatility'] else None,
                })
            for _, r in chain0.puts.sort_values('openInterest', ascending=False).head(5).iterrows():
                top_puts.append({
                    "strike": round(float(r['strike']), 2),
                    "oi":     int(r['openInterest']) if r['openInterest'] == r['openInterest'] else 0,
                    "vol":    int(r['volume']) if r['volume'] == r['volume'] else 0,
                    "iv":     round(float(r['impliedVolatility'])*100, 1) if r['impliedVolatility'] == r['impliedVolatility'] else None,
                })
        except Exception as e:
            record_warning(warnings_list, "Top open interest strikes unavailable", e)

        return safe_jsonify({
            "currentPrice":    round(float(cur_price), 2) if cur_price else None,
            "pcVolumeRatio":   pc_vol,
            "pcOIRatio":       pc_oi,
            "totalCallVol":    int(total_call_vol),
            "totalPutVol":     int(total_put_vol),
            "totalCallOI":     int(total_call_oi),
            "totalPutOI":      int(total_put_oi),
            "atmIV":           atm_iv,
            "sentiment":       flow_sentiment[0],
            "sentimentColor":  flow_sentiment[1],
            "expirations":     expirations_data,
            "topCalls":        top_calls,
            "topPuts":         top_puts,
            "nearestExp":      exps[0],
            "warnings":        warnings_list,
        })
    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500


@app.route("/api/markeds_oversikt", methods=["POST"])
def api_markeds_oversikt():
    data     = get_request_data()
    makro    = data.get("makro", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        return safe_jsonify({"error": "No API key configured"}), 400
    try:
        cache_key = f"markeds_oversikt:{hashlib.md5(makro.encode('utf-8')).hexdigest()}"
        cached = cache_get(cache_key)
        if cached:
            return safe_jsonify({"ai": cached})

        today = datetime.now().strftime("%A, %B %d, %Y")
        prompt = f"""
You are a senior macro strategist providing a daily market briefing for professional investors at The Modern Sail.

Today is {today}.

CURRENT MARKET DATA:
{makro}

Write a concise professional market briefing covering:
1. **Market Pulse** — overall risk-on/risk-off tone and what's driving it today
2. **Key Movers** — which instruments stand out and why (focus on unusual moves)
3. **Fixed Income & FX** — what yields and currency moves signal about macro expectations
4. **Watchlist** — the next macro themes or cross-asset signals investors should monitor, based only on the market data provided
5. **Modern Sail View** — one actionable takeaway for equity investors today

Keep it professional, data-driven, and concise. Max 300 words.

FINAL INSTRUCTION:
{AI_GUARDRAILS}
- Base the note only on the market data shown above.
- If the tape is mixed, say it is mixed.
- For the watchlist section, mention only categories or themes to monitor next; do not invent dated calendar events unless they were provided in the input.
- Prefer cross-asset interpretation over generic macro commentary.
"""
        ai_tekst = spør_ai(prompt, gemini_key, 1200)
        cache_set(cache_key, ai_tekst)
        return safe_jsonify({"ai": ai_tekst})
    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500


@app.route("/api/makro", methods=["GET"])
def api_makro():
    cached = cache_get("makro")
    if cached:
        return safe_jsonify(cached)

    symbol_string = " ".join(meta["yf"] for meta in MAKRO_ASSETS.values())
    try:
        hist = yf.download(symbol_string, period="6mo", interval="1d", progress=False, auto_adjust=False, threads=True)
    except Exception:
        hist = None

    def fallback_row(name, meta):
        try:
            ticker = yf.Ticker(meta["yf"])
            fi = ticker.fast_info
            raw_last = first_valid_number(fi.last_price, fi.previous_close)
            history = ticker.history(period="6mo", auto_adjust=False)
            series = normalize_macro_series(meta["yf"], get_download_close_series(history, meta["yf"]))
            if series.empty and raw_last is not None:
                series = pd.Series([raw_last], dtype=float)
            last_val = first_valid_number(raw_last, series.iloc[-1] if not series.empty else None)
            prev_val = first_valid_number(fi.previous_close, series.iloc[-2] if len(series) > 1 else last_val)
        except Exception:
            series = pd.Series(dtype=float)
            last_val = None
            prev_val = None

        if last_val is None:
            return {
                "navn": name,
                "verdi": "N/A",
                "raw": None,
                "endring": 0,
                "historikk": [],
                "analysisSymbol": meta["analysis"],
                "tvSymbol": meta["tv"],
            }

        if prev_val in (None, 0):
            prev_val = last_val
        endring = (last_val - prev_val) / prev_val * 100 if prev_val else 0
        history_vals = [round(float(v), 4 if abs(float(v)) < 1 else 2) for v in series.tail(60)]
        return {
            "navn": name,
            "verdi": format_macro_value(name, last_val),
            "raw": round(float(last_val), 4 if abs(float(last_val)) < 1 else 2),
            "endring": round(endring, 2),
            "historikk": history_vals,
            "analysisSymbol": meta["analysis"],
            "tvSymbol": meta["tv"],
        }

    rader = []
    for name, meta in MAKRO_ASSETS.items():
        series = normalize_macro_series(meta["yf"], get_download_close_series(hist, meta["yf"]))
        if series.empty:
            rader.append(fallback_row(name, meta))
            continue

        last_val = first_valid_number(series.iloc[-1])
        prev_val = first_valid_number(series.iloc[-2] if len(series) > 1 else series.iloc[-1])
        if last_val is None:
            rader.append(fallback_row(name, meta))
            continue

        prev_val = prev_val if prev_val not in (None, 0) else last_val
        endring = (last_val - prev_val) / prev_val * 100 if prev_val else 0
        history_vals = [round(float(v), 4 if abs(float(v)) < 1 else 2) for v in series.tail(60)]
        rader.append({
            "navn": name,
            "verdi": format_macro_value(name, last_val),
            "raw": round(float(last_val), 4 if abs(float(last_val)) < 1 else 2),
            "endring": round(endring, 2),
            "historikk": history_vals,
            "analysisSymbol": meta["analysis"],
            "tvSymbol": meta["tv"],
        })

    cache_set("makro", rader)
    return safe_jsonify(rader)

@app.route("/api/earnings", methods=["POST"])
def api_earnings():
    ticker = get_request_data().get("ticker","").strip().upper()
    if not ticker:
        return safe_jsonify({"error": "No ticker provided"}), 400
    warnings_list = []
    try:
        aksje = yf.Ticker(ticker)

        # Next earnings date
        next_date  = None
        next_time  = None
        next_est   = None
        try:
            cal = aksje.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed:
                        next_date = str(ed[0])[:10] if hasattr(ed, '__len__') else str(ed)[:10]
                elif hasattr(cal, 'columns'):
                    if "Earnings Date" in cal.columns:
                        next_date = str(cal["Earnings Date"].iloc[0])[:10]
                    elif "Earnings Date" in cal.index:
                        next_date = str(cal.loc["Earnings Date"].iloc[0])[:10]
        except Exception as e:
            record_warning(warnings_list, "Next earnings date unavailable", e)

        # Earnings history
        history = []
        try:
            ei = aksje.earnings_history
            if ei is not None and not ei.empty:
                ei = ei.sort_index(ascending=False).head(8)
                for idx, row in ei.iterrows():
                    actual   = round(float(row.get("epsActual",   row.get("Reported EPS", None))), 4) if row.get("epsActual",   row.get("Reported EPS", None)) is not None else None
                    estimate = round(float(row.get("epsEstimate", row.get("EPS Estimate", None))), 4) if row.get("epsEstimate", row.get("EPS Estimate", None)) is not None else None
                    try:
                        actual   = None if (actual   is not None and (actual   != actual))   else actual
                        estimate = None if (estimate is not None and (estimate != estimate)) else estimate
                    except Exception as e:
                        record_warning(warnings_list, f"Earnings row cleanup failed for {str(idx)[:10]}", e)
                    history.append({
                        "date":     str(idx)[:10],
                        "actual":   actual,
                        "estimate": estimate,
                    })
        except Exception as primary_error:
            record_warning(warnings_list, "Primary earnings history source unavailable", primary_error)
            try:
                ei = aksje.earnings_dates
                if ei is not None and not ei.empty:
                    ei = ei.dropna(how="all").head(8)
                    for idx, row in ei.iterrows():
                        actual   = float(row["Reported EPS"])  if "Reported EPS"  in row and row["Reported EPS"]  == row["Reported EPS"]  else None
                        estimate = float(row["EPS Estimate"])  if "EPS Estimate"  in row and row["EPS Estimate"]  == row["EPS Estimate"]  else None
                        history.append({"date": str(idx)[:10], "actual": actual, "estimate": estimate})
            except Exception as fallback_error:
                record_warning(warnings_list, "Fallback earnings history source unavailable", fallback_error)

        return safe_jsonify({
            "next_earnings": {"date": next_date, "time": next_time, "estimate": next_est} if next_date else None,
            "history": history,
            "warnings": warnings_list,
        })
    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500

@app.route("/api/analyst", methods=["POST"])
def api_analyst():
    ticker = get_request_data().get("ticker","").strip().upper()
    if not ticker:
        return safe_jsonify({"error": "No ticker provided"}), 400
    warnings_list = []
    try:
        aksje = yf.Ticker(ticker)
        info  = aksje.info

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        target_mean   = info.get("targetMeanPrice")
        target_high   = info.get("targetHighPrice")
        target_low    = info.get("targetLowPrice")
        currency      = info.get("currency","")

        upside = None
        if current_price and target_mean:
            upside = round((target_mean - current_price) / current_price * 100, 1)

        # Ratings
        strong_buy  = info.get("numberOfAnalystOpinions")
        rec         = info.get("recommendationKey","")
        buy         = info.get("recommendationMean")

        # Detailed counts from analyst ratings
        try:
            ratings = aksje.analyst_price_targets
        except: ratings = None

        # Upgrades/downgrades — map Yahoo action codes to readable labels
        ACTION_MAP = {
            "main":      ("MAINTAIN",  "var(--text2)", 1),
            "reit":      ("REITERATE", "var(--text2)", 1),
            "init":      ("INITIATED", "var(--blue)",  3),
            "up":        ("UPGRADE",   "var(--green)", 4),
            "down":      ("DOWNGRADE", "var(--red)",   4),
            "upgrade":   ("UPGRADE",   "var(--green)", 4),
            "downgrade": ("DOWNGRADE", "var(--red)",   4),
            "resume":    ("RESUMED",   "var(--blue)",  3),
            "suspend":   ("SUSPENDED", "var(--yellow)",3),
            "coverage":  ("INITIATED", "var(--blue)",  3),
            "new":       ("INITIATED", "var(--blue)",  3),
        }

        upgrades = []
        try:
            ug = aksje.upgrades_downgrades
            if ug is not None and not ug.empty:
                ug = ug.sort_index(ascending=False).head(100)
                for idx, row in ug.iterrows():
                    raw_action = str(row.get("Action","")).lower().strip()
                    from_grade = str(row.get("From Grade","")).strip()
                    to_grade   = str(row.get("To Grade","")).strip()
                    if from_grade.lower() in ("nan","none",""): from_grade = ""
                    if to_grade.lower()   in ("nan","none",""): to_grade   = ""

                    # Infer upgrade/downgrade from grade change
                    if raw_action in ("main","reit") and from_grade and to_grade and from_grade.lower() != to_grade.lower():
                        grade_rank = {
                            "strong buy":5,"outperform":4,"overweight":4,"buy":4,
                            "market perform":3,"neutral":3,"equal-weight":3,"hold":3,"sector perform":3,
                            "underperform":2,"underweight":2,"sell":1,"strong sell":1
                        }
                        f = grade_rank.get(from_grade.lower(), 3)
                        t = grade_rank.get(to_grade.lower(), 3)
                        if t > f:   raw_action = "upgrade"
                        elif t < f: raw_action = "downgrade"

                    label, color, priority = ACTION_MAP.get(raw_action, (raw_action.upper(), "var(--text2)", 2))

                    if from_grade and to_grade and from_grade.lower() != to_grade.lower():
                        rating = f"{from_grade} → {to_grade}"
                    elif to_grade:
                        rating = to_grade
                    elif from_grade:
                        rating = from_grade
                    else:
                        rating = ""

                    # Skip only plain maintain/reiterate with absolutely no rating
                    if priority <= 1 and not rating:
                        continue

                    upgrades.append({
                        "date":   str(idx)[:10],
                        "firm":   str(row.get("Firm","")),
                        "action": label,
                        "color":  color,
                        "rating": rating,
                    })
                    if len(upgrades) >= 20:
                        break
        except Exception as e:
            record_warning(warnings_list, "Analyst recommendations history unavailable", e)

        strong_buy_n = int(info.get("numberOfStrongBuyOpinions") or info.get("strongBuy") or 0)
        buy_n        = int(info.get("numberOfBuyOpinions")       or info.get("buy")       or 0)
        hold_n       = int(info.get("numberOfHoldOpinions")      or info.get("hold")      or 0)
        sell_n       = int(info.get("numberOfSellOpinions")      or info.get("sell")      or 0)
        strong_sell_n= int(info.get("numberOfStrongSellOpinions")or info.get("strongSell")or 0)

        # Fallback: try recommendations_summary if counts are all zero
        if strong_buy_n + buy_n + hold_n + sell_n + strong_sell_n == 0:
            try:
                rs = aksje.recommendations_summary
                if rs is not None and not rs.empty:
                    latest = rs.iloc[0]
                    strong_buy_n = int(latest.get("strongBuy",  0) or 0)
                    buy_n        = int(latest.get("buy",        0) or 0)
                    hold_n       = int(latest.get("hold",       0) or 0)
                    sell_n       = int(latest.get("sell",       0) or 0)
                    strong_sell_n= int(latest.get("strongSell", 0) or 0)
            except Exception as e:
                record_warning(warnings_list, "Analyst recommendation summary unavailable", e)

        return safe_jsonify({
            "currentPrice": round(current_price,2) if current_price else None,
            "targetMean":   round(target_mean,2)   if target_mean   else None,
            "targetHigh":   round(target_high,2)   if target_high   else None,
            "targetLow":    round(target_low,2)    if target_low    else None,
            "currency":     currency,
            "upside":       upside,
            "strongBuy":    strong_buy_n,
            "buy":          buy_n,
            "hold":         hold_n,
            "sell":         sell_n,
            "strongSell":   strong_sell_n,
            "upgrades":     upgrades,
            "warnings":     warnings_list,
        })
    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500

@app.route("/api/insider", methods=["POST"])
def api_insider():
    ticker = get_request_data().get("ticker","").strip().upper()
    if not ticker:
        return safe_jsonify({"error": "No ticker provided"}), 400
    try:
        aksje = yf.Ticker(ticker)
        transactions = []
        warnings_list = []
        try:
            ins = aksje.insider_transactions
            if ins is not None and not ins.empty:
                ins = ins.head(25)
                for _, row in ins.iterrows():
                    shares = row.get("Shares") or row.get("shares")
                    value  = row.get("Value") or row.get("value")
                    shares = safe_int(shares) if shares == shares else None
                    value = safe_int(value) if value == value else None
                    transactions.append({
                        "date":   str(row.get("Start Date", row.get("Date","")))[:10],
                        "name":   str(row.get("Insider",""))[:40],
                        "title":  str(row.get("Position", row.get("Title","")))[:30],
                        "type":   str(row.get("Transaction","")).strip(),
                        "shares": shares,
                        "value":  value,
                    })
        except Exception as e:
            record_warning(warnings_list, "Detailed insider transactions unavailable", e)

        if not transactions:
            try:
                ins2 = aksje.insider_purchases
                if ins2 is not None and not ins2.empty:
                    for _, row in ins2.head(15).iterrows():
                        transactions.append({
                            "date":   str(row.get("Date",""))[:10],
                            "name":   str(row.get("Insider",""))[:40],
                            "title":  str(row.get("Position",""))[:30],
                            "type":   "Purchase",
                            "shares": safe_int(row["Shares"]) if "Shares" in row else None,
                            "value":  safe_int(row["Value"])  if "Value"  in row else None,
                        })
            except Exception as e:
                record_warning(warnings_list, "Fallback insider purchase summary unavailable", e)

        return safe_jsonify({"transactions": transactions, "warnings": warnings_list})
    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500


@app.route("/api/sammenlign", methods=["POST"])
def api_sammenlign():
    data     = get_request_data()
    tickers  = data.get("tickers", [])
    periode  = data.get("periode", "1y")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    resultat = []
    warnings_list = []

    # Download benchmark once outside the loop
    bm_ret = None
    try:
        bm = yf.download(BENCHMARK, period="1y", progress=False, auto_adjust=True)
        if not bm.empty:
            if isinstance(bm.columns, pd.MultiIndex): bm.columns = bm.columns.get_level_values(0)
            bm_ret = bm["Close"].pct_change().dropna()
    except Exception as e:
        record_warning(warnings_list, "Benchmark download unavailable", e)

    for t in tickers:
        try:
            aksje = yf.Ticker(t)
            info  = aksje.info
            # Always use 1y — covers all sub-periods, no extra downloads needed
            df = yf.download(t, period="1y", progress=False, auto_adjust=True)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

            last   = float(df["Close"].iloc[-1])
            ret    = df["Close"].pct_change().dropna()
            avk    = (last / float(df["Close"].iloc[0]) - 1) * 100
            ret_std = ret.std()
            vol    = ret_std * np.sqrt(252) * 100
            sharpe = (ret.mean() - RISIKOFRI_RENTE/252) / ret_std * np.sqrt(252) if ret_std and not np.isnan(ret_std) else float("nan")
            dd     = ((df["Close"] - df["Close"].cummax()) / df["Close"].cummax()).min() * 100

            # Beta using pre-downloaded benchmark
            beta = None
            if bm_ret is not None:
                felles = ret.index.intersection(bm_ret.index)
                if len(felles) > 10:
                    kov  = np.cov(ret.loc[felles], bm_ret.loc[felles])
                    beta = round(kov[0,1]/kov[1,1], 2) if kov[1,1] != 0 else None

            # Period returns from existing data — no extra downloads
            def sub_pct(n):
                try: return round((last / float(df["Close"].iloc[-n]) - 1) * 100, 2)
                except: return None

            avk_data = {"1M": sub_pct(21), "3M": sub_pct(63), "6M": sub_pct(126), "1Y": round(avk, 2)}

            historikk = {
                "datoer": [str(d)[:10] for d in df.index[-60:]],
                "priser": [round(float(p), 2) for p in df["Close"].iloc[-60:]]
            }

            resultat.append({
                "ticker":       t,
                "navn":         info.get("longName", t),
                "sektor":       info.get("sector", "N/A"),
                "pris":         safe(info.get("currentPrice")),
                "mktcap":       stor_tall(info.get("marketCap")),
                "pe":           safe(info.get("trailingPE")),
                "forward_pe":   safe(info.get("forwardPE")),
                "utbytte":      safe(info.get("dividendYield"), pst=True),
                "roe":          safe(info.get("returnOnEquity"), pst=True),
                "bruttomargin": safe(info.get("grossMargins"), pst=True),
                "driftsmargin": safe(info.get("operatingMargins"), pst=True),
                "konsensus":    info.get("recommendationKey","N/A").upper(),
                "mål":          safe(info.get("targetMeanPrice")),
                "avk":          round(avk, 2),
                "avk_perioder": avk_data,
                "vol":          round(vol, 2),
                "sharpe":       round(sharpe, 3) if not np.isnan(sharpe) else None,
                "max_dd":       round(dd, 2),
                "beta":         beta,
                "historikk":    historikk,
            })
        except Exception as e:
            record_warning(warnings_list, f"Comparison data unavailable for {t}", e)

    # AI comparison
    ai_tekst = ""
    if gemini_key and len(resultat) >= 2:
        summary = "\n".join([
            f"{r['ticker']} ({r['navn']}): P/E={r['pe']}, Sharpe={r['sharpe']}, "
            f"Return={r['avk']}%, Volatility={r['vol']}%, MaxDD={r['max_dd']}%, "
            f"Sector={r['sektor']}, Consensus={r['konsensus']}"
            for r in resultat
        ])
        prompt = f"""
You are a senior equity analyst. A client wants to allocate capital and is comparing these stocks. Give a decisive comparative analysis.

STOCKS:
{summary}

Your analysis must cover:
1. **Best Risk-Adjusted Return** — which stock offers the best Sharpe ratio and why it matters
2. **Best Value** — which is most attractively priced on fundamentals (P/E, sector context)
3. **Best Momentum** — which has the strongest recent performance and technical setup
4. **Avoid** — which stock you would not buy right now and the specific reason
5. **Final Ranking** — rank all stocks from most to least attractive right now, with one sentence per stock

Be decisive. A client is making a real allocation decision. Max 300 words.

FINAL INSTRUCTION:
{AI_GUARDRAILS}
- Rank the stocks strictly on the supplied metrics and context above.
- Do not invent business descriptions, catalysts, or company-specific developments that are not in the input.
- If a stock has missing data, say so explicitly and reduce confidence.
- Be willing to say when the ranking is close or the evidence is mixed.
"""
        ai_tekst = spør_ai(prompt, gemini_key, 1200)

    return safe_jsonify({"aksjer": resultat, "ai": ai_tekst, "warnings": warnings_list})

@app.route("/api/portefolje_analyse", methods=["POST"])
def api_portefolje_analyse():
    data       = get_request_data()
    posisjoner = data.get("posisjoner", [])
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    if not posisjoner:
        return safe_jsonify({"error": "No positions provided"}), 400

    resultat      = []
    total_verdi   = 0
    total_kostnad = 0

    for pos in posisjoner:
        try:
            t       = pos["ticker"].upper()
            antall  = float(pos["antall"])
            snitt   = float(pos["snittpris"])
            kostnad = antall * snitt

            aksje = yf.Ticker(t)
            info  = aksje.info
            df    = yf.download(t, period="1y", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

            curr_price = float(df["Close"].iloc[-1]) if not df.empty else snitt
            curr_value = antall * curr_price
            gain       = curr_value - kostnad
            gain_pct   = (gain / kostnad) * 100

            ret    = df["Close"].pct_change().dropna()
            ret_std = ret.std()
            vol    = ret_std*np.sqrt(252)*100
            sharpe = (ret.mean()-RISIKOFRI_RENTE/252)/ret_std*np.sqrt(252) if ret_std and not np.isnan(ret_std) else float("nan")

            total_verdi   += curr_value
            total_kostnad += kostnad

            # Price history (90 days)
            historikk = {
                "datoer": [str(d)[:10] for d in df.index[-90:]],
                "priser": [round(float(p),2) for p in df["Close"].iloc[-90:]]
            }

            resultat.append({
                "ticker":    t,
                "navn":      info.get("longName", t),
                "sektor":    info.get("sector","N/A"),
                "antall":    antall,
                "snittpris": snitt,
                "nåpris":    round(curr_price, 2),
                "kostnad":   round(kostnad, 2),
                "nåverdi":   round(curr_value, 2),
                "gevinst":   round(gain, 2),
                "pst":       round(gain_pct, 2),
                "vol":       round(vol, 2),
                "sharpe":    round(sharpe, 3) if not np.isnan(sharpe) else None,
                "pe":        safe(info.get("trailingPE")),
                "konsensus": info.get("recommendationKey","N/A").upper(),
                "mål":       safe(info.get("targetMeanPrice")),
                "historikk": historikk,
            })
        except Exception as e:
            resultat.append({"ticker": pos.get("ticker","?"), "feil": str(e)})

    total_gain = total_verdi - total_kostnad
    total_pst  = (total_gain/total_kostnad*100) if total_kostnad else 0

    # Sector breakdown
    sektorer = {}
    for r in resultat:
        if "feil" not in r:
            s = r.get("sektor","Unknown")
            sektorer[s] = sektorer.get(s,0) + r["nåverdi"]

    # AI analysis
    ai_tekst = ""
    if gemini_key and resultat:
        pos_info = "\n".join([
            f"{r['ticker']} ({r.get('navn',r['ticker'])}): "
            f"Shares={r.get('antall')}, AvgPrice={r.get('snittpris')}, "
            f"CurrentPrice={r.get('nåpris')}, Gain={r.get('pst','?')}%, "
            f"Sector={r.get('sektor','N/A')}, Sharpe={r.get('sharpe','N/A')}, "
            f"Consensus={r.get('konsensus','N/A')}, P/E={r.get('pe','N/A')}"
            for r in resultat if "feil" not in r
        ])
        prompt = f"""
You are a portfolio manager at a professional investment firm. Analyze this client portfolio and provide actionable recommendations.

PORTFOLIO SUMMARY (total value: ${round(total_verdi,2):,.2f}, total return: {round(total_pst,2)}%):
{pos_info}

SECTOR ALLOCATION: {json.dumps({k: f"{round(v/total_verdi*100,1)}%" for k,v in sektorer.items()})}

Provide a professional portfolio review:
1. **Portfolio Health** — is this portfolio well-constructed? Comment on diversification, concentration risk, and overall quality
2. **Top Performer** — the strongest position and whether to increase exposure
3. **Weakest Link** — the position with the worst risk/reward and a specific recommendation (sell, reduce, or hold with reasoning)
4. **Sector & Concentration Risk** — any dangerous overweights or missing sectors
5. **Gaps** — what asset classes, sectors, or geographies are missing for a balanced portfolio
6. **Position-by-Position** — for each holding: INCREASE / HOLD / REDUCE / EXIT with one-line reasoning
7. **Risk Rating: LOW / MEDIUM / HIGH** — overall portfolio risk with explanation

Be direct and action-oriented. This client needs clear guidance. Max 380 words.

FINAL INSTRUCTION:
{AI_GUARDRAILS}
- Base the review only on the holdings, weights, sectors, returns, and valuation metrics provided above.
- Do not assume tax situation, account type, liquidity needs, or investment horizon beyond what is stated.
- Make the risk discussion concrete: concentration, diversification, factor balance, and weakest links.
- If evidence is insufficient for a strong call on a position, say that explicitly.
"""
        ai_tekst = spør_ai(prompt, gemini_key, 1500)

    return safe_jsonify({
        "posisjoner":     resultat,
        "total_verdi":    round(total_verdi, 2),
        "total_kostnad":  round(total_kostnad, 2),
        "total_gevinst":  round(total_gain, 2),
        "total_pst":      round(total_pst, 2),
        "sektorer":       sektorer,
        "ai":             ai_tekst,
    })

@app.route("/api/nyheter", methods=["POST"])
def api_nyheter():
    data     = get_request_data()
    ticker   = data.get("ticker","").strip().upper()
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not ticker:
        return safe_jsonify({"error": "No ticker provided"}), 400
    try:
        aksje  = yf.Ticker(ticker)
        nyheter = aksje.news or []
        resultat = []
        for n in nyheter[:10]:
            try:
                # yfinance news structure
                content = n.get("content", {})
                tittel  = content.get("title") or n.get("title","No title")
                kilde   = content.get("provider", {}).get("displayName","") or n.get("publisher","")
                lenke   = content.get("canonicalUrl", {}).get("url","") or n.get("link","")
                ts      = content.get("pubDate") or ""
                # Parse dato
                if ts:
                    try:
                        dato = datetime.strptime(ts[:10], "%Y-%m-%d").strftime("%d %b %Y")
                    except:
                        dato = ts[:10]
                else:
                    dato = ""
                if tittel:
                    resultat.append({"tittel": tittel, "kilde": kilde, "lenke": lenke, "dato": dato})
            except:
                continue

        # AI sentiment analysis
        ai_sentiment = ""
        if gemini_key and resultat:
            headlines = "\n".join([f"- {n['tittel']}" for n in resultat])
            prompt = f"""
You are a financial news analyst. Assess the market sentiment for {ticker} based on these recent headlines.

HEADLINES:
{headlines}

Provide:
1. **Sentiment: POSITIVE / NEUTRAL / NEGATIVE** — one word verdict
2. **Key Story** — the single most market-moving headline and why it matters for the stock
3. **Price Impact** — what this news flow suggests for near-term price action (be specific)

Max 120 words. Focus on what actually moves the stock price.

FINAL INSTRUCTION:
{AI_GUARDRAILS}
- Base the conclusion only on the headlines above.
- Do not assume facts, financial impact, or company events beyond what the headlines state.
- If the headlines are mixed or thin, say the signal is mixed or low-confidence.
- Keep the sentiment label explicitly as POSITIVE, NEUTRAL, or NEGATIVE.
"""
            ai_sentiment = spør_ai(prompt, gemini_key, 400)

        return safe_jsonify({"nyheter": resultat, "ai_sentiment": ai_sentiment})
    except Exception as e:
        return safe_jsonify({"error": str(e)}), 500

@app.route("/api/watchlist_kurs", methods=["POST"])
def api_watchlist_kurs():
    data    = get_request_data()
    tickers = data.get("tickers", [])
    if not tickers:
        return safe_jsonify([])

    def fetch_one(t):
        try:
            cached = cache_get(f"wl:{t}")
            if cached: return cached

            aksje = yf.Ticker(t)
            info  = aksje.info
            df = yf.download(t, period="1y", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if df.empty:
                return {"ticker": t, "feil": "No data"}

            last = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last

            def pct(n):
                try:
                    p = float(df["Close"].iloc[-n])
                    return round((last - p) / p * 100, 2)
                except: return None

            try:
                year_start = df[df.index.year == df.index[-1].year].iloc[0]
                ytd = round((last - float(year_start["Close"])) / float(year_start["Close"]) * 100, 2)
            except: ytd = None

            entry = {
                "ticker":    t,
                "navn":      info.get("longName", t),
                "sektor":    info.get("sector", "N/A"),
                "pris":      round(last, 2),
                "endring":   round((last - prev) / prev * 100, 2),
                "uke":       pct(5),
                "maaned":    pct(21),
                "tre_mnd":   pct(63),
                "ytd":       ytd,
                "høy52":     round(float(df["Close"].max()), 2),
                "lav52":     round(float(df["Close"].min()), 2),
                "mktcap":    stor_tall(info.get("marketCap")),
                "pe":        safe(info.get("trailingPE")),
                "konsensus": info.get("recommendationKey", "N/A").upper(),
                "sparkline": [round(float(p), 2) for p in df["Close"].iloc[-30:]],
            }
            cache_set(f"wl:{t}", entry)
            return entry
        except Exception as e:
            return {"ticker": t, "feil": str(e)[:50]}

    # Fetch all tickers in parallel (max 6 concurrent)
    resultat = [None] * len(tickers)
    with ThreadPoolExecutor(max_workers=min(len(tickers), 6)) as ex:
        futures = {ex.submit(fetch_one, t): i for i, t in enumerate(tickers)}
        for future in as_completed(futures):
            resultat[futures[future]] = future.result()

    return safe_jsonify([r for r in resultat if r is not None])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
