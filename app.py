"""
Aksjeanalyse PRO — Flask Web App
"""

import os, io, base64, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from flask import Flask, request, jsonify, render_template
from groq import Groq

warnings.filterwarnings("ignore")

app = Flask(__name__)
RISIKOFRI_RENTE = 0.045
BENCHMARK       = "^GSPC"
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")

MAKRO_TICKERS = {
    "S&P 500":       "^GSPC",
    "Nasdaq":        "^IXIC",
    "VIX":           "^VIX",
    "US 10y Yield":  "^TNX",
    "Gold":          "GC=F",
    "Oil (Brent)":  "BZ=F",
    "EUR/USD":       "EURUSD=X",
    "USD/NOK":       "USDNOK=X",
}

# ── Hjelpefunksjoner ──────────────────────────────────────────────────────────

def safe(v, pst=False, dec=2):
    try:
        if v is None or (isinstance(v, float) and np.isnan(float(v))): return "N/A"
        return f"{float(v)*100:.{dec}f}%" if pst else f"{float(v):.{dec}f}"
    except: return "N/A"

def stor_tall(n):
    try:
        n = float(n)
        for g, e in [(1e12,"T"),(1e9,"B"),(1e6,"M")]:
            if abs(n) >= g: return f"{n/g:.2f}{e}"
        return f"{n:.2f}"
    except: return "N/A"

def spør_groq(prompt: str, api_key: str, maks=1200) -> str:
    key = api_key or GROQ_API_KEY
    if not key: return "No Groq API key provided."
    try:
        klient = Groq(api_key=key)
        svar = klient.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            max_tokens=maks,
        )
        return svar.choices[0].message.content.strip()
    except Exception as e:
        return f"Groq error: {e}"

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
    total  = (df["Close"].iloc[-1]/df["Close"].iloc[0]-1)*100
    dager  = (df.index[-1]-df.index[0]).days
    cagr   = ((df["Close"].iloc[-1]/df["Close"].iloc[0])**(365/max(dager,1))-1)*100
    dagvol = ret.std()
    årvol  = dagvol*np.sqrt(hd)*100
    rf     = RISIKOFRI_RENTE/hd
    sharpe = (ret.mean()-rf)/dagvol*np.sqrt(hd)
    neg    = ret[ret < rf]
    dside  = neg.std()*np.sqrt(hd) if len(neg)>1 else float("nan")
    sortino = (ret.mean()*hd-RISIKOFRI_RENTE)/dside if dside and dside!=0 else float("nan")
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
                sharpe=round(sharpe,3), sortino=round(sortino,3) if not np.isnan(sortino) else None,
                calmar=round(calmar,3) if not np.isnan(calmar) else None,
                max_dd=round(max_dd,2), beta=round(beta,3) if not np.isnan(beta) else None,
                alpha=round(alpha,2), var95=round(var95,2), var99=round(var99,2),
                dd_series=dd.tolist(), dd_dates=[str(d)[:10] for d in dd.index])

# ── Graf → base64 ─────────────────────────────────────────────────────────────

def lag_graf(df: pd.DataFrame, ticker: str) -> str:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 12), facecolor="#0a0f0a")
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.3)
    C   = {"kurs":"#00ff88","sma20":"#ffd166","sma50":"#00b4d8",
           "sma200":"#ef476f","bull":"#00ff88","bear":"#ef476f","text":"#8a9e8a"}

    ax1 = fig.add_subplot(gs[0,:])
    ax1.set_facecolor("#0a0f0a")
    ax1.plot(df.index, df["Close"],  color=C["kurs"],  lw=1.5, label="Kurs")
    ax1.plot(df.index, df["SMA20"],  color=C["sma20"], lw=0.8, ls="--", label="SMA20")
    ax1.plot(df.index, df["SMA50"],  color=C["sma50"], lw=0.8, ls="--", label="SMA50")
    ax1.plot(df.index, df["SMA200"], color=C["sma200"],lw=0.8, ls="--", label="SMA200")
    ax1.fill_between(df.index, df["BB_Up"], df["BB_Lo"],
                     alpha=0.05, color=C["kurs"])
    ax1.set_title(f"{ticker} — Kurs & Indikatorer", color=C["text"], fontsize=10)
    ax1.legend(fontsize=7, labelcolor=C["text"])
    ax1.tick_params(colors=C["text"]); ax1.spines[:].set_color("#1a2e1a")

    ax2 = fig.add_subplot(gs[1,:])
    ax2.set_facecolor("#0a0f0a")
    fc = [C["bull"] if df["Close"].iloc[i]>=df["Close"].iloc[i-1] else C["bear"]
          for i in range(1,len(df))]
    ax2.bar(df.index[1:], df["Volume"].iloc[1:], color=fc, alpha=0.6, width=1)
    ax2.plot(df.index, df["Vol_SMA20"], color=C["sma20"], lw=0.8, ls="--")
    ax2.set_title("Volume", color=C["text"], fontsize=10)
    ax2.tick_params(colors=C["text"]); ax2.spines[:].set_color("#1a2e1a")

    ax3 = fig.add_subplot(gs[2,0])
    ax3.set_facecolor("#0a0f0a")
    ax3.plot(df.index, df["RSI"], color="#f4a261", lw=1.2)
    ax3.axhline(70, color=C["bear"], ls="--", lw=0.7, alpha=0.6)
    ax3.axhline(30, color=C["bull"], ls="--", lw=0.7, alpha=0.6)
    ax3.fill_between(df.index, df["RSI"], 70, where=df["RSI"]>=70, alpha=0.1, color=C["bear"])
    ax3.fill_between(df.index, df["RSI"], 30, where=df["RSI"]<=30, alpha=0.1, color=C["bull"])
    ax3.set_ylim(0,100); ax3.set_title("RSI (14d)", color=C["text"], fontsize=10)
    ax3.tick_params(colors=C["text"]); ax3.spines[:].set_color("#1a2e1a")

    ax4 = fig.add_subplot(gs[2,1])
    ax4.set_facecolor("#0a0f0a")
    ax4.plot(df.index, df["MACD"],   color=C["sma50"], lw=1.2, label="MACD")
    ax4.plot(df.index, df["MACD_S"], color=C["sma20"], lw=1.0, ls="--", label="Signal")
    hc = [C["bull"] if v>=0 else C["bear"] for v in df["MACD_H"]]
    ax4.bar(df.index, df["MACD_H"], color=hc, alpha=0.5, width=1)
    ax4.set_title("MACD", color=C["text"], fontsize=10)
    ax4.legend(fontsize=7, labelcolor=C["text"])
    ax4.tick_params(colors=C["text"]); ax4.spines[:].set_color("#1a2e1a")

    ax5 = fig.add_subplot(gs[3,0])
    ax5.set_facecolor("#0a0f0a")
    rm = df["Close"].cummax()
    dd = (df["Close"]-rm)/rm*100
    ax5.fill_between(df.index, dd, 0, color=C["bear"], alpha=0.5)
    ax5.plot(df.index, dd, color=C["bear"], lw=0.8)
    ax5.set_title("Drawdown (%)", color=C["text"], fontsize=10)
    ax5.tick_params(colors=C["text"]); ax5.spines[:].set_color("#1a2e1a")

    ax6 = fig.add_subplot(gs[3,1])
    ax6.set_facecolor("#0a0f0a")
    dagret = df["Close"].pct_change().dropna()*100
    ax6.hist(dagret, bins=50, color=C["sma50"], alpha=0.6, edgecolor="none")
    ax6.axvline(dagret.mean(), color=C["sma20"], lw=1.2, ls="--")
    ax6.axvline(np.percentile(dagret,5), color=C["bear"], lw=1.0, ls=":")
    ax6.set_title("Return Distribution", color=C["text"], fontsize=10)
    ax6.tick_params(colors=C["text"]); ax6.spines[:].set_color("#1a2e1a")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0a0f0a")
    plt.close()
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

# ── Flask ruter ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    data   = request.json
    ticker = data.get("ticker","").strip().upper()
    periode= data.get("periode","1y")
    groq_key = data.get("groq_key","")

    if not ticker:
        return jsonify({"error": "Ingen ticker angitt"}), 400

    try:
        aksje = yf.Ticker(ticker)
        df    = yf.download(ticker, period=periode, progress=False, auto_adjust=True)
        if df.empty:
            return jsonify({"error": f"No data for {ticker}. Check the ticker symbol."}), 404
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        info = aksje.info
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Benchmark
    bm_df = df.copy()
    for bm in [BENCHMARK, "^GSPC"]:
        try:
            _bm = yf.download(bm, period=periode, progress=False, auto_adjust=True)
            if not _bm.empty:
                if isinstance(_bm.columns, pd.MultiIndex): _bm.columns = _bm.columns.get_level_values(0)
                bm_df = _bm; break
        except: pass

    df  = beregn_tekniske(df)
    sig = hent_signaler(df)
    ri  = beregn_risiko(df, bm_df)
    graf = lag_graf(df, ticker)

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
        "pris":       safe(info.get("currentPrice")),
        "pe":         safe(info.get("trailingPE")),
        "forward_pe": safe(info.get("forwardPE")),
        "pb":         safe(info.get("priceToBook")),
        "ps":         safe(info.get("priceToSalesTrailing12Months")),
        "eps":        safe(info.get("trailingEps")),
        "mktcap":     stor_tall(info.get("marketCap")),
        "omsetning":  stor_tall(info.get("totalRevenue")),
        "ebitda":     stor_tall(info.get("ebitda")),
        "frikontant": stor_tall(info.get("freeCashflow")),
        "gjeld_ek":   safe(info.get("debtToEquity")),
        "utbytte":    safe(info.get("dividendYield"), pst=True),
        "roe":        safe(info.get("returnOnEquity"), pst=True),
        "roa":        safe(info.get("returnOnAssets"), pst=True),
        "bruttomargin":  safe(info.get("grossMargins"), pst=True),
        "driftsmargin":  safe(info.get("operatingMargins"), pst=True),
        "nettmargin":    safe(info.get("profitMargins"), pst=True),
        "52u_høy":    safe(info.get("fiftyTwoWeekHigh")),
        "52u_lav":    safe(info.get("fiftyTwoWeekLow")),
        "mål":        safe(info.get("targetMeanPrice")),
        "konsensus":  info.get("recommendationKey","N/A").upper(),
    }

    # AI analysis
    ai_tekst = ""
    if data.get("ai", False):
        prompt = f"""
You are a stock analyst. Give a concise investment assessment for {fundamental['navn']} ({ticker}).

Key metrics: P/E={fundamental['pe']}, Forward P/E={fundamental['forward_pe']},
Market Cap={fundamental['mktcap']}, ROE={fundamental['roe']}, Operating Margin={fundamental['driftsmargin']},
Consensus={fundamental['konsensus']}

Technical: RSI={sig['rsi']}, MACD={'Bullish' if sig['macd_bull'] else 'Bearish'},
SMA50={'Above' if sig['over_sma50'] else 'Below'}, SMA200={'Above' if sig['over_sma200'] else 'Below'}

Risk: Sharpe={ri.get('sharpe','N/A')}, MaxDD={ri.get('max_dd','N/A')}%, Beta={ri.get('beta','N/A')}

Provide:
1. Overall assessment (2-3 sentences)
2. Key strengths (2-3 bullet points)
3. Key risks (2-3 bullet points)
4. Technical timing
5. Conclusion: BUY / HOLD / SELL

Max 220 words. Be direct and specific.
"""
        ai_tekst = spør_groq(prompt, groq_key)

    return jsonify({
        "ticker":       ticker,
        "fundamental":  fundamental,
        "signaler":     sig,
        "risiko":       ri,
        "graf":         graf,
        "historikk":    historikk,
        "ai":           ai_tekst,
    })

@app.route("/api/makro", methods=["GET"])
def api_makro():
    rader = []
    for navn, sym in MAKRO_TICKERS.items():
        try:
            df = yf.download(sym, period="5d", progress=False, auto_adjust=True)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            siste   = float(df["Close"].iloc[-1])
            forrige = float(df["Close"].iloc[-2]) if len(df)>1 else siste
            endring = (siste-forrige)/forrige*100
            rader.append({"navn":navn,"verdi":round(siste,2),"endring":round(endring,2)})
        except: rader.append({"navn":navn,"verdi":"N/A","endring":0})
    return jsonify(rader)

@app.route("/api/sammenlign", methods=["POST"])
def api_sammenlign():
    data     = request.json
    tickers  = data.get("tickers", [])
    periode  = data.get("periode", "1y")
    groq_key = data.get("groq_key", "")
    resultat = []

    for t in tickers:
        try:
            aksje = yf.Ticker(t)
            info  = aksje.info
            df    = yf.download(t, period=periode, progress=False, auto_adjust=True)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            ret    = df["Close"].pct_change().dropna()
            avk    = (df["Close"].iloc[-1]/df["Close"].iloc[0]-1)*100
            vol    = ret.std()*np.sqrt(252)*100
            sharpe = (ret.mean()-RISIKOFRI_RENTE/252)/ret.std()*np.sqrt(252)
            dd     = ((df["Close"]-df["Close"].cummax())/df["Close"].cummax()).min()*100

            # Beta mot benchmark
            bm = yf.download("^GSPC", period=periode, progress=False, auto_adjust=True)
            if isinstance(bm.columns, pd.MultiIndex): bm.columns = bm.columns.get_level_values(0)
            bret  = bm["Close"].pct_change().dropna()
            felles = ret.index.intersection(bret.index)
            if len(felles) > 10:
                kov  = np.cov(ret.loc[felles], bret.loc[felles])
                beta = round(kov[0,1]/kov[1,1], 2) if kov[1,1] != 0 else None
            else:
                beta = None

            # Avkastning på ulike perioder
            avk_data = {}
            for p_navn, p_kode in [("1M","1mo"),("3M","3mo"),("6M","6mo"),("1Y","1y")]:
                try:
                    _df = yf.download(t, period=p_kode, progress=False, auto_adjust=True)
                    if isinstance(_df.columns, pd.MultiIndex): _df.columns = _df.columns.get_level_values(0)
                    if not _df.empty:
                        avk_data[p_navn] = round((_df["Close"].iloc[-1]/_df["Close"].iloc[0]-1)*100, 2)
                except: avk_data[p_navn] = None

            # Kurshistorikk for graf
            historikk = {
                "datoer": [str(d)[:10] for d in df.index[-60:]],
                "priser": [round(float(p),2) for p in df["Close"].iloc[-60:]]
            }

            resultat.append({
                "ticker":     t,
                "navn":       info.get("longName", t),
                "sektor":     info.get("sector", "N/A"),
                "pris":       safe(info.get("currentPrice")),
                "mktcap":     stor_tall(info.get("marketCap")),
                "pe":         safe(info.get("trailingPE")),
                "forward_pe": safe(info.get("forwardPE")),
                "utbytte":    safe(info.get("dividendYield"), pst=True),
                "roe":        safe(info.get("returnOnEquity"), pst=True),
                "bruttomargin": safe(info.get("grossMargins"), pst=True),
                "driftsmargin": safe(info.get("operatingMargins"), pst=True),
                "konsensus":  info.get("recommendationKey","N/A").upper(),
                "mål":        safe(info.get("targetMeanPrice")),
                "avk":        round(avk, 2),
                "avk_perioder": avk_data,
                "vol":        round(vol, 2),
                "sharpe":     round(sharpe, 3),
                "max_dd":     round(dd, 2),
                "beta":       beta,
                "historikk":  historikk,
            })
        except: pass

    # AI comparison
    ai_tekst = ""
    if groq_key and len(resultat) >= 2:
        summary = "\n".join([
            f"{r['ticker']} ({r['navn']}): P/E={r['pe']}, Sharpe={r['sharpe']}, "
            f"Return={r['avk']}%, Volatility={r['vol']}%, MaxDD={r['max_dd']}%, "
            f"Sector={r['sektor']}, Consensus={r['konsensus']}"
            for r in resultat
        ])
        prompt = f"""
You are a stock analyst. Compare these stocks and give a structured assessment.

STOCKS:
{summary}

Provide:
1. Which stock has the best risk-adjusted return and why?
2. Which is the most attractively valued (P/E, fundamentals)?
3. Which has the best momentum?
4. Which would you avoid and why?
5. Ranking from best to worst investment right now with reasoning

Be specific and direct. Max 280 words.
"""
        ai_tekst = spør_groq(prompt, groq_key, 1200)

    return jsonify({"aksjer": resultat, "ai": ai_tekst})

@app.route("/api/portefolje_analyse", methods=["POST"])
def api_portefolje_analyse():
    data       = request.json
    posisjoner = data.get("posisjoner", [])
    groq_key   = data.get("groq_key", "")

    if not posisjoner:
        return jsonify({"error": "No positions provided"}), 400

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
            vol    = ret.std()*np.sqrt(252)*100
            sharpe = (ret.mean()-RISIKOFRI_RENTE/252)/ret.std()*np.sqrt(252)

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
                "sharpe":    round(sharpe, 3),
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
    if groq_key and resultat:
        pos_info = "\n".join([
            f"{r['ticker']} ({r.get('navn',r['ticker'])}): "
            f"Shares={r.get('antall')}, AvgPrice={r.get('snittpris')}, "
            f"CurrentPrice={r.get('nåpris')}, Gain={r.get('pst','?')}%, "
            f"Sector={r.get('sektor','N/A')}, Sharpe={r.get('sharpe','N/A')}, "
            f"Consensus={r.get('konsensus','N/A')}, P/E={r.get('pe','N/A')}"
            for r in resultat if "feil" not in r
        ])
        prompt = f"""
You are a portfolio manager. Analyze this portfolio and give concrete recommendations.

PORTFOLIO (total value: {round(total_verdi,2)}, total gain: {round(total_pst,2)}%):
{pos_info}

SECTOR BREAKDOWN: {json.dumps({k: round(v/total_verdi*100,1) for k,v in sektorer.items()})}

Provide a thorough assessment:
1. Overall portfolio quality — is it well diversified?
2. Strongest position and why
3. Weakest position — should it be sold or held?
4. Sector concentration — too much in one sector?
5. What is the portfolio missing? (sectors, geography, growth vs. value)
6. Specific recommendations: INCREASE / HOLD / REDUCE / SELL for each position
7. Overall risk rating (low/medium/high) with reasoning

Be direct and action-oriented. Max 350 words.
"""
        ai_tekst = spør_groq(prompt, groq_key, 1500)

    return jsonify({
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
    data     = request.json
    ticker   = data.get("ticker","").strip().upper()
    groq_key = data.get("groq_key","")
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400
    try:
        aksje  = yf.Ticker(ticker)
        nyheter = aksje.news or []
        resultat = []
        for n in nyheter[:10]:
            try:
                # yfinance news structure
                content = n.get("content", {})
                tittel  = content.get("title") or n.get("title","Uten tittel")
                kilde   = content.get("provider", {}).get("displayName","") or n.get("publisher","")
                lenke   = content.get("canonicalUrl", {}).get("url","") or n.get("link","")
                ts      = content.get("pubDate") or ""
                # Parse dato
                if ts:
                    try:
                        from datetime import datetime
                        dato = datetime.strptime(ts[:10], "%Y-%m-%d").strftime("%d.%m.%Y")
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
        if groq_key and resultat:
            headlines = "\n".join([f"- {n['tittel']}" for n in resultat])
            prompt = f"""
Analyze these news headlines for {ticker} and give a brief sentiment assessment.

NEWS:
{headlines}

Provide:
1. Overall sentiment: POSITIVE / NEUTRAL / NEGATIVE
2. What is the most important news and why?
3. What does this mean for the stock price short-term? (1-2 sentences)

Max 100 words. Be direct.
"""
            ai_sentiment = spør_groq(prompt, groq_key, 400)

        return jsonify({"nyheter": resultat, "ai_sentiment": ai_sentiment})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/watchlist_kurs", methods=["POST"])
def api_watchlist_kurs():
    data     = request.json
    tickers  = data.get("tickers", [])
    resultat = []
    for t in tickers:
        try:
            aksje = yf.Ticker(t)
            info  = aksje.info
            df    = yf.download(t, period="5d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if df.empty:
                resultat.append({"ticker": t, "feil": "No data"})
                continue
            last    = float(df["Close"].iloc[-1])
            prev    = float(df["Close"].iloc[-2]) if len(df) > 1 else last
            change  = (last - prev) / prev * 100
            df52 = yf.download(t, period="1y", progress=False, auto_adjust=True)
            if isinstance(df52.columns, pd.MultiIndex): df52.columns = df52.columns.get_level_values(0)
            high52    = float(df52["Close"].max()) if not df52.empty else last
            low52     = float(df52["Close"].min()) if not df52.empty else last
            sparkline = [round(float(p), 2) for p in df52["Close"].iloc[-30:]] if not df52.empty else []
            resultat.append({
                "ticker":    t,
                "navn":      info.get("longName", t),
                "sektor":    info.get("sector", "N/A"),
                "pris":      round(last, 2),
                "endring":   round(change, 2),
                "høy52":     round(high52, 2),
                "lav52":     round(low52, 2),
                "mktcap":    stor_tall(info.get("marketCap")),
                "pe":        safe(info.get("trailingPE")),
                "konsensus": info.get("recommendationKey", "N/A").upper(),
                "sparkline": sparkline,
            })
        except Exception as e:
            resultat.append({"ticker": t, "feil": str(e)[:50]})
    return jsonify(resultat)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
