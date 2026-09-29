# Trade Wind Partners — Analysis Terminal

Professional stock analysis platform built with Flask, deployed on Render.

🔗 **Live:** [stockmaxing.onrender.com](https://stockmaxing.onrender.com)

---

## Features

### Stock Analysis
- **Technical** — RSI, MACD, Bollinger Bands, SMA 20/50/200, ATR, volume analysis
- **Fundamental** — P/E, Forward P/E, EV/EBITDA, margins, ROE, ROA, debt/equity, free cash flow
- **Risk** — Beta, Sharpe, Sortino, Calmar, VaR (95%/99%), max drawdown, CAGR
- **Charts** — Interactive TradingView chart with Volume by Price + static matplotlib chart
- **News** — Latest headlines with AI sentiment analysis
- **Earnings** — Next earnings date + EPS beat/miss history
- **Analysts** — Wall Street consensus, price targets (low/mean/high), upgrades & downgrades
- **Insider** — SEC Form 4 filings — executives buying and selling
- **Short Interest** — Short % of float, days to cover, month-over-month change
- **Options Flow** — Put/call ratio, ATM implied volatility, top strikes by open interest
- **AI Analysis** — LLaMA 3.3 70B: assessment, strengths, risks, smart money signals, verdict

### Other Views
- **Watchlist** — Live prices, sparklines, 1D/1W/1M/3M/YTD returns
- **Portfolio** — Live P&L, sector breakdown, AI portfolio review with position-by-position recommendations
- **Compare** — Side-by-side multi-stock comparison with AI ranking
- **Sectors** — S&P 500 sector heatmap with 1D/1W/1M/3M/YTD toggle, top 10 holdings per sector

### UI / UX
- Trade Wind Partners branding — logo, favicon, navy blue palette, splash screen
- Bloomberg-style dark terminal design (IBM Plex Mono)
- Light/dark mode toggle
- Mobile-friendly with hamburger menu
- PDF export of analysis
- Smooth tab fade animations, ticker flash on update, staggered sector bar animations

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask, Gunicorn (3 workers) |
| Data | yfinance (Yahoo Finance) |
| AI | Gemini/GROQ API — LLaMA 3.3 70B Versatile |
| Charts | TradingView Widget, Matplotlib, Seaborn |
| Frontend | Vanilla JS, IBM Plex Mono, CSS Grid |
| Hosting | Render Starter |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/analyse` | POST | Market data, technicals, fundamentals, risk (~2s) |
| `/api/ai_analyse` | POST | AI analysis — runs async after main analyse |
| `/api/makro` | GET | Live macro indicators (S&P, Nasdaq, VIX, yields, FX) |
| `/api/earnings` | POST | Next earnings date + EPS history |
| `/api/analyst` | POST | Analyst ratings, price targets, upgrades/downgrades |
| `/api/insider` | POST | Insider transactions (SEC Form 4) |
| `/api/short_interest` | POST | Short % of float, days to cover |
| `/api/options_flow` | POST | Put/call ratio, ATM IV, top strikes |
| `/api/sammenlign` | POST | Multi-stock comparison with AI ranking |
| `/api/portefolje_analyse` | POST | Portfolio analysis with AI |
| `/api/nyheter` | POST | News with AI sentiment |
| `/api/watchlist_kurs` | POST | Prices + 1D/1W/1M/3M/YTD returns + sparklines |

---

## Setup & Deployment

### Local development
```bash
pip install -r requirements.txt
python app.py
```

### Deploy to Render
1. Fork this repo
2. Create a new **Web Service** on [render.com](https://render.com)
3. Connect your GitHub repo
4. Add environment variable:
   ```
   GROQ_API_KEY = your_key_here
   ```
5. Render auto-deploys on every push to `main`

Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys)

---

## Project Structure

```
STOCKMAXING/
├── app.py               # Flask backend + all API endpoints
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment config (3 workers)
└── templates/
    └── index.html       # Full frontend (HTML + CSS + JS)
```

---

## Performance Notes

- Market data and AI analysis are **decoupled** — results appear in ~2s, AI fills in async
- `watchlist_kurs` uses a single 1y download per ticker to compute all period returns
- Benchmark (S&P 500) is downloaded **once** per compare request, not per ticker
- Gunicorn runs **3 workers** so macro fetches are never blocked by slow analysis requests

---

## Notes

- Norwegian stocks use `.OL` suffix (e.g. `EQNR.OL`, `DNB.OL`)
- Data from Yahoo Finance — availability and accuracy varies by ticker
- All user data (watchlist, portfolio) stored in browser `localStorage`
- **Not financial advice**

---

*Built for Trade Wind Partners*
