import io
import base64
import time as _time
import os
import threading
import requests
import cloudscraper
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Search autocomplete via Yahoo Finance ─────────────────────────────────────
@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    if len(q) < 1:
        return jsonify([])
    try:
        url = (
            f"https://query1.finance.yahoo.com/v1/finance/search"
            f"?q={q}&quotesCount=10&newsCount=0&enableFuzzyQuery=false"
        )
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        data = r.json()
        results = []
        for item in data.get("quotes", []):
            results.append({
                "ticker":   item.get("symbol", ""),
                "name":     item.get("longname") or item.get("shortname", ""),
                "type":     item.get("quoteType", ""),
                "exchange": item.get("exchDisp", ""),
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Chart generation ──────────────────────────────────────────────────────────
@app.route("/analyze")
def analyze():
    ticker_sym = request.args.get("ticker", "").upper().strip()
    if not ticker_sym:
        return jsonify({"error": "Ticker manquant"}), 400

    start_param = request.args.get("start", "").strip()
    end_param   = request.args.get("end",   "").strip()

    try:
        df = yf.download(ticker_sym, period="max", auto_adjust=True, progress=False)
        if df.empty:
            return jsonify({"error": f"Aucune donnée pour {ticker_sym}"}), 404

        df = df["Close"].dropna()
        df.index = pd.to_datetime(df.index)

        # Plage complète avant filtrage
        full_start_iso = df.index[0].strftime("%Y-%m-%d")
        full_end_iso   = df.index[-1].strftime("%Y-%m-%d")

        # Filtrage par date si fourni
        if start_param:
            df = df[df.index >= pd.to_datetime(start_param)]
        if end_param:
            df = df[df.index <= pd.to_datetime(end_param)]

        if len(df) < 30:
            return jsonify({"error": "Pas assez de données sur cette période (minimum 30 jours)."}), 400

        prices = df.values.flatten()
        dates  = df.index

        # Info société
        info = {}
        try:
            t    = yf.Ticker(ticker_sym)
            info = t.info or {}
        except Exception:
            pass

        company_name = info.get("longName") or info.get("shortName") or ticker_sym
        currency     = info.get("currency", "$")
        sector       = info.get("sector", "")
        market_cap   = info.get("marketCap")

        # Régression log-linéaire
        x = np.array([(d - dates[0]).days for d in dates], dtype=float)
        y = np.log(prices)

        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        y_fit     = slope * x + intercept
        residuals = y - y_fit
        sigma     = residuals.std()

        annual_return = (np.exp(slope * 365) - 1) * 100
        last_price    = float(prices[-1])
        sigma_pct     = (np.exp(sigma) - 1) * 100
        sigma_abs     = last_price * (np.exp(sigma) - 1)

        last_x       = x[-1]
        last_reg_log = slope * last_x + intercept
        last_sigma_n = (np.log(last_price) - last_reg_log) / sigma

        # Niveau de valorisation
        if last_sigma_n >= 2:
            valuation_label = "⚠️ Très surévalué"
            valuation_color = "#ef5350"
        elif last_sigma_n >= 1:
            valuation_label = "📈 Surévalué"
            valuation_color = "#ffa726"
        elif last_sigma_n <= -2:
            valuation_label = "🟢 Très sous-évalué"
            valuation_color = "#66bb6a"
        elif last_sigma_n <= -1:
            valuation_label = "💙 Sous-évalué"
            valuation_color = "#4fc3f7"
        else:
            valuation_label = "⚖️ Juste valeur"
            valuation_color = "#ffffff"

        # ── Graphique ──────────────────────────────────────────────────────────
        fig, (ax_main, ax_info) = plt.subplots(
            2, 1, figsize=(13, 8),
            gridspec_kw={"height_ratios": [6, 1]},
            facecolor="#0d1117"
        )
        fig.suptitle(
            f"{company_name} ({ticker_sym}) — Régression log-linéaire",
            color="white", fontsize=13, fontweight="bold", y=0.99
        )

        ax_main.set_facecolor("#0d1117")

        # Prix
        ax_main.semilogy(dates, prices, color="#4fc3f7", linewidth=1.1,
                         label=f"{ticker_sym} (cours ajusté)", zorder=3)
        # Droite
        ax_main.semilogy(dates, np.exp(y_fit), color="#ffffff", linewidth=1.4,
                         linestyle="--", label="Régression", zorder=4)

        # Bandes colorées
        kw = dict(zorder=1)
        ax_main.fill_between(dates,
            np.exp(y_fit - sigma), np.exp(y_fit + sigma),
            color="#66bb6a", alpha=0.18, label="± 1σ", **kw)
        ax_main.fill_between(dates,
            np.exp(y_fit - 2*sigma), np.exp(y_fit - sigma),
            color="#ffa726", alpha=0.12, **kw)
        ax_main.fill_between(dates,
            np.exp(y_fit + sigma), np.exp(y_fit + 2*sigma),
            color="#ffa726", alpha=0.12, label="± 2σ", **kw)
        ax_main.fill_between(dates,
            np.exp(y_fit - 3*sigma), np.exp(y_fit - 2*sigma),
            color="#ef5350", alpha=0.09, **kw)
        ax_main.fill_between(dates,
            np.exp(y_fit + 2*sigma), np.exp(y_fit + 3*sigma),
            color="#ef5350", alpha=0.09, label="± 3σ", **kw)

        # Lignes de niveau
        for k, col, ls in [(1,"#66bb6a","-"),(2,"#ffa726","--"),
                           (-1,"#66bb6a","-"),(-2,"#ffa726","--")]:
            ax_main.semilogy(dates, np.exp(y_fit + k*sigma),
                             color=col, linewidth=0.55, linestyle=ls, alpha=0.65, zorder=2)

        ax_main.scatter([dates[-1]], [last_price], color="#4fc3f7", s=55, zorder=5,
                        label=f"Dernier : {last_price:.2f} {currency}")

        # Annotation position
        ax_main.annotate(
            f"{last_sigma_n:+.2f}σ",
            xy=(dates[-1], last_price),
            xytext=(-55, 12), textcoords="offset points",
            color=valuation_color, fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=valuation_color, lw=0.8)
        )

        ax_main.set_ylabel(f"Prix ({currency}) — échelle log", color="white", fontsize=10)
        ax_main.tick_params(colors="white")
        ax_main.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v:.0f}")
        )
        for spine in ax_main.spines.values():
            spine.set_edgecolor("#2a2a3a")
        ax_main.grid(True, color="#1a1f2b", linewidth=0.5)
        ax_main.legend(loc="upper left", facecolor="#12131f",
                       edgecolor="#2a2a3a", labelcolor="white", fontsize=8.5)
        ax_main.text(0.01, 0.97, f"R² = {r_value**2:.4f}",
                     transform=ax_main.transAxes, color="#888888", fontsize=8, va="top")

        # Panneau bas
        ax_info.set_facecolor("#0a0b10")
        ax_info.axis("off")
        mc_str = ""
        if market_cap:
            if market_cap >= 1e12:
                mc_str = f"  |  Mkt Cap : {market_cap/1e12:.2f}T {currency}"
            elif market_cap >= 1e9:
                mc_str = f"  |  Mkt Cap : {market_cap/1e9:.2f}B {currency}"
        info_text = (
            f"  Pente annuelle : {annual_return:.2f}%/an"
            f"    |    1σ = ±{sigma_pct:.2f}% (±{sigma_abs:.2f} {currency})"
            f"    |    Position : {last_sigma_n:+.2f}σ ({valuation_label})"
            f"{mc_str}"
            f"    |    {sector}    |    Source : Yahoo Finance"
            f"    |    {dates[0].strftime('%d/%m/%Y')} → {dates[-1].strftime('%d/%m/%Y')}"
        )
        ax_info.text(0.0, 0.5, info_text, transform=ax_info.transAxes,
                     color="#aaaaaa", fontsize=8, va="center", fontfamily="monospace")

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="#0d1117")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return jsonify({
            "ticker":          ticker_sym,
            "name":            company_name,
            "sector":          sector,
            "currency":        currency,
            "last_price":      round(last_price, 2),
            "annual_return":   round(annual_return, 2),
            "sigma_pct":       round(sigma_pct, 2),
            "sigma_abs":       round(sigma_abs, 2),
            "r_squared":       round(r_value**2, 4),
            "current_sigma":   round(float(last_sigma_n), 2),
            "valuation":       valuation_label,
            "valuation_color": valuation_color,
            "start_date":      dates[0].strftime("%d/%m/%Y"),
            "end_date":        dates[-1].strftime("%d/%m/%Y"),
            "start_iso":       dates[0].strftime("%Y-%m-%d"),
            "end_iso":         dates[-1].strftime("%Y-%m-%d"),
            "full_start_iso":  full_start_iso,
            "full_end_iso":    full_end_iso,
            "data_points":     len(dates),
            "chart":           img_b64,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Momentum scanner ─────────────────────────────────────────────────────────

_mom_cache = {}
_qs_lock   = threading.Lock()   # sérialise les downloads /quick-stats

# ── Supabase / PostgreSQL cache persistant ────────────────────────────────────
_DATABASE_URL = os.environ.get("DATABASE_URL")

def _db_get(table, key_col, key_val, ttl_seconds):
    """Lit le cache DB. Retourne le dict data ou None si absent/expiré."""
    if not _DATABASE_URL:
        return None
    try:
        conn = psycopg2.connect(_DATABASE_URL, connect_timeout=5)
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(f"SELECT data, computed_at FROM {table} WHERE {key_col} = %s", (key_val,))
        row = cur.fetchone()
        conn.close()
        if row:
            age = _time.time() - row["computed_at"].timestamp()
            if age < ttl_seconds:
                return dict(row["data"])
    except Exception as e:
        print(f"[DB GET] {table}/{key_val}: {e}")
    return None

def _db_set(table, key_col, key_val, data):
    """Écrit dans le cache DB (upsert)."""
    if not _DATABASE_URL:
        return
    try:
        conn = psycopg2.connect(_DATABASE_URL, connect_timeout=5)
        cur  = conn.cursor()
        cur.execute(
            f"""INSERT INTO {table} ({key_col}, data, computed_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT ({key_col}) DO UPDATE
                SET data = EXCLUDED.data, computed_at = NOW()""",
            (key_val, psycopg2.extras.Json(data))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB SET] {table}/{key_val}: {e}")
_rank_hist_cache = {}
CACHE_TTL       = 4  * 3600   # 4h  — petits indices
CACHE_TTL_LARGE = 24 * 3600   # 24h — Russell 2000 / STOXX 600
_LARGE_INDICES  = {'russell2000', 'stoxx600'}

# Index components: (yahoo_ticker, display_name, sector)
_INDICES = {
    'dji': {
        'label': 'Dow Jones 30',
        'wiki':  None,
        'components': [
            ('AAPL','Apple','Technology'),('AMGN','Amgen','Healthcare'),
            ('AXP','American Express','Financials'),('BA','Boeing','Industrials'),
            ('CAT','Caterpillar','Industrials'),('CRM','Salesforce','Technology'),
            ('CSCO','Cisco','Technology'),('CVX','Chevron','Energy'),
            ('DIS','Disney','Communication'),('GS','Goldman Sachs','Financials'),
            ('HD','Home Depot','Consumer Disc.'),('HON','Honeywell','Industrials'),
            ('IBM','IBM','Technology'),('JNJ','Johnson & Johnson','Healthcare'),
            ('JPM','JPMorgan Chase','Financials'),('KO','Coca-Cola','Consumer Staples'),
            ('MCD',"McDonald's",'Consumer Disc.'),('MMM','3M','Industrials'),
            ('MRK','Merck','Healthcare'),('MSFT','Microsoft','Technology'),
            ('NKE','Nike','Consumer Disc.'),('PG','Procter & Gamble','Consumer Staples'),
            ('TRV','Travelers','Financials'),('UNH','UnitedHealth','Healthcare'),
            ('V','Visa','Financials'),('VZ','Verizon','Communication'),
            ('WMT','Walmart','Consumer Staples'),('DOW','Dow Inc.','Materials'),
            ('AMZN','Amazon','Consumer Disc.'),('SHW','Sherwin-Williams','Materials'),
        ],
    },
    'cac40': {
        'label': 'CAC 40',
        'wiki':  None,
        'components': [
            ('AC.PA','Accor','Consumer Disc.'),('AI.PA','Air Liquide','Materials'),
            ('AIR.PA','Airbus','Industrials'),('ALO.PA','Alstom','Industrials'),
            ('MT.AS','ArcelorMittal','Materials'),('CS.PA','AXA','Financials'),
            ('BNP.PA','BNP Paribas','Financials'),('EN.PA','Bouygues','Industrials'),
            ('CAP.PA','Capgemini','Technology'),('CA.PA','Carrefour','Consumer Staples'),
            ('ACA.PA','Crédit Agricole','Financials'),('BN.PA','Danone','Consumer Staples'),
            ('DSY.PA','Dassault Systèmes','Technology'),('ENGI.PA','Engie','Utilities'),
            ('EL.PA','EssilorLuxottica','Healthcare'),('RMS.PA','Hermès','Consumer Disc.'),
            ('KER.PA','Kering','Consumer Disc.'),('OR.PA',"L'Oréal",'Consumer Staples'),
            ('LR.PA','Legrand','Industrials'),('MC.PA','LVMH','Consumer Disc.'),
            ('ORA.PA','Orange','Communication'),('RI.PA','Pernod Ricard','Consumer Staples'),
            ('PUB.PA','Publicis','Communication'),('RNO.PA','Renault','Consumer Disc.'),
            ('SGO.PA','Saint-Gobain','Materials'),('SAN.PA','Sanofi','Healthcare'),
            ('SAF.PA','Safran','Industrials'),('SU.PA','Schneider Electric','Industrials'),
            ('GLE.PA','Société Générale','Financials'),('STM.PA','STMicroelectronics','Technology'),
            ('TTE.PA','TotalEnergies','Energy'),('HO.PA','Thales','Industrials'),
            ('URW.AS','Unibail-Rodamco','Real Estate'),('VIE.PA','Veolia','Utilities'),
            ('DG.PA','Vinci','Industrials'),('VIV.PA','Vivendi','Communication'),
            ('ML.PA','Michelin','Consumer Disc.'),('TEP.PA','Teleperformance','Industrials'),
            ('ERF.PA','Eurofins Scientific','Healthcare'),('SW.PA','Sodexo','Consumer Disc.'),
        ],
    },
    'dax40': {
        'label': 'DAX 40',
        'wiki':  None,
        'components': [
            ('ADS.DE','Adidas','Consumer Disc.'),('AIR.DE','Airbus','Industrials'),
            ('ALV.DE','Allianz','Financials'),('BAS.DE','BASF','Materials'),
            ('BAYN.DE','Bayer','Healthcare'),('BMW.DE','BMW','Consumer Disc.'),
            ('BEI.DE','Beiersdorf','Consumer Staples'),('BNR.DE','Brenntag','Materials'),
            ('CON.DE','Continental','Consumer Disc.'),('1COV.DE','Covestro','Materials'),
            ('DTG.DE','Daimler Truck','Industrials'),('DBK.DE','Deutsche Bank','Financials'),
            ('DB1.DE','Deutsche Börse','Financials'),('DHL.DE','DHL Group','Industrials'),
            ('DTE.DE','Deutsche Telekom','Communication'),('EOAN.DE','E.ON','Utilities'),
            ('FRE.DE','Fresenius','Healthcare'),('FME.DE','Fresenius Medical','Healthcare'),
            ('G1A.DE','GEA Group','Industrials'),('HEI.DE','HeidelbergMaterials','Materials'),
            ('HEN3.DE','Henkel','Consumer Staples'),('IFX.DE','Infineon','Technology'),
            ('MBG.DE','Mercedes-Benz','Consumer Disc.'),('MRK.DE','Merck KGaA','Healthcare'),
            ('MTX.DE','MTU Aero','Industrials'),('MUV2.DE','Munich Re','Financials'),
            ('PAH3.DE','Porsche Holding','Consumer Disc.'),('P911.DE','Porsche AG','Consumer Disc.'),
            ('RHM.DE','Rheinmetall','Industrials'),('RWE.DE','RWE','Utilities'),
            ('SAP.DE','SAP','Technology'),('SRT3.DE','Sartorius','Healthcare'),
            ('SIE.DE','Siemens','Industrials'),('ENR.DE','Siemens Energy','Energy'),
            ('SHL.DE','Siemens Healthineers','Healthcare'),('SY1.DE','Symrise','Materials'),
            ('VOW3.DE','Volkswagen','Consumer Disc.'),('VNA.DE','Vonovia','Real Estate'),
            ('ZAL.DE','Zalando','Consumer Disc.'),('QIA.DE','Qiagen','Healthcare'),
        ],
    },
    'stoxx50': {
        'label': 'Euro Stoxx 50',
        'wiki':  None,
        'components': [
            ('ASML.AS','ASML','Technology'),('MC.PA','LVMH','Consumer Disc.'),
            ('SAP.DE','SAP','Technology'),('SIE.DE','Siemens','Industrials'),
            ('TTE.PA','TotalEnergies','Energy'),('SAN.PA','Sanofi','Healthcare'),
            ('AIR.PA','Airbus','Industrials'),('ALV.DE','Allianz','Financials'),
            ('OR.PA',"L'Oréal",'Consumer Staples'),('INGA.AS','ING Group','Financials'),
            ('BNP.PA','BNP Paribas','Financials'),('AI.PA','Air Liquide','Materials'),
            ('RMS.PA','Hermès','Consumer Disc.'),('MUV2.DE','Munich Re','Financials'),
            ('ABI.BR','AB InBev','Consumer Staples'),('ENEL.MI','Enel','Utilities'),
            ('IBE.MC','Iberdrola','Utilities'),('DTE.DE','Deutsche Telekom','Communication'),
            ('BMW.DE','BMW','Consumer Disc.'),('MBG.DE','Mercedes-Benz','Consumer Disc.'),
            ('AD.AS','Ahold Delhaize','Consumer Staples'),('SU.PA','Schneider Electric','Industrials'),
            ('BAS.DE','BASF','Materials'),('DHL.DE','DHL Group','Industrials'),
            ('BAYN.DE','Bayer','Healthcare'),('ENI.MI','ENI','Energy'),
            ('VOW3.DE','Volkswagen','Consumer Disc.'),('IFX.DE','Infineon','Technology'),
            ('PHIA.AS','Philips','Healthcare'),('ISP.MI','Intesa Sanpaolo','Financials'),
            ('DBK.DE','Deutsche Bank','Financials'),('KER.PA','Kering','Consumer Disc.'),
            ('SAN.MC','Banco Santander','Financials'),('SAF.PA','Safran','Industrials'),
            ('SGO.PA','Saint-Gobain','Materials'),('RWE.DE','RWE','Utilities'),
            ('ITX.MC','Inditex','Consumer Disc.'),('EL.PA','EssilorLuxottica','Healthcare'),
            ('DG.PA','Vinci','Industrials'),('ADYEN.AS','Adyen','Technology'),
            ('CS.PA','AXA','Financials'),('UMG.AS','Universal Music','Communication'),
            ('URW.AS','Unibail-Rodamco','Real Estate'),('VNA.DE','Vonovia','Real Estate'),
            ('PRX.AS','Prosus','Technology'),('NOKIA.HE','Nokia','Technology'),
            ('CRH.L','CRH','Materials'),('VIV.PA','Vivendi','Communication'),
            ('MTX.DE','MTU Aero','Industrials'),('MRK.DE','Merck KGaA','Healthcare'),
        ],
    },
    'sectors_us': {
        'label': 'Secteurs US (SPDR)',
        'wiki':  None,
        'components': [
            ('XLC',  'Communication Services Select Sector', 'Communication Services'),
            ('XLY',  'Consumer Discretionary Select Sector', 'Consumer Discretionary'),
            ('XLP',  'Consumer Staples Select Sector',       'Consumer Staples'),
            ('XLE',  'Energy Select Sector',                 'Energy'),
            ('XLF',  'Financials Select Sector',             'Financials'),
            ('XLV',  'Health Care Select Sector',            'Health Care'),
            ('XLI',  'Industrials Select Sector',            'Industrials'),
            ('XLK',  'Information Technology Select Sector', 'Information Technology'),
            ('XLB',  'Materials Select Sector',              'Materials'),
            ('XLRE', 'Real Estate Select Sector',            'Real Estate'),
            ('XLU',  'Utilities Select Sector',              'Utilities'),
        ],
    },
}

_WIKI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def _wiki_tables(url):
    html = requests.get(url, headers=_WIKI_HEADERS, timeout=15).text
    return pd.read_html(io.StringIO(html))

def _wiki_sp500():
    tables = _wiki_tables('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = tables[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    return list(zip(df['Symbol'], df['Security'], df['GICS Sector']))

def _wiki_nasdaq100():
    tables = _wiki_tables('https://en.wikipedia.org/wiki/Nasdaq-100')
    for t in tables:
        cols_lower = {str(c).lower(): c for c in t.columns}
        if 'ticker' in cols_lower or 'symbol' in cols_lower:
            tc = cols_lower.get('ticker') or cols_lower.get('symbol')
            nc = next((cols_lower[k] for k in cols_lower if 'company' in k or 'name' in k), None)
            sc = next((cols_lower[k] for k in cols_lower if 'sector' in k or 'industry' in k), None)
            tickers = t[tc].dropna().astype(str).tolist()
            names   = t[nc].tolist() if nc else ['']*len(tickers)
            sectors = t[sc].tolist() if sc else ['']*len(tickers)
            return list(zip(tickers, names, sectors))
    return None

# ── Country → Yahoo Finance suffix (STOXX 600 Wikipedia) ─────────────────────
_COUNTRY_SUFFIX = {
    'Switzerland':      '.SW',
    'United Kingdom':   '.L',
    'Germany':          '.DE',
    'France':           '.PA',
    'Netherlands':      '.AS',
    'Italy':            '.MI',
    'Spain':            '.MC',
    'Sweden':           '.ST',
    'Denmark':          '.CO',
    'Finland':          '.HE',
    'Norway':           '.OL',
    'Belgium':          '.BR',
    'Portugal':         '.LS',
    'Ireland':          '.IR',
    'Austria':          '.VI',
    'Luxembourg':       '.LU',
    'Poland':           '.WA',
}


def _iwm_russell2000():
    """Russell 2000 via Vanguard VTWO ETF API (iShares bloqué en datacenter)."""
    base = "https://investor.vanguard.com/investment-products/etfs/profile/api/vtwo/portfolio-holding/stock"
    hdrs = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept": "application/json",
    }
    result = []
    page = 1
    while True:
        r = requests.get(base, headers=hdrs, params={"sortBy": "weighting", "sortOrder": "desc", "perPage": 500, "page": page}, timeout=30)
        print(f"[VANGUARD] page={page} status={r.status_code} content_type={r.headers.get('Content-Type','?')} body_start={r.text[:80]!r}")
        r.raise_for_status()
        data = r.json()
        entities = data.get("fund", {}).get("entity", [])
        print(f"[VANGUARD] page={page} entities={len(entities)}")
        if not entities:
            break
        for e in entities:
            t = str(e.get("ticker", "")).strip()
            if not t or t in ("-", "nan", ""):
                continue
            result.append((t, str(e.get("longName", t)).strip(), ""))
        if len(entities) < 500:
            break
        page += 1
    if not result:
        raise ValueError("Aucun composant récupéré depuis Vanguard VTWO")
    return result

def _wiki_stoxx600():
    """STOXX 600 via Wikipedia — 534 composants avec mapping pays→suffixe Yahoo."""
    tables = _wiki_tables('https://en.wikipedia.org/wiki/STOXX_Europe_600')
    for t in tables:
        if 'Ticker' in t.columns and 'Country' in t.columns and len(t) > 100:
            result = []
            for _, row in t.iterrows():
                ticker = str(row['Ticker']).strip()
                if not ticker or ticker == 'nan':
                    continue
                country = str(row.get('Country', '')).strip()
                sector  = str(row.get('ICB Sector', '')).strip()
                name    = str(row.get('Company', ticker)).strip()
                suffix  = _COUNTRY_SUFFIX.get(country, '')
                result.append((ticker + suffix, name, sector))
            return result
    raise ValueError("Table STOXX 600 introuvable sur Wikipedia")

# ── RSI (Wilder) ──────────────────────────────────────────────────────────────
def _rsi(prices, period=21):
    delta  = prices.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_l  = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs     = avg_g / avg_l
    rsi    = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else None

# ── Market cap (parallel) ─────────────────────────────────────────────────────
def _fetch_market_caps(tickers, max_workers=50):
    caps = {}
    def _fetch(t):
        try:
            return t, yf.Ticker(t).fast_info.market_cap
        except Exception:
            return t, None
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch, t): t for t in tickers}
        for f in as_completed(futures):
            t, cap = f.result()
            caps[t] = cap
    return caps

def _fmt_cap(v):
    if not v:
        return None
    if v >= 1e12: return f"{v/1e12:.2f}T"
    if v >= 1e9:  return f"{v/1e9:.1f}B"
    if v >= 1e6:  return f"{v/1e6:.0f}M"
    return str(int(v))

def _get_components(index_name):
    if index_name == 'sp500':       return _wiki_sp500()
    if index_name == 'nasdaq100':   return _wiki_nasdaq100()
    if index_name == 'russell2000': return _iwm_russell2000()
    if index_name == 'stoxx600':    return _wiki_stoxx600()
    if index_name in _INDICES:      return _INDICES[index_name]['components']
    return None

def _index_label(index_name):
    labels = {
        'sp500': 'S&P 500', 'nasdaq100': 'Nasdaq 100',
        'russell2000': 'Russell 2000', 'stoxx600': 'STOXX Europe 600',
        **{k: v['label'] for k, v in _INDICES.items()}
    }
    return labels.get(index_name, index_name.upper())


@app.route("/momentum")
def momentum():
    index_name = request.args.get("index", "sp500").lower()
    force      = request.args.get("force", "0") == "1"

    # Cache (24h pour les grands indices, 4h pour les autres)
    ttl = CACHE_TTL_LARGE if index_name in _LARGE_INDICES else CACHE_TTL
    now = _time.time()

    # L1 : cache mémoire
    if not force and index_name in _mom_cache:
        ts, data = _mom_cache[index_name]
        if now - ts < ttl:
            return jsonify(data)

    # L2 : cache Supabase (survit aux redémarrages)
    if not force:
        db_data = _db_get("momentum_cache", "index_name", index_name, ttl)
        if db_data:
            _mom_cache[index_name] = (_time.time(), db_data)
            return jsonify(db_data)

    try:
        components = _get_components(index_name)
    except Exception as e:
        return jsonify({"error": f"Impossible de récupérer les composants : {e}"}), 500

    if not components:
        return jsonify({"error": f"Indice '{index_name}' non reconnu."}), 400

    tickers_list = [c[0] for c in components]
    name_map     = {c[0]: c[1] for c in components}
    sector_map   = {c[0]: c[2] for c in components}

    try:
        # ── Download par batchs parallèles (évite timeout gunicorn) ────────
        BATCH_SIZE = 50   # 50 tickers/batch = rapide, peu de risque rate-limit
        MAX_WORKERS = 5   # 5 downloads simultanés

        def _download_chunk(chunk):
            try:
                raw = yf.download(chunk, period="2y", auto_adjust=True, progress=False)
                if raw.empty:
                    print(f"[BATCH] vide: {chunk[:2]}")
                    return None
                close = raw["Close"] if len(chunk) > 1 else raw[["Close"]].rename(columns={"Close": chunk[0]})
                return close if not close.empty else None
            except Exception as e:
                print(f"[BATCH] erreur: {e}")
                return None

        chunks = [tickers_list[i:i+BATCH_SIZE] for i in range(0, len(tickers_list), BATCH_SIZE)]
        frames = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(_download_chunk, chunk): chunk for chunk in chunks}
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    frames.append(result)

        print(f"[MOMENTUM] {index_name}: {len(frames)}/{len(chunks)} chunks OK")
        close_df = pd.concat(frames, axis=1) if frames else pd.DataFrame()

        # ── Market cap (parallel fetch) ──────────────────────────────────
        valid_tickers = [t for t in tickers_list if t in close_df.columns]
        # Pour les grands indices (>600 tickers), skip market cap — trop lent en prod
        if len(valid_tickers) > 600:
            mktcap_map = {}
        else:
            mktcap_map = _fetch_market_caps(valid_tickers)

        # ── Per-ticker calculations ───────────────────────────────────────
        results = []
        for t in tickers_list:
            try:
                if t not in close_df.columns:
                    continue
                prices = close_df[t].dropna()
                if len(prices) < 63:
                    continue

                c    = float(prices.iloc[-1])
                prev = float(prices.iloc[-2]) if len(prices) >= 2 else c

                def ret(n):
                    if len(prices) <= n:
                        return None
                    old = float(prices.iloc[-(n + 1)])
                    return (c - old) / old if old else None

                m1m, m3m, m6m, m12m = ret(21), ret(63), ret(126), ret(252)
                m5,  m30, m60       = ret(5),  ret(30),  ret(60)

                mom_312   = (m3m + m6m + m12m) if None not in (m3m, m6m, m12m) else None
                mom_136   = (m1m + m3m + m6m)  if None not in (m1m, m3m, m6m)  else None
                mom_short = (m5 + m30 + m60)   if None not in (m5, m30, m60)   else None

                rsi21 = round(_rsi(prices, 21), 1) if len(prices) >= 22 else None

                def pct(v): return round(v * 100, 2) if v is not None else None

                cap_raw = mktcap_map.get(t)
                results.append({
                    'ticker':    t,
                    'name':      name_map.get(t, t),
                    'sector':    sector_map.get(t, ''),
                    'mktcap':    _fmt_cap(cap_raw),
                    'mktcap_raw': int(cap_raw) if cap_raw else 0,
                    'price':     round(c, 2),
                    'change_1d': pct((c - prev) / prev) if prev else None,
                    'rsi21':     rsi21,
                    'mom_1m':    pct(m1m),
                    'mom_3m':    pct(m3m),
                    'mom_6m':    pct(m6m),
                    'mom_12m':   pct(m12m),
                    'mom_short': round(mom_short, 3) if mom_short is not None else None,
                    'mom_136':   round(mom_136,   3) if mom_136   is not None else None,
                    'mom_312':   round(mom_312,   3) if mom_312   is not None else None,
                })
            except Exception:
                continue

        results.sort(key=lambda x: x['mom_312'] if x['mom_312'] is not None else -999, reverse=True)
        # Add rank
        for i, r in enumerate(results, 1):
            r['rank'] = i

        payload = {
            'index':  index_name,
            'label':  _index_label(index_name),
            'count':  len(results),
            'stocks': results,
        }
        _mom_cache[index_name] = (now, payload)
        _db_set("momentum_cache", "index_name", index_name, payload)
        return jsonify(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/price-history")
def price_history():
    ticker_sym = request.args.get("ticker", "").upper().strip()
    if not ticker_sym:
        return jsonify({"error": "ticker manquant"}), 400
    cache_key = f"ph_{ticker_sym}"
    now_t = _time.time()
    if cache_key in _mom_cache:
        ts, data = _mom_cache[cache_key]
        if now_t - ts < 3600:
            return jsonify(data)
    try:
        raw    = yf.download(ticker_sym, period="2y", auto_adjust=True, progress=False)
        series = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        if hasattr(series, "columns"):
            series = series.iloc[:, 0]
        series  = series.dropna()
        payload = {
            "ticker": ticker_sym,
            "dates":  [d.strftime("%Y-%m-%d") for d in series.index],
            "prices": [round(float(v), 4) for v in series.values],
        }
        _mom_cache[cache_key] = (now_t, payload)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/rank-history")
def rank_history():
    """
    Retourne l'historique des classements momentum pour tous les stocks d'un indice.
    2.5 ans de données → 252 jours warmup → ~18 mois d'historique valide.
    Cache 23h.
    """
    index_name = request.args.get("index", "sp500").lower()

    now_t     = _time.time()
    cache_key = f"rh_{index_name}"
    if cache_key in _rank_hist_cache:
        ts, data = _rank_hist_cache[cache_key]
        if now_t - ts < 23 * 3600:
            return jsonify(data)

    try:
        components = _get_components(index_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not components:
        return jsonify({"error": f"Indice '{index_name}' non reconnu."}), 400

    tickers_list = [c[0] for c in components]
    if index_name in _LARGE_INDICES:
        tickers_list = tickers_list[:100]  # top 100 pour les grands indices

    try:
        start_date = (datetime.now() - timedelta(days=int(2.5 * 365))).strftime("%Y-%m-%d")
        raw = yf.download(tickers_list, start=start_date, auto_adjust=True, progress=False)

        if len(tickers_list) > 1:
            close_df = raw["Close"]
        else:
            close_df = raw[["Close"]].rename(columns={"Close": tickers_list[0]})
        close_df = close_df.dropna(how='all')

        if len(close_df) < 260:
            return jsonify({"error": "Pas assez de données historiques"}), 400

        def _mom(shifts):
            total = None
            for s in shifts:
                ret = (close_df - close_df.shift(s)) / close_df.shift(s)
                total = ret if total is None else total + ret
            return total

        mom_312_df   = _mom([63, 126, 252])
        mom_136_df   = _mom([21,  63, 126])
        mom_short_df = _mom([ 5,  30,  60])

        # Valid dates : from row 252 onwards
        mom_312_v   = mom_312_df.iloc[252:]
        mom_136_v   = mom_136_df.iloc[252:]
        mom_short_v = mom_short_df.iloc[252:]

        # Row-wise rank : 1 = best (highest score)
        ranks_312   = mom_312_v.rank(axis=1, ascending=False, method='min', na_option='keep')
        ranks_136   = mom_136_v.rank(axis=1, ascending=False, method='min', na_option='keep')
        ranks_short = mom_short_v.rank(axis=1, ascending=False, method='min', na_option='keep')

        dates_list = [d.strftime("%Y-%m-%d") for d in ranks_312.index]

        by_ticker = {}
        for t in ranks_312.columns:
            t_str = str(t)
            by_ticker[t_str] = {
                'mom_312':   [int(v) if pd.notna(v) else None for v in ranks_312[t]],
                'mom_136':   [int(v) if pd.notna(v) else None for v in ranks_136[t]],
                'mom_short': [int(v) if pd.notna(v) else None for v in ranks_short[t]],
            }

        payload = {
            'index':        index_name,
            'dates':        dates_list,
            'ranks':        by_ticker,
            'count_dates':  len(dates_list),
            'count_stocks': len(by_ticker),
        }
        _rank_hist_cache[cache_key] = (now_t, payload)
        return jsonify(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stock-info")
def stock_info():
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "ticker manquant"}), 400

    cache_key = f"si_{ticker}"
    now = _time.time()
    if cache_key in _mom_cache:
        ts, data = _mom_cache[cache_key]
        if now - ts < 900:
            return jsonify(data)

    try:
        t   = yf.Ticker(ticker)
        fi  = t.fast_info
        inf = t.info

        raw = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
        series = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        if hasattr(series, "columns"):
            series = series.iloc[:, 0]
        series = series.dropna()
        prices = pd.Series(series.values.flatten().astype(float))

        def mom_pct(n):
            # iloc[-(n+1)] = n bars back (consistent avec /momentum qui utilise iloc[-(n+1)])
            return round((float(prices.iloc[-1]) / float(prices.iloc[-(n+1)]) - 1) * 100, 2) if len(prices) > n else None

        def mom_score(*pairs):
            parts = [(prices.iloc[-1] - prices.iloc[-(n+1)]) / prices.iloc[-(n+1)] for n in pairs if len(prices) > n]
            return round(float(sum(parts)), 4) if len(parts) == len(pairs) else None

        last  = float(fi.last_price)     if fi.last_price     else None
        prev  = float(fi.previous_close) if fi.previous_close else None
        ch_1d = round((last / prev - 1) * 100, 2) if last and prev else None

        sector = inf.get("sector") or inf.get("category") or inf.get("fundFamily") or "—"

        result = {
            "ticker":     ticker,
            "name":       inf.get("longName") or inf.get("shortName") or ticker,
            "sector":     sector,
            "price":      round(last, 2) if last else None,
            "mktcap":     _fmt_cap(fi.market_cap),
            "mktcap_raw": fi.market_cap,
            "change_1d":  ch_1d,
            "mom_1m":     mom_pct(21),
            "mom_3m":     mom_pct(63),
            "mom_6m":     mom_pct(126),
            "mom_12m":    mom_pct(252),
            "mom_short":  mom_score(5, 30, 60),
            "mom_136":    mom_score(21, 63, 126),
            "mom_312":    mom_score(63, 126, 252),
            "rsi21":      round(_rsi(prices), 1) if len(prices) >= 21 else None,
        }
        _mom_cache[cache_key] = (_time.time(), result)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/quick-stats")
def quick_stats():
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "ticker manquant"}), 400

    cache_key = f"qs_{ticker}"
    now = _time.time()

    # L1 : mémoire
    if cache_key in _mom_cache:
        ts, data = _mom_cache[cache_key]
        if now - ts < 3600:
            return jsonify(data)

    # L2 : Supabase
    db_data = _db_get("quick_stats_cache", "ticker", ticker, 3600)
    if db_data:
        _mom_cache[cache_key] = (_time.time(), db_data)
        return jsonify(db_data)

    try:
        # yf.download fonctionne en datacenter (crumb géré automatiquement)
        raw = yf.download(ticker, period="10y", auto_adjust=True, progress=False)
        if raw.empty:
            return jsonify({"error": "Aucune donnée"}), 400

        close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        # Aplatir si MultiIndex colonnes (yfinance peut retourner DataFrame à 2 col)
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        series = close.dropna()
        if len(series) < 60:
            return jsonify({"error": "Historique insuffisant"}), 400

        prices = np.array(series.values, dtype=float).flatten()
        dates  = series.index

        x = np.array([(d - dates[0]).days for d in dates], dtype=float)
        y = np.log(prices)

        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        y_fit    = slope * x + intercept
        sigma    = (y - y_fit).std()

        annual_return   = (np.exp(slope * 365) - 1) * 100
        last_price      = float(prices[-1])
        sigma_pct       = (np.exp(sigma) - 1) * 100
        sigma_dollars   = last_price * (np.exp(sigma) - 1)

        last_reg_log    = slope * x[-1] + intercept
        deviation_sigma = (np.log(last_price) - last_reg_log) / sigma
        deviation_pct   = (last_price / np.exp(last_reg_log) - 1) * 100

        result = {
            "ticker":          ticker,
            "annual_return":   round(annual_return, 2),
            "sigma_pct":       round(sigma_pct, 2),
            "sigma_dollars":   round(sigma_dollars, 2),
            "deviation_sigma": round(deviation_sigma, 2),
            "deviation_pct":   round(deviation_pct, 2),
            "r2":              round(r_value ** 2, 4),
            "data_start":      dates[0].strftime("%d/%m/%Y"),
            "n_years":         round((dates[-1] - dates[0]).days / 365.25, 1),
        }
        _mom_cache[cache_key] = (_time.time(), result)
        _db_set("quick_stats_cache", "ticker", ticker, result)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug-download")
def debug_download():
    """Diagnostique la structure de yf.download multi-tickers sur Railway."""
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    try:
        raw = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
        info = {
            "shape": list(raw.shape),
            "empty": raw.empty,
            "columns_type": str(type(raw.columns).__name__),
            "columns_sample": str(raw.columns.tolist()[:10]),
            "has_Close_key": "Close" in raw.columns,
        }
        if not raw.empty and "Close" in raw.columns:
            close = raw["Close"]
            info["close_type"] = str(type(close).__name__)
            info["close_cols"] = str(list(close.columns)[:5]) if hasattr(close, 'columns') else "Series"
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/version")
def version():
    return {"version": "2.0-vanguard-supabase", "russell2000_source": "vanguard_vtwo"}

@app.route("/debug-russell")
def debug_russell():
    import sys
    result = {}
    # 1. Check memory cache
    result["in_memory"] = "russell2000" in _mom_cache
    # 2. Check DB cache
    try:
        db = _db_get("momentum_cache", "index_name", "russell2000", 999999)
        result["in_db"] = db is not None
        if db:
            result["db_keys"] = list(db.keys())[:5]
    except Exception as e:
        result["db_error"] = str(e)
    # 3. Test Vanguard API
    try:
        r = requests.get(
            "https://investor.vanguard.com/investment-products/etfs/profile/api/vtwo/portfolio-holding/stock",
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            params={"sortBy": "weighting", "sortOrder": "desc", "perPage": 5, "page": 1},
            timeout=15
        )
        result["vanguard_status"] = r.status_code
        result["vanguard_content_type"] = r.headers.get("Content-Type", "?")
        result["vanguard_body_start"] = r.text[:100]
    except Exception as e:
        result["vanguard_error"] = str(e)
    return jsonify(result)

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, port=port, threaded=True)
