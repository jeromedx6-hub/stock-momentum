import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import pandas as pd
from datetime import datetime

# ── 1. DATA ──────────────────────────────────────────────────────────────────
print("Téléchargement des données SCHD depuis Yahoo Finance…")
df = yf.download("SCHD", start="2011-10-20", auto_adjust=True, progress=False)
df = df["Close"].dropna()
df.index = pd.to_datetime(df.index)

prices = df.values.flatten()
dates  = df.index

# ── 2. RÉGRESSION LOG-LINÉAIRE ────────────────────────────────────────────────
# x = jours depuis le premier point (numérique)
x = np.array([(d - dates[0]).days for d in dates], dtype=float)
y = np.log(prices)

slope, intercept, r_value, _, _ = stats.linregress(x, y)

# Droite de régression + résidus
y_fit      = slope * x + intercept
residuals  = y - y_fit
sigma      = residuals.std()          # 1 écart type dans l'espace log

# ── 3. MÉTRIQUES ─────────────────────────────────────────────────────────────
# Pente annuelle : e^(slope * 365) - 1
annual_return = (np.exp(slope * 365) - 1) * 100

# Distance d'un sigma en % (à la date la plus récente)
last_price    = prices[-1]
price_plus1   = last_price * np.exp(sigma)
sigma_pct     = (price_plus1 / last_price - 1) * 100
sigma_dollars = last_price * (np.exp(sigma) - 1)

print(f"Pente annuelle      : {annual_return:.2f} %/an")
print(f"1 écart type (σ)    : ±{sigma_pct:.2f}% / ±${sigma_dollars:.2f}")
print(f"R²                  : {r_value**2:.4f}")
print(f"Données             : {dates[0].date()} → {dates[-1].date()} ({len(dates)} jours)")

# ── 4. GRAPHIQUE ──────────────────────────────────────────────────────────────
fig, (ax_main, ax_info) = plt.subplots(
    2, 1,
    figsize=(14, 9),
    gridspec_kw={"height_ratios": [6, 1]},
    facecolor="#0d1117"
)

fig.suptitle(
    "SCHD — Droite de régression log-linéaire avec écarts types",
    color="white", fontsize=14, fontweight="bold", y=0.98
)

# ── Axe principal ─────────────────────────────────────────────────────────────
ax_main.set_facecolor("#0d1117")

# Prix réels
ax_main.semilogy(dates, prices, color="#4fc3f7", linewidth=1.2,
                 label="SCHD (cours ajusté)", zorder=3)

# Droite de régression (espace prix)
reg_prices = np.exp(y_fit)
ax_main.semilogy(dates, reg_prices, color="#ffffff", linewidth=1.5,
                 linestyle="--", label="Régression", zorder=4)

# Bandes
ax_main.fill_between(dates,
    np.exp(y_fit - sigma), np.exp(y_fit + sigma),
    color="#66bb6a", alpha=0.20, label="± 1σ", zorder=1)
ax_main.fill_between(dates,
    np.exp(y_fit - 2*sigma), np.exp(y_fit - sigma),
    color="#ffa726", alpha=0.13, zorder=1)
ax_main.fill_between(dates,
    np.exp(y_fit + sigma), np.exp(y_fit + 2*sigma),
    color="#ffa726", alpha=0.13, label="± 2σ", zorder=1)
ax_main.fill_between(dates,
    np.exp(y_fit - 3*sigma), np.exp(y_fit - 2*sigma),
    color="#ef5350", alpha=0.10, zorder=1)
ax_main.fill_between(dates,
    np.exp(y_fit + 2*sigma), np.exp(y_fit + 3*sigma),
    color="#ef5350", alpha=0.10, label="± 3σ", zorder=1)

# Lignes de sigma
for k, col, ls in [(1, "#66bb6a", "-"), (2, "#ffa726", "--"), (-1, "#66bb6a", "-"), (-2, "#ffa726", "--")]:
    ax_main.semilogy(dates, np.exp(y_fit + k*sigma),
                     color=col, linewidth=0.6, linestyle=ls, alpha=0.7, zorder=2)

# Point actuel
ax_main.scatter([dates[-1]], [prices[-1]], color="#4fc3f7",
                s=50, zorder=5, label=f"Dernier : ${prices[-1]:.2f}")

# Mise en forme
ax_main.set_ylabel("Prix ($) — échelle log", color="white", fontsize=10)
ax_main.tick_params(colors="white")
ax_main.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"${v:.0f}"))
for spine in ax_main.spines.values():
    spine.set_edgecolor("#333333")
ax_main.grid(True, color="#1e2530", linewidth=0.5)
ax_main.legend(loc="upper left", facecolor="#1a1a2e", edgecolor="#333333",
               labelcolor="white", fontsize=8.5)

# Annotation R²
ax_main.text(0.01, 0.97, f"R² = {r_value**2:.4f}", transform=ax_main.transAxes,
             color="#aaaaaa", fontsize=8, va="top")

# ── Panneau info ──────────────────────────────────────────────────────────────
ax_info.set_facecolor("#111827")
ax_info.axis("off")

# Calcul position actuelle en sigmas
last_x       = x[-1]
last_reg_log = slope * last_x + intercept
last_sigma_n = (np.log(prices[-1]) - last_reg_log) / sigma

info_text = (
    f"  Pente annuelle (CAGR régression) : {annual_return:.2f} %/an    |    "
    f"1 σ = ±{sigma_pct:.2f}% (±${sigma_dollars:.2f})    |    "
    f"Position actuelle : {last_sigma_n:+.2f} σ par rapport à la médiane    |    "
    f"Source : Yahoo Finance via yfinance   |   "
    f"Données : {dates[0].strftime('%d/%m/%Y')} → {dates[-1].strftime('%d/%m/%Y')}"
)
ax_info.text(0.0, 0.5, info_text, transform=ax_info.transAxes,
             color="#cccccc", fontsize=8.5, va="center",
             fontfamily="monospace")

plt.tight_layout(rect=[0, 0, 1, 0.97])

output_path = "/Users/jeromedx6/Desktop/claude code/Nina Voit/SCHD_regression.png"
plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="#0d1117")
print(f"\nGraphique sauvegardé : {output_path}")
plt.show()
