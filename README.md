# Volatility Targeting — EGARCH(1,1) with AFML Validation Suite

Stratégie de **volatility targeting** sur GLD (Gold ETF) utilisant un modèle **EGARCH(1,1)** en rolling window, avec une suite de validation complète inspirée de *Advances in Financial Machine Learning* (López de Prado).

## Principe

La volatilité conditionnelle est estimée quotidiennement via un EGARCH(1,1) à distribution Student-t, recalibré sur fenêtre glissante de 1000 jours. Le levier est ajusté dynamiquement pour stabiliser la volatilité du portefeuille autour d'une cible mobile.

```
Levier(t) = VolTarget(t) / VolForecast(t)    clippé [0, 2]
Return_strat(t+1) = Levier(t) × Return_asset(t+1)
```

### Anti look-ahead bias — 3 niveaux

| Couche | Mécanisme | Garantie |
|--------|-----------|----------|
| EGARCH | Fit sur `[t-W, t]`, forecast pour `t+1` | Aucune donnée future dans la fenêtre |
| Vol target | `rolling(252).mean().shift(1)` | Cible en `t` ne voit que `t-1` et avant |
| Levier | `leverage.shift(1) × log_ret` | Décision en `t`, appliquée au return de `t+1` |

## Validation AFML — 500k simulations

| Module | Volume | Méthode |
|--------|--------|---------|
| **Monte Carlo** | 250,000 paths | Block bootstrap (blocs 21j), vectorisé par batch de 5000 |
| **CPCV** | 45 combinaisons | 10 groupes, k=2 test, purge 5j + embargo 2j |
| **DSR** | — | Formule Lo (2002) corrigée skew/kurtosis, E\[max SR\] Gumbel |
| **CSCV PBO** | 126 paires | Partitions symétriques purgées, dégradation IS→OOS |
| **Walk-Forward** | ~1,900 fenêtres + 250,000 MC-WF | Rolling step=1j OOS=63j, puis bootstrap sur pool OOS |
| **Stress 6σ** | 10,000 scénarios | Student-t normalisé + injection 1-6 chocs 6σ aléatoires |

### Verdict automatique — 8 critères

| Check | Seuil | Source |
|-------|-------|--------|
| Sharpe observé > 0 | — | Performance de base |
| MC P(Sharpe ≤ 0) < 10% | 250k paths | Monte Carlo |
| CPCV Sharpe OOS moyen > 0 | 45 combos | Cross-validation combinatoire |
| DSR p-value < 0.05 | — | Deflated Sharpe Ratio |
| PBO < 0.50 | 126 paires | Probability of Backtest Overfitting |
| WF Rolling % Sharpe > 0 > 60% | ~1,900 fenêtres | Walk-Forward |
| WF MC P(Sharpe ≤ 0) < 10% | 250k paths | MC sur OOS |
| Stress P(ruine > 50%) < 10% | 10k scénarios | Stress test 6σ |

```
≥ 7/8 → STRATÉGIE VALIDÉE
≥ 5/8 → STRATÉGIE MARGINALE
< 5/8 → STRATÉGIE NON VALIDÉE
```

## Pipeline

```
2011───────────2015──────────────────────────today
 │  warm-up     │        évaluation              │
 │  EGARCH fit   │                                │
 │  + lookback   │                                │
 └──────────────┴────────────────────────────────┘
                 ↓
         ┌──────────────┐
         │ Performance   │ CAGR, Vol, Sharpe, MaxDD, PnL
         │ de base       │
         └──────┬───────┘
                ↓
    ┌───────────┴───────────┐
    ↓                       ↓
 MC 250k              CPCV + DSR
 block bootstrap      purge/embargo
    ↓                       ↓
    ├───────────┬───────────┤
    ↓           ↓           ↓
 CSCV PBO    WF massif   Stress 6σ
 IS↔OOS      250k MC-WF  Student-t
    ↓           ↓           ↓
    └───────────┴───────────┘
                ↓
          VERDICT x/8
```

## Output

Le script produit un dashboard PNG de 10 panneaux :

| Panneau | Contenu |
|---------|---------|
| Equity curve | Log returns cumulés B&H vs stratégie |
| Levier | Exposition dynamique dans le temps |
| Volatilité | Forecast annualisé vs target |
| PnL | Courbe en dollars ($100k initial) |
| MC Sharpe | Distribution 250k + observé |
| CPCV Sharpe | Distribution OOS 45 combos |
| WF Rolling | Time series Sharpe sur ~1,900 fenêtres |
| CSCV IS↔OOS | Scatter + corrélation + PBO |
| MC-WF | Distribution 250k + MaxDD |
| Stress | Terminal PnL + MaxDD sous 6σ |
| Recap | Tableau des 13 métriques + verdict |

## Installation

```bash
pip install numpy pandas yfinance arch scipy matplotlib tqdm
```

## Usage

```bash
python vol_targeting_egarch.py
```

Le script télécharge automatiquement les données GLD via yfinance. Temps d'exécution estimé : ~20-40 min (dominé par le rolling EGARCH).

## Configuration

Tous les paramètres sont regroupés en tête de script :

```python
TICKER = "GLD"
EVAL_START = "2015-01-01"
WINDOW_SIZE = 1000        # fenêtre EGARCH
L_MAX = 2.0               # levier max
MC_ITERATIONS = 250_000   # paths Monte Carlo
WF_MC_ITERATIONS = 250_000
STRESS_N_SCENARIOS = 10000
STRESS_SIGMA = 6
```

## QA Audit Trail

Le code a subi 4 passes d'audit QA. Corrections majeures :

| Version | Corrections |
|---------|-------------|
| v2 | DSR Var(SR) formula (Lo 2002), Student-t normalisation, fallback EGARCH, pandas deprecation, CPCV reste, purge/embargo symétrique |
| v3 | Vectorisation MC 250k par batch, WF 2 phases (rolling + MC-WF 250k) |
| v4 | **ddof=1 unifié** sur les 10 appels `.std()` du script. Élimination du mismatch ddof=0 (scalaire) vs ddof=1 (vectorisé). WF Phase 1 vectorisé via `sliding_window_view`. |

## Références

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio*. Journal of Portfolio Management.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). *The Probability of Backtest Overfitting*. Journal of Computational Finance.
- Lo, A. W. (2002). *The Statistics of Sharpe Ratios*. Financial Analysts Journal.
- Nelson, D. B. (1991). *Conditional Heteroskedasticity in Asset Returns: A New Approach*. Econometrica.

## Licence

Usage personnel et éducatif. Pas de conseil financier.
