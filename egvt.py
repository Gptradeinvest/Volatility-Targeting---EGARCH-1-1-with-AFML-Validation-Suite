"""
Volatility Targeting — EGARCH(1,1) rolling forecast
Ticker: GLD | Periode: 2015-today | Anti look-ahead bias
+ Validation AFML: MC 250k, CPCV, DSR, CSCV PBO, WF massif 250k, 6sigma stress

QA AUDIT v4 — ddof=1 unifie, WF vectorise
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from arch import arch_model
from tqdm import tqdm
from itertools import combinations
from scipy.stats import norm, t as t_dist, skew, kurtosis as kurt_fn
from numpy.lib.stride_tricks import sliding_window_view
import time

pd.options.mode.chained_assignment = None
plt.style.use("dark_background")


# CONFIG

TICKER = "GLD"
DOWNLOAD_START = "2011-01-01"
EVAL_START = "2015-01-01"
SCALE = 100.0
WINDOW_SIZE = 1000
L_MAX = 2.0
LOOKBACK_TARGET = 252
INITIAL_CAPITAL = 100_000
PURGE_DAYS = 5
EMBARGO_DAYS = 2

# --- Validation ---
MC_ITERATIONS = 250_000
MC_BLOCK_SIZE = 21
MC_BATCH_SIZE = 5000          # vectorisation par batch (RAM ~100MB/batch)
CPCV_N_GROUPS = 10
CPCV_K_TEST = 2
WF_MIN_TRAIN_DAYS = 504       # 2 ans
WF_OOS_WINDOW = 63            # 1 trimestre OOS par fenetre rolling
WF_MC_ITERATIONS = 250_000    # MC bootstrap sur OOS returns
WF_MC_BLOCK_SIZE = 21         # block size pour MC-WF
WF_MC_BATCH_SIZE = 5000
STRESS_N_SCENARIOS = 10000
STRESS_SIGMA = 6


# 1. DATA

print(f"[1] Download {TICKER} from {DOWNLOAD_START}")
df = yf.download(TICKER, start=DOWNLOAD_START, progress=False, auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df = df.xs(TICKER, axis=1, level=1)

prices = df["Close"].replace(0, np.nan).ffill()
log_ret = np.log(prices / prices.shift(1))
log_ret.replace([np.inf, -np.inf], np.nan, inplace=True)
log_ret = log_ret.dropna()
log_ret.name = "log_ret"
returns_scaled = log_ret * SCALE


# 2. ROLLING EGARCH FORECAST

print(f"[2] Rolling EGARCH — window={WINDOW_SIZE}, scale=x{SCALE}")

forecasts = {}
last_params = None
previous_vol = None
_fallback_vol = returns_scaled.std(ddof=1) / SCALE

for i in tqdm(range(len(returns_scaled) - WINDOW_SIZE)):
    window = returns_scaled.iloc[i : i + WINDOW_SIZE]
    calc_date = window.index[-1]
    mdl = arch_model(window, vol="EGARCH", p=1, o=1, q=1, dist="t")
    fit_kw = dict(disp=False, show_warning=False, options={"ftol": 1e-10, "maxiter": 5000})
    if last_params is not None:
        fit_kw["starting_values"] = last_params
    try:
        res = mdl.fit(**fit_kw)
        fcast = res.forecast(horizon=1, reindex=False)
        pred_vol = np.sqrt(fcast.variance.iloc[0, 0]) / SCALE
    except Exception:
        pred_vol = np.nan
    if not np.isfinite(pred_vol) or pred_vol >= 100 or pred_vol <= 1e-6:
        pred_vol = previous_vol if previous_vol is not None else _fallback_vol
        last_params = None
    else:
        last_params = res.params
    previous_vol = pred_vol
    forecasts[calc_date] = pred_vol

print(f"  -> {len(forecasts)} forecasts")


# 3. STRATEGY ASSEMBLY — ANTI LOOK-AHEAD

print("[3] Construction strategie")

forecast_series = pd.Series(forecasts, name="egarch_vol_fcast")
portfolio = pd.DataFrame({"log_ret": log_ret})
portfolio = portfolio.join(forecast_series).dropna()

portfolio["vol_target"] = (
    portfolio["egarch_vol_fcast"]
    .rolling(window=LOOKBACK_TARGET).mean().shift(1)
    .fillna(0.4 * np.sqrt(1 / 252))
)
portfolio["leverage"] = (
    (portfolio["vol_target"] / portfolio["egarch_vol_fcast"]).fillna(0).clip(0, L_MAX)
)
portfolio["strat_ret"] = portfolio["leverage"].shift(1) * portfolio["log_ret"]
portfolio.dropna(inplace=True)
portfolio = portfolio.loc[EVAL_START:]

strat_ret = portfolio["strat_ret"].values.copy()
bh_ret = portfolio["log_ret"].values.copy()
dates = portfolio.index
N = len(strat_ret)

print(f"  -> {dates[0].date()} -> {dates[-1].date()} ({N} jours)")


# 4. UTILITAIRES

def calc_metrics(r, freq=252):
    n = len(r)
    if n < 10:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    r_arr = np.asarray(r, dtype=np.float64)
    cagr = np.exp(r_arr.sum()) ** (freq / n) - 1
    vol = r_arr.std(ddof=1) * np.sqrt(freq)
    sharpe = (r_arr.mean() * freq) / (vol + 1e-9)
    cum = np.cumsum(r_arr)
    dd = cum - np.maximum.accumulate(cum)
    max_dd = np.exp(dd.min()) - 1
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": max_dd}

def sharpe_ratio(r, freq=252):
    """Sharpe scalaire, ddof=1 aligne avec sharpe_vec."""
    if len(r) < 10:
        return 0.0
    r_arr = np.asarray(r, dtype=np.float64)
    return (r_arr.mean() * freq) / (r_arr.std(ddof=1) * np.sqrt(freq) + 1e-9)

def sharpe_vec(paths_2d, freq=252):
    """Sharpe vectorise sur matrice (n_paths, n_days)."""
    mu = paths_2d.mean(axis=1) * freq
    vol = paths_2d.std(axis=1, ddof=1) * np.sqrt(freq)
    return mu / (vol + 1e-9)

def maxdd_vec(paths_2d):
    """MaxDD vectorise sur matrice (n_paths, n_days)."""
    cum = np.cumsum(paths_2d, axis=1)
    running_max = np.maximum.accumulate(cum, axis=1)
    dd = cum - running_max
    return np.exp(dd.min(axis=1)) - 1

def cagr_vec(paths_2d, freq=252):
    """CAGR vectorise."""
    n = paths_2d.shape[1]
    cum_total = paths_2d.sum(axis=1)
    return np.exp(cum_total) ** (freq / n) - 1

def build_purge_embargo_set(test_group_ids, groups, n_total, purge_d, embargo_d):
    purge_set = set()
    for g in test_group_ids:
        g_start, g_end = groups[g][0], groups[g][-1]
        for p in range(max(0, g_start - purge_d), g_start):
            purge_set.add(p)
        for p in range(g_end + 1, min(n_total, g_end + embargo_d + 1)):
            purge_set.add(p)
    return purge_set

def block_bootstrap_batch(data, n_paths, path_len, block_size, rng):
    """Genere n_paths block-bootstrap paths vectorise."""
    n_data = len(data)
    eff_block = min(block_size, path_len)
    n_blocks = int(np.ceil(path_len / eff_block))
    # (n_paths, n_blocks) random start indices
    starts = rng.integers(0, n_data - eff_block + 1, size=(n_paths, n_blocks))
    # Build index matrix (n_paths, n_blocks * eff_block)
    offsets = np.arange(eff_block)  # (eff_block,)
    # (n_paths, n_blocks, eff_block)
    idx = starts[:, :, None] + offsets[None, None, :]
    idx = idx.reshape(n_paths, -1)[:, :path_len]
    return data[idx]  # (n_paths, path_len)


# 5. PERFORMANCE DE BASE

m_bh = calc_metrics(pd.Series(bh_ret))
m_st = calc_metrics(pd.Series(strat_ret))
df_m = pd.DataFrame({"Buy&Hold": m_bh, "EGARCH VolTarget": m_st})
print("\n" + "=" * 60)
print("PERFORMANCE DE BASE")
print("=" * 60)
print(df_m.round(4).to_string())

pnl_bh = INITIAL_CAPITAL * (np.exp(np.cumsum(bh_ret)) - 1)
pnl_st = INITIAL_CAPITAL * (np.exp(np.cumsum(strat_ret)) - 1)
print(f"\nPnL Buy&Hold  : ${pnl_bh[-1]:,.0f}")
print(f"PnL VolTarget : ${pnl_st[-1]:,.0f}")

observed_sharpe = sharpe_ratio(strat_ret)


# 6. MONTE CARLO 250k — BLOCK BOOTSTRAP VECTORISE

print(f"\n{'=' * 60}")
print(f"MONTE CARLO — {MC_ITERATIONS:,} iterations, block={MC_BLOCK_SIZE}d, batch={MC_BATCH_SIZE}")
print("=" * 60)
t0 = time.time()

rng = np.random.default_rng(42)
mc_sharpes = np.empty(MC_ITERATIONS)
mc_cagrs = np.empty(MC_ITERATIONS)
mc_maxdds = np.empty(MC_ITERATIONS)

n_batches = int(np.ceil(MC_ITERATIONS / MC_BATCH_SIZE))

for b in tqdm(range(n_batches), desc="MC 250k"):
    i_start = b * MC_BATCH_SIZE
    i_end = min(i_start + MC_BATCH_SIZE, MC_ITERATIONS)
    batch_n = i_end - i_start

    paths = block_bootstrap_batch(strat_ret, batch_n, N, MC_BLOCK_SIZE, rng)

    mc_sharpes[i_start:i_end] = sharpe_vec(paths)
    mc_cagrs[i_start:i_end] = cagr_vec(paths)
    mc_maxdds[i_start:i_end] = maxdd_vec(paths)

mc_time = time.time() - t0
mc_sharpe_p = np.mean(mc_sharpes <= 0)
mc_sharpe_ci = np.percentile(mc_sharpes, [2.5, 97.5])

print(f"  Temps                  : {mc_time:.1f}s")
print(f"  Sharpe observe         : {observed_sharpe:.4f}")
print(f"  Sharpe MC median       : {np.median(mc_sharpes):.4f}")
print(f"  Sharpe MC mean         : {np.mean(mc_sharpes):.4f}")
print(f"  Sharpe MC std          : {np.std(mc_sharpes):.4f}")
print(f"  Sharpe MC CI 95%       : [{mc_sharpe_ci[0]:.4f}, {mc_sharpe_ci[1]:.4f}]")
print(f"  Sharpe MC CI 99%       : [{np.percentile(mc_sharpes, 0.5):.4f}, {np.percentile(mc_sharpes, 99.5):.4f}]")
print(f"  P(Sharpe <= 0)         : {mc_sharpe_p:.6f}")
print(f"  CAGR MC CI 95%         : [{np.percentile(mc_cagrs, 2.5):.4f}, {np.percentile(mc_cagrs, 97.5):.4f}]")
print(f"  MaxDD MC CI 95%        : [{np.percentile(mc_maxdds, 2.5):.4f}, {np.percentile(mc_maxdds, 97.5):.4f}]")


# 7. CPCV — COMBINATORIAL PURGED CROSS-VALIDATION

print(f"\n{'=' * 60}")
print(f"CPCV — N={CPCV_N_GROUPS}, k={CPCV_K_TEST}, purge={PURGE_DAYS}d, embargo={EMBARGO_DAYS}d")
print("=" * 60)

group_size = N // CPCV_N_GROUPS
groups = []
for i in range(CPCV_N_GROUPS):
    start = i * group_size
    end = (i + 1) * group_size if i < CPCV_N_GROUPS - 1 else N
    groups.append(np.arange(start, end))

combos = list(combinations(range(CPCV_N_GROUPS), CPCV_K_TEST))
cpcv_sharpes = []

for test_ids in tqdm(combos, desc="CPCV"):
    test_idx = np.concatenate([groups[g] for g in test_ids])
    train_idx_raw = np.concatenate([groups[g] for g in range(CPCV_N_GROUPS) if g not in test_ids])
    purge_set = build_purge_embargo_set(test_ids, groups, N, PURGE_DAYS, EMBARGO_DAYS)
    test_set = set(test_idx)
    train_idx = np.array([i for i in train_idx_raw if i not in purge_set and i not in test_set])
    if len(train_idx) < 50 or len(test_idx) < 10:
        continue
    test_r = strat_ret[test_idx]
    cpcv_sharpes.append(sharpe_ratio(test_r))

cpcv_sharpes = np.array(cpcv_sharpes)
cpcv_mean = cpcv_sharpes.mean()
cpcv_std = cpcv_sharpes.std(ddof=1)
cpcv_ci = np.percentile(cpcv_sharpes, [2.5, 97.5])
cpcv_p_negative = np.mean(cpcv_sharpes <= 0)

print(f"  Combinaisons           : {len(cpcv_sharpes)}")
print(f"  Sharpe OOS moyen       : {cpcv_mean:.4f}")
print(f"  Sharpe OOS std         : {cpcv_std:.4f}")
print(f"  Sharpe OOS CI 95%      : [{cpcv_ci[0]:.4f}, {cpcv_ci[1]:.4f}]")
print(f"  P(Sharpe OOS <= 0)     : {cpcv_p_negative:.4f}")


# 8. DSR — DEFLATED SHARPE RATIO

print(f"\n{'=' * 60}")
print("DEFLATED SHARPE RATIO")
print("=" * 60)

sr = observed_sharpe
T = N
gamma3 = skew(strat_ret)
gamma4 = kurt_fn(strat_ret, fisher=True)
sr_var = (1.0 / (T - 1)) * (1.0 - gamma3 * sr + ((gamma4 + 2.0) / 4.0) * sr**2)
sr_std = np.sqrt(max(sr_var, 1e-12))

N_trials = max(len(combos), 10)
if N_trials > 1:
    _a = np.sqrt(2.0 * np.log(N_trials))
    e_max_z = _a - (np.log(np.log(N_trials)) + np.log(4 * np.pi)) / (2.0 * _a)
else:
    e_max_z = 0.0
e_max_sr = e_max_z * np.sqrt(1.0 / T)

dsr_z = (sr - e_max_sr) / (sr_std + 1e-9)
dsr_pval = 1 - norm.cdf(dsr_z)
dsr_pass = dsr_pval < 0.05

print(f"  Sharpe observe         : {sr:.4f}")
print(f"  gamma3={gamma3:.4f}  gamma4={gamma4:.4f}")
print(f"  SR std error           : {sr_std:.4f}")
print(f"  E[max SR] H0           : {e_max_sr:.4f} (N_trials={N_trials})")
print(f"  DSR z-stat             : {dsr_z:.4f}")
print(f"  DSR p-value            : {dsr_pval:.4f}")
print(f"  DSR PASS (p<0.05)      : {'OUI' if dsr_pass else 'NON'}")


# 9. CSCV PBO

print(f"\n{'=' * 60}")
print(f"CSCV PBO — {CPCV_N_GROUPS} groups")
print("=" * 60)

half = CPCV_N_GROUPS // 2
sym_combos = list(combinations(range(CPCV_N_GROUPS), half))
sym_pairs = []
seen = set()
for c in sym_combos:
    complement = tuple(g for g in range(CPCV_N_GROUPS) if g not in c)
    key = tuple(sorted([c, complement]))
    if key not in seen:
        seen.add(key)
        sym_pairs.append((c, complement))

logit_lambdas = []
is_sharpes_pbo = []
oos_sharpes_pbo = []

for s_groups, sc_groups in tqdm(sym_pairs, desc="CSCV PBO"):
    s_idx = np.concatenate([groups[g] for g in s_groups])
    sc_idx = np.concatenate([groups[g] for g in sc_groups])
    purge_s = build_purge_embargo_set(s_groups, groups, N, PURGE_DAYS, EMBARGO_DAYS)
    purge_sc = build_purge_embargo_set(sc_groups, groups, N, PURGE_DAYS, EMBARGO_DAYS)
    all_purge = purge_s | purge_sc
    s_idx_clean = np.array([i for i in s_idx if i not in all_purge])
    sc_idx_clean = np.array([i for i in sc_idx if i not in all_purge])
    if len(s_idx_clean) < 50 or len(sc_idx_clean) < 50:
        continue
    sr_is = sharpe_ratio(strat_ret[s_idx_clean])
    sr_oos = sharpe_ratio(strat_ret[sc_idx_clean])
    is_sharpes_pbo.append(sr_is)
    oos_sharpes_pbo.append(sr_oos)
    logit_lambdas.append(1 if sr_oos <= 0 else 0)

is_sharpes_pbo = np.array(is_sharpes_pbo)
oos_sharpes_pbo = np.array(oos_sharpes_pbo)
pbo = np.mean(logit_lambdas) if logit_lambdas else 1.0
degradation = is_sharpes_pbo.mean() - oos_sharpes_pbo.mean()
correlation_is_oos = np.corrcoef(is_sharpes_pbo, oos_sharpes_pbo)[0, 1] if len(is_sharpes_pbo) > 2 else 0

print(f"  Paires symetriques     : {len(sym_pairs)}")
print(f"  Sharpe IS moyen        : {is_sharpes_pbo.mean():.4f}")
print(f"  Sharpe OOS moyen       : {oos_sharpes_pbo.mean():.4f}")
print(f"  Degradation IS->OOS    : {degradation:.4f}")
print(f"  Correlation IS<->OOS   : {correlation_is_oos:.4f}")
print(f"  PBO                    : {pbo:.4f}")
print(f"  PBO PASS (<0.50)       : {'OUI' if pbo < 0.50 else 'NON'}")


# 10. WALK-FORWARD MASSIF — ROLLING + MC 250k

print(f"\n{'=' * 60}")
print(f"WALK-FORWARD MASSIF")
print(f"  Phase 1: Rolling step=1d, OOS={WF_OOS_WINDOW}d, min_train={WF_MIN_TRAIN_DAYS}d")
print(f"  Phase 2: MC-WF {WF_MC_ITERATIONS:,} bootstrap paths")
print("=" * 60)
t0 = time.time()

# --- Phase 1: Rolling WF step=1 ---
n_wf_windows = N - WF_MIN_TRAIN_DAYS - WF_OOS_WINDOW + 1

if n_wf_windows < 10:
    print(f"  ERREUR: pas assez de donnees (N={N}, besoin>{WF_MIN_TRAIN_DAYS+WF_OOS_WINDOW})")
    wf_rolling_sharpes = np.array([observed_sharpe])
    wf_oos_returns_concat = strat_ret.copy()
else:
    # Vectorise: construire matrice (n_windows, OOS_WINDOW) via stride_tricks
    oos_matrix = sliding_window_view(strat_ret[WF_MIN_TRAIN_DAYS:], WF_OOS_WINDOW)
    oos_matrix = oos_matrix[:n_wf_windows]  # (n_wf_windows, WF_OOS_WINDOW)
    wf_rolling_sharpes = sharpe_vec(oos_matrix)  # ddof=1, aligne avec MC

    # Pool OOS pour Phase 2 (tous les OOS returns concatenes, avec overlap)
    wf_oos_returns_concat = oos_matrix.ravel()

wf_roll_mean = wf_rolling_sharpes.mean()
wf_roll_std = wf_rolling_sharpes.std(ddof=1)
wf_roll_pct_pos = np.mean(wf_rolling_sharpes > 0)
wf_roll_ci = np.percentile(wf_rolling_sharpes, [2.5, 97.5])

print(f"\n  --- Phase 1: Rolling WF ---")
print(f"  Fenetres OOS           : {len(wf_rolling_sharpes):,}")
print(f"  Sharpe OOS moyen       : {wf_roll_mean:.4f}")
print(f"  Sharpe OOS std         : {wf_roll_std:.4f}")
print(f"  Sharpe OOS CI 95%      : [{wf_roll_ci[0]:.4f}, {wf_roll_ci[1]:.4f}]")
print(f"  % fenetres Sharpe > 0  : {wf_roll_pct_pos:.2%}")
print(f"  Pire fenetre           : {wf_rolling_sharpes.min():.4f}")
print(f"  Meilleure fenetre      : {wf_rolling_sharpes.max():.4f}")

# --- Phase 2: MC-WF 250k bootstrap ---
# On bootstrap des paths de longueur N depuis les OOS returns concatenes
print(f"\n  --- Phase 2: MC-WF {WF_MC_ITERATIONS:,} paths ---")

wf_mc_sharpes = np.empty(WF_MC_ITERATIONS)
wf_mc_cagrs = np.empty(WF_MC_ITERATIONS)
wf_mc_maxdds = np.empty(WF_MC_ITERATIONS)
oos_pool = wf_oos_returns_concat  # pool de OOS returns

n_wf_batches = int(np.ceil(WF_MC_ITERATIONS / WF_MC_BATCH_SIZE))

for b in tqdm(range(n_wf_batches), desc="MC-WF 250k"):
    i_start = b * WF_MC_BATCH_SIZE
    i_end = min(i_start + WF_MC_BATCH_SIZE, WF_MC_ITERATIONS)
    batch_n = i_end - i_start

    paths = block_bootstrap_batch(oos_pool, batch_n, N, WF_MC_BLOCK_SIZE, rng)

    wf_mc_sharpes[i_start:i_end] = sharpe_vec(paths)
    wf_mc_cagrs[i_start:i_end] = cagr_vec(paths)
    wf_mc_maxdds[i_start:i_end] = maxdd_vec(paths)

wf_time = time.time() - t0
wf_mc_p = np.mean(wf_mc_sharpes <= 0)
wf_mc_ci = np.percentile(wf_mc_sharpes, [2.5, 97.5])
wf_mc_pct_positive = np.mean(wf_mc_sharpes > 0)

print(f"  Temps total WF         : {wf_time:.1f}s")
print(f"  Sharpe MC-WF median    : {np.median(wf_mc_sharpes):.4f}")
print(f"  Sharpe MC-WF mean      : {np.mean(wf_mc_sharpes):.4f}")
print(f"  Sharpe MC-WF std       : {np.std(wf_mc_sharpes):.4f}")
print(f"  Sharpe MC-WF CI 95%    : [{wf_mc_ci[0]:.4f}, {wf_mc_ci[1]:.4f}]")
print(f"  Sharpe MC-WF CI 99%    : [{np.percentile(wf_mc_sharpes, 0.5):.4f}, {np.percentile(wf_mc_sharpes, 99.5):.4f}]")
print(f"  P(Sharpe WF <= 0)      : {wf_mc_p:.6f}")
print(f"  CAGR MC-WF CI 95%      : [{np.percentile(wf_mc_cagrs, 2.5):.4f}, {np.percentile(wf_mc_cagrs, 97.5):.4f}]")
print(f"  MaxDD MC-WF CI 95%     : [{np.percentile(wf_mc_maxdds, 2.5):.4f}, {np.percentile(wf_mc_maxdds, 97.5):.4f}]")

# Pour le recap / verdict, on utilise le rolling WF % positif
wf_pct_positive = wf_roll_pct_pos


# 11. STRESS TEST — SCENARIOS 6sigma

print(f"\n{'=' * 60}")
print(f"STRESS TEST 6sigma — {STRESS_N_SCENARIOS} scenarios")
print("=" * 60)

mu_daily = strat_ret.mean()
sigma_daily = strat_ret.std(ddof=1)
skewness = skew(strat_ret)
excess_kurt = kurt_fn(strat_ret, fisher=True)

if excess_kurt > 0:
    df_est = max(6.0 / excess_kurt + 4, 4.5)
else:
    df_est = 30.0

t_scale = np.sqrt(df_est / (df_est - 2)) if df_est > 2 else 1.0

print(f"  Stats daily: mu={mu_daily:.6f}, sigma={sigma_daily:.6f}")
print(f"  Skew={skewness:.4f}, ExKurt={excess_kurt:.4f}, df_est={df_est:.1f}, t_scale={t_scale:.4f}")

stress_terminal = np.empty(STRESS_N_SCENARIOS)
stress_maxdds = np.empty(STRESS_N_SCENARIOS)
stress_worst_day = np.empty(STRESS_N_SCENARIOS)
stress_max_consec_loss = np.empty(STRESS_N_SCENARIOS)

for s in tqdm(range(STRESS_N_SCENARIOS), desc="Stress 6sigma"):
    raw_shocks = t_dist.rvs(df=df_est, size=N, random_state=rng.integers(0, 2**31))
    shocks = raw_shocks / t_scale
    path = mu_daily + sigma_daily * shocks

    n_6sigma = rng.integers(1, 7)
    for _ in range(n_6sigma):
        pos = rng.integers(0, N)
        sign = rng.choice([-1, 1])
        path[pos] = mu_daily + sign * STRESS_SIGMA * sigma_daily

    cum = np.cumsum(path)
    stress_terminal[s] = np.exp(cum[-1]) - 1
    dd = cum - np.maximum.accumulate(cum)
    stress_maxdds[s] = np.exp(dd.min()) - 1
    stress_worst_day[s] = path.min()

    losses = (path < 0).astype(int)
    max_consec = 0
    current = 0
    for is_loss in losses:
        if is_loss:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0
    stress_max_consec_loss[s] = max_consec

stress_pnl_terminal = INITIAL_CAPITAL * stress_terminal
pct_ruin = np.mean(stress_terminal < -0.50)
pct_positive_stress = np.mean(stress_terminal > 0)

print(f"\n  Terminal PnL median    : ${np.median(stress_pnl_terminal):,.0f}")
print(f"  Terminal PnL P5/P95   : ${np.percentile(stress_pnl_terminal, 5):,.0f} / ${np.percentile(stress_pnl_terminal, 95):,.0f}")
print(f"  MaxDD median           : {np.median(stress_maxdds):.4f}")
print(f"  MaxDD P1 (worst 1%)   : {np.percentile(stress_maxdds, 1):.4f}")
print(f"  Pire jour median       : {np.median(stress_worst_day):.6f} ({np.median(stress_worst_day)/sigma_daily:.1f}sigma)")
print(f"  Max consec loss P95    : {np.percentile(stress_max_consec_loss, 95):.0f} jours")
print(f"  P(ruine > 50%)        : {pct_ruin:.4f}")
print(f"  P(terminal > 0)       : {pct_positive_stress:.4f}")


# 12. RECAPITULATIF

print(f"\n{'=' * 60}")
print("RECAPITULATIF VALIDATION")
print("=" * 60)

checks = {
    "Sharpe observe":               f"{observed_sharpe:.4f}",
    "MC Sharpe CI 95% (250k)":      f"[{mc_sharpe_ci[0]:.4f}, {mc_sharpe_ci[1]:.4f}]",
    "MC P(Sharpe<=0)":              f"{mc_sharpe_p:.6f}",
    "CPCV Sharpe OOS moyen":        f"{cpcv_mean:.4f}",
    "CPCV P(Sharpe<=0)":            f"{cpcv_p_negative:.4f}",
    "DSR p-value":                  f"{dsr_pval:.4f} {'PASS' if dsr_pass else 'FAIL'}",
    "PBO":                          f"{pbo:.4f} {'PASS' if pbo < 0.5 else 'FAIL'}",
    "CSCV Correl IS<->OOS":        f"{correlation_is_oos:.4f}",
    "WF Rolling %Sharpe>0":         f"{wf_roll_pct_pos:.2%} ({len(wf_rolling_sharpes):,} fenetres)",
    "WF MC Sharpe CI 95% (250k)":   f"[{wf_mc_ci[0]:.4f}, {wf_mc_ci[1]:.4f}]",
    "WF MC P(Sharpe<=0)":           f"{wf_mc_p:.6f}",
    "Stress P(ruine>50%)":          f"{pct_ruin:.4f}",
    "Stress P(terminal>0)":         f"{pct_positive_stress:.4f}",
}

for k, v in checks.items():
    print(f"  {k:<35} : {v}")

pass_count = sum([
    observed_sharpe > 0,
    mc_sharpe_p < 0.10,
    cpcv_mean > 0,
    dsr_pass,
    pbo < 0.50,
    wf_roll_pct_pos > 0.60,
    wf_mc_p < 0.10,
    pct_ruin < 0.10,
])
total_checks = 8
print(f"\n  VERDICT : {pass_count}/{total_checks} checks passes")
if pass_count >= 7:
    print("  >>> STRATEGIE VALIDEE")
elif pass_count >= 5:
    print("  >>> STRATEGIE MARGINALE")
else:
    print("  >>> STRATEGIE NON VALIDEE")


# 13. PLOTS

fig = plt.figure(figsize=(22, 30))
gs = gridspec.GridSpec(10, 2, height_ratios=[2, 1, 1, 1, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
                       hspace=0.35, wspace=0.25)

# Equity
ax0 = plt.subplot(gs[0, :])
ax0.plot(dates, np.cumsum(bh_ret), label="Buy&Hold (log)", color="white", alpha=0.5, lw=1)
ax0.plot(dates, np.cumsum(strat_ret), label="EGARCH VolTarget (log)", color="#00ff00", lw=1.5)
ax0.set_title(f"{TICKER} — Vol Targeting EGARCH(1,1) + AFML 250k Validation — {EVAL_START} -> today", fontsize=13)
ax0.legend(loc="upper left")
ax0.grid(True, alpha=0.1)

# Leverage
ax1 = plt.subplot(gs[1, :], sharex=ax0)
ax1.plot(dates, portfolio["leverage"].values, color="cyan", lw=0.8)
ax1.axhline(1, color="white", ls=":", alpha=0.3)
ax1.set_ylabel("Levier")
ax1.grid(True, alpha=0.1)

# Vol
ax2 = plt.subplot(gs[2, :], sharex=ax0)
ax2.plot(dates, portfolio["egarch_vol_fcast"].values * np.sqrt(252), color="orange", lw=0.8, label="Forecast (ann)")
ax2.plot(dates, portfolio["vol_target"].values * np.sqrt(252), color="dodgerblue", ls="--", lw=0.8, label="Target (ann)")
ax2.set_ylabel("Vol")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.1)

# PnL
ax3 = plt.subplot(gs[3, :], sharex=ax0)
ax3.plot(dates, pnl_bh, color="white", alpha=0.5, lw=1, label="B&H PnL")
ax3.plot(dates, pnl_st, color="#00ff00", lw=1.5, label="VolTarget PnL")
ax3.set_ylabel("PnL ($)")
ax3.legend(loc="upper left")
ax3.grid(True, alpha=0.1)

# MC Sharpe 250k
ax4 = plt.subplot(gs[4, 0])
ax4.hist(mc_sharpes, bins=200, color="steelblue", alpha=0.7, edgecolor="none")
ax4.axvline(observed_sharpe, color="#00ff00", lw=2, label=f"Obs={observed_sharpe:.3f}")
ax4.axvline(0, color="red", ls="--", lw=1)
ax4.set_title(f"MC Sharpe ({MC_ITERATIONS:,} paths)")
ax4.legend()
ax4.grid(True, alpha=0.1)

# CPCV
ax5 = plt.subplot(gs[4, 1])
ax5.hist(cpcv_sharpes, bins=40, color="darkorange", alpha=0.7, edgecolor="none")
ax5.axvline(cpcv_mean, color="#00ff00", lw=2, label=f"Moy={cpcv_mean:.3f}")
ax5.axvline(0, color="red", ls="--", lw=1)
ax5.set_title(f"CPCV Sharpe OOS ({len(cpcv_sharpes)} combos)")
ax5.legend()
ax5.grid(True, alpha=0.1)

# WF Rolling Sharpe time series
ax6 = plt.subplot(gs[5, 0])
wf_dates_rolling = dates[WF_MIN_TRAIN_DAYS:WF_MIN_TRAIN_DAYS + len(wf_rolling_sharpes)]
ax6.plot(wf_dates_rolling, wf_rolling_sharpes, color="cyan", lw=0.5, alpha=0.7)
ax6.axhline(0, color="red", ls="--", lw=0.8)
ax6.axhline(wf_roll_mean, color="#00ff00", ls="-", lw=1, label=f"Mean={wf_roll_mean:.3f}")
ax6.axhline(wf_roll_ci[0], color="cyan", ls=":", lw=0.6, alpha=0.5, label=f"CI95=[{wf_roll_ci[0]:.2f}, {wf_roll_ci[1]:.2f}]")
ax6.axhline(wf_roll_ci[1], color="cyan", ls=":", lw=0.6, alpha=0.5)
ax6.set_title(f"WF Rolling Sharpe ({len(wf_rolling_sharpes):,} fenetres, OOS={WF_OOS_WINDOW}d)")
ax6.legend()
ax6.grid(True, alpha=0.1)

# CSCV IS vs OOS
ax7 = plt.subplot(gs[5, 1])
ax7.scatter(is_sharpes_pbo, oos_sharpes_pbo, alpha=0.3, s=10, color="cyan")
lims = [min(is_sharpes_pbo.min(), oos_sharpes_pbo.min()) - 0.1, max(is_sharpes_pbo.max(), oos_sharpes_pbo.max()) + 0.1]
ax7.plot(lims, lims, "r--", lw=0.8, label="IS=OOS")
ax7.set_xlabel("Sharpe IS")
ax7.set_ylabel("Sharpe OOS")
ax7.set_title(f"CSCV IS<->OOS (corr={correlation_is_oos:.3f}, PBO={pbo:.3f})")
ax7.legend()
ax7.grid(True, alpha=0.1)

# MC-WF Sharpe 250k
ax8 = plt.subplot(gs[6, 0])
ax8.hist(wf_mc_sharpes, bins=200, color="mediumseagreen", alpha=0.7, edgecolor="none")
ax8.axvline(observed_sharpe, color="#00ff00", lw=2, label=f"Obs={observed_sharpe:.3f}")
ax8.axvline(0, color="red", ls="--", lw=1)
ax8.set_title(f"MC-WF Sharpe ({WF_MC_ITERATIONS:,} paths)")
ax8.legend()
ax8.grid(True, alpha=0.1)

# MC-WF MaxDD 250k
ax9 = plt.subplot(gs[6, 1])
ax9.hist(wf_mc_maxdds * 100, bins=200, color="salmon", alpha=0.7, edgecolor="none")
ax9.set_title(f"MC-WF MaxDD ({WF_MC_ITERATIONS:,} paths)")
ax9.set_xlabel("MaxDD (%)")
ax9.grid(True, alpha=0.1)

# Stress PnL
ax10 = plt.subplot(gs[7, 0])
ax10.hist(stress_pnl_terminal / 1000, bins=100, color="mediumpurple", alpha=0.7, edgecolor="none")
ax10.axvline(0, color="red", ls="--", lw=1)
ax10.set_title(f"Stress 6sigma — Terminal PnL ({STRESS_N_SCENARIOS} scenarios)")
ax10.set_xlabel("PnL ($k)")
ax10.grid(True, alpha=0.1)

# Stress MaxDD
ax11 = plt.subplot(gs[7, 1])
ax11.hist(stress_maxdds * 100, bins=100, color="crimson", alpha=0.7, edgecolor="none")
ax11.set_title("Stress 6sigma — Max Drawdown")
ax11.set_xlabel("MaxDD (%)")
ax11.grid(True, alpha=0.1)

# MC vs MC-WF comparison
ax12 = plt.subplot(gs[8, 0])
ax12.hist(mc_sharpes, bins=150, alpha=0.5, color="steelblue", label="MC Standard", density=True, edgecolor="none")
ax12.hist(wf_mc_sharpes, bins=150, alpha=0.5, color="mediumseagreen", label="MC-WF (OOS)", density=True, edgecolor="none")
ax12.axvline(observed_sharpe, color="#00ff00", lw=2)
ax12.axvline(0, color="red", ls="--", lw=1)
ax12.set_title("MC Standard vs MC-WF (densite)")
ax12.legend()
ax12.grid(True, alpha=0.1)

# MC CAGR overlay
ax13 = plt.subplot(gs[8, 1])
ax13.hist(mc_cagrs * 100, bins=150, alpha=0.5, color="steelblue", label="MC CAGR", density=True, edgecolor="none")
ax13.hist(wf_mc_cagrs * 100, bins=150, alpha=0.5, color="mediumseagreen", label="MC-WF CAGR", density=True, edgecolor="none")
ax13.axvline(0, color="red", ls="--", lw=1)
ax13.set_title("CAGR Distribution (%)")
ax13.set_xlabel("CAGR (%)")
ax13.legend()
ax13.grid(True, alpha=0.1)

# Summary table
ax14 = plt.subplot(gs[9, :])
ax14.axis("off")
table_data = [[k, v] for k, v in checks.items()]
tbl = ax14.table(cellText=table_data, colLabels=["Metrique", "Valeur"],
                 loc="center", cellLoc="left", colWidths=[0.50, 0.45])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("gray")
    cell.set_facecolor("#1a1a2e" if row % 2 == 0 else "#16213e")
    cell.set_text_props(color="white")
    if row == 0:
        cell.set_facecolor("#0f3460")
        cell.set_text_props(color="white", fontweight="bold")
ax14.set_title(f"Validation 250k — Verdict: {pass_count}/{total_checks} PASS", fontsize=12, pad=10)

plt.savefig("vol_targeting_egarch_250k.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n[Done] vol_targeting_egarch_250k.png")
